# Gemini 2.5 Flash Findings

## Executive summary

I would approve this integration for a production pilot at up to 24 RPS for
this workload, with thinking explicitly disabled, the current output bound,
and monitoring enabled. The highest tested rate, 40 RPS, stayed healthy —
5,611 measured live requests produced exactly one failure (a single HTTP 504,
0.08% of its stage) — so the hard capacity knee was still not reached, though
tail-latency pressure is now visible from 15 RPS upward. **No soak test was
run: it was skipped deliberately to save credit usage, and it remains
necessary before production readiness** (sustained-rate behavior, credential
refresh, connection reuse, and memory over hours are unmeasured).

Two findings stand out beyond raw capacity. First, a factorial quality
experiment found a reproducible bad interaction: with the model's default
thinking enabled and a 256-token output cap, one golden case failed in all
three repeats because thinking consumed ~1,360 tokens and starved the visible
answer, while the same cap with thinking disabled passed 10/10 — explicit
`thinking_budget=0` is not just cheaper, it is required for correctness under
tight output caps. Second, the failure-handling path is now demonstrated
rather than assumed: focused offline fault injection drives mixed 429/503 and
stalled requests through the provider and load harness, while provider tests
cover blocked empty responses. The campaign's single live 504 was correctly
classified in the report, which the previous classifier would have mislabeled.

## Changes since the prior campaign

This campaign re-measured the system after substantial hardening, so numbers
are not directly comparable with the earlier `20260816T230754Z` run:

- Both providers now run a 20-second attempt timeout inside a 60-second
  end-to-end request deadline, honor server `Retry-After`/`retry-after-ms`
  directives (as a floor over jittered backoff), and share a token-bucket
  retry budget per provider instance so concurrent failures cannot multiply
  into a retry storm.
- HTTP status classification in the load report was fixed (google-genai errors
  carry a string `status`; the numeric `code` is now preferred) — proven live
  by the correctly bucketed `http_504`.
- The Gemini transport pool is sized to the configured parallelism instead of
  httpx's 20-keepalive default, and this campaign aligned provider parallelism
  with harness concurrency (both 96).
- Quality validators were made word-bounded and sign-normalizing, so correct
  answers containing abbreviations ("vs.") or signed percentages ("+12%") no
  longer fail; pass criteria therefore differ slightly from the prior run.
- The capacity ramp gained a measured-request budget and abort thresholds
  (stop at the first stage with >1% failures or p95 above 5 s), and warmup
  blips no longer abort a healthy measured campaign.
- Artifacts now separate hidden thinking tokens from visible answer tokens.

## Test environment

The campaign ran on 2026-08-17 under run ID `20260817T034759Z`.

| Setting | Value |
| --- | --- |
| Provider | Google Vertex AI |
| Location | `global` |
| Model | `gemini-2.5-flash` |
| Python | 3.12.8 |
| `google-genai` | 2.13.0 |
| `httpx` | 0.28.1 |
| Load temperature | 0 |
| Load output limit | 256 tokens |
| Load thinking budget | 0 |
| Per-attempt timeout | 20 seconds |
| End-to-end request deadline | 60 seconds |
| Retry budget | 32 tokens, refilling 2/second (per provider instance) |
| Ramp/retry harness concurrency | 96 (provider parallelism aligned at 96) |
| Ramp request budget | 4,000 measured (3,210 used) |
| Ramp abort thresholds | >1% stage failures or service p95 > 5,000 ms |

The open-loop workload is unchanged from the prior campaign: five short
marketing-assistant questions, SHA-256
`ea8a690cb6b3d4189e403cd279a0628d12e20c2d247963b00c2fd0f2b979799a`. The
harness schedules arrivals independently of completions and bounds pending
arrivals at four times the configured concurrency while retaining original
scheduled timestamps, so overload appears as visible queue delay rather than
unbounded pending work. Campaign-wide, the live runs consumed 275,608 input
tokens and 124,920 output tokens (9,866 of them hidden thinking, almost
entirely from the factorial's default-thinking cells).

## Offline verification

- All 119 offline tests passed. They use fake provider clients and make no
  live LLM calls.
- Focused fault injection drives the real provider, retry engine, and load
  harness with scripted faults offline: a stalled request is bounded by the
  per-attempt timeout, and a mixed 503/429 sequence honors `Retry-After: 3`
  while verifying attempt counts, per-status retries, and cumulative backoff
  end-to-end. A separate load-harness integration test verifies nonzero retry
  exhaustion aggregation.
- The synthetic harness completed 10,000 burst-scheduled requests with zero
  failures (service p50 1.09 ms, p95 1.42 ms), validating the scheduler and
  metrics path.
- The one-request Vertex smoke test succeeded with no retry at 1.41 s,
  including cold connection and authentication effects.

## Output quality: factorial experiment

The prior campaign's single paired comparison changed two variables at once.
This campaign ran the full 2×2 factorial — thinking {model default, 0} ×
output cap {none, 256} — with three repeats per cell against the version 2
golden dataset (10 deterministic cases, SHA-256
`ce990b305480357f887dc6fdf4cbc5a6e6e596af25a4850818bdf2cff443c6a5`).

| Thinking | Output cap | Passed (3 repeats) | Time/run | Output tokens/run | Thought tokens/run |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 256 | 10/10, 10/10, 10/10 | ~3.4 s | 106 | 0 |
| 0 | none | 10/10, 10/10, 10/10 | ~3.6 s | 96–106 | 0 |
| default | none | 10/10, 10/10, 10/10 | ~9.6 s | 1,862–2,142 | 1,767–2,047 |
| default | 256 | **9/10, 9/10, 9/10** | ~8.1 s | 1,439–1,457 | 1,359–1,377 |

Findings:

- **Thinking budget dominates cost and latency.** Disabling thinking cut
  output tokens ~20× and runtime ~2.7× with no quality loss on this dataset,
  across all repeats — confirming the prior campaign's directional result
  with the confound removed.
- **Default thinking plus a tight output cap is actively unsafe.** The
  `campaign-summary-001` case failed in all three repeats of the
  default-thinking/256-cap cell: the model spent ~1,360 tokens thinking and
  the visible answer was truncated before its percentage facts (the validator
  saw no percentages at all). The same cap with thinking disabled passed
  every repeat. Observed thought tokens exceeded the cap while visible tokens
  stayed far under it, so the exact capping semantics between thoughts and
  candidates are murky — the operational rule is simple regardless: when
  capping output, set an explicit thinking budget.
- Run-to-run variance within cells was small (token counts within ±8%,
  identical pass/fail outcomes), so three repeats were sufficient for these
  effects.

## Capacity ramp

Eight 30-second stages, five warmups each, retries off, thinking 0, 256-token
cap, concurrency 96, under a 4,000-request budget with 1%-failure/5 s-p95
abort rails.

| Offered RPS | Requests | Success | Achieved RPS | Service p50 | Service p95 | Service p99 | Queue p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 100% | 1.00 | 567 ms | 963 ms | 1,027 ms | 4.53 ms |
| 3 | 90 | 100% | 2.98 | 603 ms | 920 ms | 1,375 ms | 5.65 ms |
| 5 | 150 | 100% | 4.92 | 585 ms | 787 ms | 1,139 ms | 4.20 ms |
| 8 | 240 | 100% | 7.77 | 610 ms | 1,106 ms | 1,833 ms | 5.09 ms |
| 10 | 300 | 100% | 9.70 | 622 ms | 1,166 ms | 1,544 ms | 6.37 ms |
| 15 | 450 | 100% | 13.74 | 644 ms | 1,851 ms | 2,601 ms | 5.05 ms |
| 25 | 750 | 100% | 22.37 | 717 ms | 2,800 ms | 4,069 ms | 5.95 ms |
| 40 | 1,200 | 99.9% | 31.98 | 614 ms | 1,492 ms | 2,104 ms | 6.84 ms |

Observations:

- **No hard knee through 40 RPS.** No 429 appeared at any stage. The single
  failure across 3,210 ramp requests was one `http_504` at 40 RPS (0.08% of
  the stage) that took 17.7 s server-side — under the 1% threshold, so the
  rail correctly let the run complete. The previous any-failure rule would
  have aborted the campaign on that one request.
- **Tail pressure begins around 15–25 RPS.** p95 climbed from ~1.1 s (10 RPS)
  to 1.9 s (15) and 2.8 s (25) while p50 stayed near 0.6 s; the 40 RPS
  stage's lower p95 (1.5 s) shows substantial between-stage variance, so the
  25 RPS tail should be read as pressure, not a wall.
- Queue-delay p95 stayed below 7 ms everywhere, so the client was never the
  bottleneck; "achieved RPS" dilution at higher stages is dominated by the
  fixed 30 s schedule plus tail-drain arithmetic (the 504's 17.7 s stretched
  the 40 RPS stage's window to 37.5 s).
- The capacity knee is therefore greater than 40 RPS for this short-prompt
  workload. A provisional 24 RPS pilot keeps 40% distance below the highest
  validated rate — operational conservatism, not measured headroom.

## Retry comparison

The comparison ran at the highest healthy ramp rate: 1,200 measured requests
per configuration at 40 RPS (five warmups each), thinking 0, 256-token cap,
concurrency 96.

| Max retries | Success | Observed retries | Service p50 | Service p95 | Service p99 | Queue p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100% | 0 | 566 ms | 809 ms | 1,283 ms | 5.45 ms |
| 4 | 100% | 0 | 560 ms | 874 ms | 1,480 ms | 5.29 ms |

No transient error occurred during either measured phase, so the retry policy
again had no opportunity to fire live; the latency differences are between-run
variance. Unlike the prior campaign, however, the retry path is no longer
unexercised: focused end-to-end fault injection covers Retry-After and timeout
behavior against scripted 429/503/stall faults, while shared-engine tests cover
budget and deadline invariants under concurrency.

## Failure modes and operational observations

- One live failure in 5,611 measured requests: a single HTTP 504 at 40 RPS,
  correctly classified as `http_504` in the stage report, with a 17.7 s
  server-side duration. No 429, no 5xx bursts, and no empty-text responses
  were observed live.
- The default-thinking + tight-cap quality failure (above) is the most
  actionable failure mode found: it is silent (HTTP success, well-formed
  response, wrong or empty content) and reproducible.
- Observed token totals remain a billing lower bound: an attempt that times
  out locally may still be billed without returning usage metadata.
- Thinking tokens are now reported separately in every artifact
  (`observed_thought_tokens`) while still counting toward output usage.

## Design decisions and tradeoffs

- One reusable async SDK client per provider; the httpx connection pool is
  sized to the configured parallelism (this campaign: 96) so keepalive churn
  cannot distort tail latencies at high concurrency.
- The shared retry engine owns timing, budgets, deadlines, and telemetry;
  providers own error classification. Server `Retry-After` directives floor
  the jittered delay rather than replacing the policy, and each provider
  instance carries one process-local retry budget (separately deployed
  workers have independent buckets).
- Retries stayed disabled during the ramp so raw failure rates are visible;
  the ramp additionally carries a measured-request budget and abort
  thresholds so a knee search cannot run away on cost or keep hammering a
  degrading service. Warmup failures are reported but no longer abort a
  campaign whose measured phase is healthy.
- The load harness bounds pending arrivals (four times concurrency by
  default) while retaining scheduled timestamps, so overload produces visible
  queue delay without unbounded pending-work growth.
- Provider capabilities are declared in the registry, so unsupported controls
  (for example `THINKING_BUDGET` with the Together provider) fail as clean
  usage errors before any billable work.
- Quality and load tooling resolve providers through the registry; the
  factorial experiment, datasets, workloads, Make targets, and result schema
  work for any registered implementation.
- Evidence is generated by `evidence_manifest.py` (`make evidence`) rather
  than hand-authored: it aggregates the run's artifacts into a sanitized,
  SHA-256-checksummed manifest with no prompts, outputs, error messages,
  credentials, or project identifier.

## Limitations

- **No soak test was run — skipped to save credit usage.** A multi-hour soak
  at the pilot rate is necessary before production readiness: quota behavior
  over time, credential refresh, connection reuse, memory growth, and daily
  traffic cycles are all unmeasured.
- The ramp stopped at 40 RPS (a budget-bounded design choice) without
  reaching a failure or latency knee; capacity beyond that is unknown.
- Thirty-second stages reveal gross errors and queue growth, not sustained
  behavior; the 25-vs-40 RPS tail inversion shows meaningful between-stage
  variance at this stage length.
- The workload is five short prompts; larger contexts, long outputs,
  multimodal inputs, and tool calls could behave very differently.
- The deterministic quality set is small and unambiguous; the factorial's
  thinking result may not transfer to genuinely hard analytical tasks, which
  is exactly where a nonzero thinking budget would earn its cost.
- Tests ran from one client and one endpoint location; no billing export or
  server-side quota telemetry was correlated with client metrics.

## Production next steps

1. **Run the soak** (deliberately omitted here to conserve credits): multiple
   hours at the chosen pilot rate with realistic prompt and output-size
   distributions, watching latency drift, quota events, credential refresh,
   and memory.
2. Extend the ramp beyond 40 RPS under a raised budget until a latency,
   quota, or error knee is actually observed, then set the operating point
   against that measured boundary.
3. Enforce a configuration rule that a tight `max_output_tokens` requires an
   explicit `thinking_budget`, given the reproducible silent quality failure
   when both defaults collide.
4. Add dashboards and alerts for p95/p99 latency, queue delay, 429/5xx/504
   rates, retry amplification and exhaustions, thought-vs-answer token mix,
   and cost.
5. Replace developer ADC with production identity and explicit quota
   ownership.
6. Expand quality coverage with ambiguous analytical tasks to test whether a
   nonzero thinking budget pays for itself where reasoning is genuinely hard.
7. Correlate client request IDs with Vertex quota and billing telemetry so
   billed-but-failed attempts become visible.

## Reproduction and artifacts

The campaign used these commands; the soak target was intentionally omitted:

```bash
make test synthetic-smoke RUN_ID=20260817T034759Z

make provider-smoke \
  RUN_ID=20260817T034759Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=128

make quality-factorial \
  RUN_ID=20260817T034759Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  FACTORIAL_REPEATS=3

GEMINI_PARALLELISM=96 make capacity-ramp \
  RUN_ID=20260817T034759Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=256 \
  RAMP_RPS="1 3 5 8 10 15 25 40" RAMP_DURATION_SECONDS=30 \
  RAMP_CONCURRENCY=96 RAMP_REQUEST_BUDGET=4000

GEMINI_PARALLELISM=96 make retry-off retry-on \
  RUN_ID=20260817T034759Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=256 \
  RETRY_RPS=40 RETRY_DURATION_SECONDS=30 RETRY_CONCURRENCY=96 \
  PRODUCTION_RETRIES=4

make evidence RUN_ID=20260817T034759Z PROVIDER=gemini MODEL=gemini-2.5-flash
```

Raw reports are stored locally under:

- `load-results/gemini/gemini-2.5-flash/20260817T034759Z/`
- `load-results/synthetic/local/20260817T034759Z/`
- `eval-results/gemini/gemini-2.5-flash/20260817T034759Z/`

Those directories are intentionally gitignored because reports may contain
model outputs. The sanitized aggregate manifest with raw-artifact SHA-256
checksums is generated by `make evidence` and stored at
`evidence/gemini/gemini-2.5-flash/20260817T034759Z-summary.json`; it contains
no prompts, generated outputs, error messages, credentials, or project
identifier. The manifest and the tables above are the reviewed evidence
intended for the Git submission; raw reports can be supplied separately if
independent output-level inspection is required.
