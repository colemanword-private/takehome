# Gemini 2.5 Flash Findings

## Executive summary

I would approve this integration for a monitored production pilot at **24
RPS** for this short-prompt workload — thinking explicitly disabled, a
256-token output bound, and the implemented retry controls — with **12.5×
demonstrated headroom** behind that number.

Two campaigns support this. The single-process campaign confirmed the
operating point (7,200/7,200 requests over five minutes at a realized 24.000
RPS, p95 ≈ 0.84 s) but saturated its own event loop at the 200-RPS target.
The multi-process follow-up (4–8 generator shards) resolved that ambiguity:
Vertex served **200–300 RPS aggregate with zero failures** at ~1.1 s p95,
burst to **600 RPS with 0.15% failures and no rail breach**, and sustained
300 RPS with production retries at **18,000/18,000**. Sustaining 600 RPS
degraded service p95 to 8.9 s while every request still succeeded — at the
edge this system fails through latency inflation, not an error storm, so
capacity alarms must watch p95 drift rather than error rate.

Across 153,000+ live requests there were exactly **two HTTP 429s**: one
terminal at 150 RPS with retries off, one recovered at sustained 600. That
matches Vertex's current **Standard PayGo usage tiers**, where baseline
throughput scales with the organization's rolling 30-day spend and a 429
signals shared-pool contention rather than a fixed quota. Guaranteed capacity
requires Provisioned Throughput — the first thing to price past the pilot.

The model's sharpest quirk: with default thinking and a 256-token cap, hidden
thought tokens consume the output budget — the same golden case failed all
three repeats in **both campaigns**, with ~241 of its 252 output tokens spent
on thoughts. Thinking disabled passed 120/120 across both days at **~11×
lower cost**. The workload costs **$0.065 per 1,000 requests** ($135/day
sustained at 24 RPS).

**No multi-hour soak was run**; the longest sustained observations are five
minutes at 24 RPS and 60 seconds at 300 RPS. A soak remains required before
general production readiness.

## Test environment

Two campaigns, both against Vertex AI location `global`, model
`gemini-2.5-flash`, Python 3.12.8, `google-genai` 2.13.0, `httpx` 0.28.1,
temperature 0, 256-token output cap, thinking budget 0, 20 s per-attempt
timeout, 60 s end-to-end deadline.

| Campaign | Run ID | Generator |
| --- | --- | --- |
| Single-process | `20260817T062702Z` | One event loop, concurrency 96–512 |
| Multi-process | `20260817T155149Z` | 4–8 shard processes × 128 workers, 10-core host |

Each shard of the multi-process generator is the unmodified single-process
harness; an orchestrator (`multi_load.py`) plans per-shard rates and merges
**raw per-request samples** into pooled percentiles (shard percentile
summaries are never averaged). A stage only counts as a provider observation
if every shard's scheduler-lag p95 stayed in single-digit milliseconds; shard
arrival overlap ran 90–99%. Acceptance rails throughout: >1% failures,
service p95 > 5 s, or end-to-end p95 > 5 s.

The workload is five short marketing-assistant questions, SHA-256
`ea8a690cb6b3d4189e403cd279a0628d12e20c2d247963b00c2fd0f2b979799a`. Combined,
the campaigns issued ~167,000 measured live requests for **$10.98 at list
prices** (a lower bound; failed attempts may not return usage metadata).

Thirty-five offline tests pass in ~5 s (provider behavior,
retry/timeout/budget semantics, open-loop scheduling, shard merging and
sibling reaping, campaign isolation, evidence redaction, cost arithmetic),
and a synthetic two-shard run exercises the orchestrator through real
subprocesses. Live
one-request smokes preceded each campaign's billable stages. Developer ADC
emitted Google's no-quota-project warning both days; production identity and
explicit quota ownership remain required.

## Output quality: 2×2 factorial

The factorial varies thinking {model default, 0} and output cap {none, 256},
three repeats per cell, against the ten-case deterministic dataset (v2,
SHA-256 `ce990b305480357f887dc6fdf4cbc5a6e6e596af25a4850818bdf2cff443c6a5`).
The table shows the first campaign.

| Thinking | Output cap | Passed by repeat | Time/run | Output tokens/run | Thought tokens/run |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 256 | 10/10, 10/10, 10/10 | 2.53–2.74 s | 106 | 0 |
| 0 | none | 10/10, 10/10, 10/10 | 2.46–2.50 s | 106–107 | 0 |
| default | 256 | **9/10, 9/10, 9/10** | 4.29–5.10 s | 1,429–1,445 | 1,349–1,365 |
| default | none | 10/10, 9/10, 10/10 | 6.64–7.03 s | 1,864–2,094 | 1,769–2,003 |

Default thinking plus the 256-token cap reproducibly failed
`campaign-summary-001` — 241 of its 252 output tokens were hidden thoughts —
in all three repeats of *both* campaigns, while thinking-off cells passed
120/120 across the two days. (One day-1 default-thinking/no-cap repeat
failed on a transport `ConnectError` — the evaluator failing closed, not a
quality regression; day 2 had no transport errors.) Operational rule: **a
tight `max_output_tokens` requires an explicit thinking budget** — zero for
this workload.

## Multi-process bracket: the 150–300 RPS region, resolved

The original single-process ramp was clean through a realized 150 RPS
(service p95 925 ms) but collapsed at the 200-RPS target with 80-second
scheduler lag — the event loop fell behind, not Vertex. The sharded rerun
(four shards, 30 s stages, retries off) settles what the provider does in
that region; "worst shard sched p95" is the client-cleanliness proof.

| Target RPS | Realized | Overlap | Requests | Failures | Service p50/p95/p99 | Worst shard sched p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 147.5 | 96.5% | 4,500 | 1 (`429`) | 673/1,130/1,794 ms | 1.13 ms |
| 200 | 196.2 | 96.1% | 6,000 | 0 | 649/1,142/1,770 ms | 1.16 ms |
| 250 | 247.7 | 98.1% | 7,500 | 0 | 627/1,110/1,775 ms | 1.11 ms |
| 300 | 296.8 | 97.8% | 9,000 | 0 | 635/1,146/1,911 ms | 2.09 ms |

At 200 RPS the sharded generator measured a 1.14 s service p95 where the
single process had recorded 5.84 s. The terminal 429 — the project's first in
40k requests — arrived at the *lowest* stage, consistent with shared-capacity
contention rather than a rate threshold. Service p95 ran ~1.1 s campaign-wide
versus ~0.85–0.95 s the previous day: between-day provider variance worth
tracking in production, not a function of offered rate.

## Above the bracket: 400–600 RPS, burst and sustained

Eight shards; the 600-RPS ceiling is a deliberate cap on a shared project,
not a technical limit. The first three rows are 30-second probe stages with
retries off; the last two are 60-second sustained runs with production
retries and no rails (observational).

| Target RPS | Design | Realized | Requests | Failures | Retries | Service p50/p95 | Worst shard sched p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 30 s burst | 389.4 | 12,000 | 0 | off | 724/1,332 ms | 9.41 ms |
| 500 | 30 s burst | 489.2 | 15,000 | 27 (23×`499`, 4×`504`) | off | 706/1,361 ms | 1.10 ms |
| 600 | 30 s burst | 570.5 | 18,000 | 27 (18×`499`, 9×`504`) | off | 705/1,417 ms | 1.16 ms |
| 300 | 60 s sustained | 298.5 | 18,000 | 0 | 5 (`504`) | 706/1,214 ms | 2.12 ms |
| 600 | 60 s sustained | 408.0 | 36,000 | 0 | 5 (1×`429`, 4×transport) | 792/**8,864** ms | 37,261 ms |

- **No throttling up to 600 RPS burst** — 25× the pilot — and no 429 at any
  probe rate. Under spend-tiered shared capacity that is good weather, not a
  guarantee; a capacity commitment needs Provisioned Throughput.
- **Vertex returns HTTP 499 (`CANCELLED`) sporadically at ≥500 RPS**, a
  status absent from the retry policy's retryable set (408/429/5xx). At
  ~0.17% it barely matters, but production should decide 499 handling
  deliberately; for these idempotent calls I lean retryable.
- Sustained 300 RPS is the cleanest high-rate evidence in the project:
  end-to-end p95 equals service p95 (no queueing) and all five 504s were
  absorbed by one retry each. Sustained 600 completed all 36,000 requests but
  degraded — service p95 8.9 s, end-to-end p95 42.8 s, 128-worker shards
  saturated, realized rate 408. The same rate was clean for 30 s, so the
  degradation develops under sustained load; with one host I cannot fully
  separate provider pushback from client limits, but either way **beyond
  ~300 RPS sustained this configuration degrades through latency at zero
  error rate**.

## Operating point (original campaign)

Five minutes at 24 RPS with production retries: 7,200/7,200 requests,
realized 24.000 RPS, service p50/p95/p99 of 590/836/1,266 ms, three 504s each
recovered with one retry, peak in-flight 21/96. A 150-RPS retry comparison
(0 vs 4 retries, 4,500 requests each) saw no transient errors in either arm,
so live retry behavior is instead evidenced by the recovered 504s/429 above
and by offline fault injection.

## Cost

List prices as of 2026-08-17: $0.30/M input, $2.50/M output tokens, thinking
tokens billed as output. Lower bounds, computed by `evidence_summary.py` from
observed usage.

| Item | Cost |
| --- | --- |
| Operating point (24 RPS, this workload) | **$0.065 / 1,000 requests** |
| Sustained day at 24 RPS | **$134.61 / day** (~$4k/month) |
| Quality run, thinking disabled | $0.0004 / run (both days) |
| Quality run, default thinking | $0.0044–0.0046 / run (**11–11.5×**) |
| Single-process campaign (40,981 requests) | $2.68 |
| Multi-process campaign (126,121 requests) | $8.30 |

## Design decisions and tradeoffs

- The orchestrator shards the *tested* harness rather than introducing a new
  load tool; only raw samples cross the process boundary, and a failed or
  interrupted run terminates and reaps every sibling shard so nothing keeps
  billing unsupervised.
- Merged reports publish the shard arrival-overlap fraction so aggregate-rate
  claims are measured, not assumed; bounded stage lists and the 600-RPS
  ceiling cap spend and aggression on a shared project (~126k of a
  250k-request budget used).
- Ramp, bracket, and probe keep retries off so raw failure rates stay
  visible; operating-point and sustained runs use production retry settings.
- Committed evidence is generated, redacted, and hashed: per-source SHA-256s
  (including each merged stage's raw shard reports), numeric aggregates,
  retry breakdowns, and effective configuration — no prompts, outputs, error
  messages, or project identifiers. Stage artifacts are never overwritten
  and the summary rejects shard sets that don't match their merged report,
  so a reused RUN_ID cannot mix campaigns. (An earlier revision also
  validated campaign shape; it cost more review surface than the risk it
  defended against and was cut.)

## Limitations

- No multi-hour soak: quota-over-time, credential refresh, connection reuse,
  memory growth, and latency drift beyond 5 minutes are unmeasured.
- Sustained evidence above 24 RPS is 60 seconds long; 30-second stages reveal
  gross saturation, not slow drift.
- All shards ran on one 10-core host and one egress path. The 600-RPS
  sustained degradation is confounded between provider pushback and client
  resources; the 400-RPS stage's 9.4 ms worst-shard scheduler lag suggests
  the host was near its comfortable limit at 8 shards.
- Shared PayGo capacity makes every clean high-rate stage weather-dependent;
  none of these numbers are capacity guarantees, and the usage-tier baseline
  depends on organization spend, which a take-home project does not
  represent.
- Short prompts, small outputs, one model, one region. Larger contexts,
  multimodal inputs, tool use, and streaming were untested; time-to-first-
  token was not measured, which a streaming consumer would need.
- The quality set is small and deterministic — a regression tripwire, not a
  general quality assessment.

## Production next steps

1. Multi-hour soak at 24 RPS (and a shorter one at ~100 RPS), watching p95
   drift, 429/499/5xx rates, retry amplification, credentials, connections,
   and memory.
2. Price **Provisioned Throughput** against measured token throughput before
   any capacity commitment; spend-tiered shared capacity is fine for a pilot,
   not for an SLO.
3. Decide 499 handling explicitly, and alarm on latency inflation — at the
   observed edge, p95 drifts while error rate stays zero, so error-rate-only
   alerting would miss it.
4. Enforce the rule that a tight `max_output_tokens` requires an explicit
   thinking budget.
5. Replace developer ADC with production identity and explicit quota/billing
   ownership; correlate client request IDs with server-side telemetry.
6. Distribute the load generator before claiming anything above 600 RPS;
   expand quality coverage before reusing these conclusions on other workload
   shapes.

## Reproduction and artifacts

`make bootstrap test` runs everything offline. The live targets, each taking
`RUN_ID=<id>`: `provider-smoke`, `quality-factorial`, `capacity-ramp`,
`operating-point`, and `retry-comparison` for the single-process campaign;
`provider-smoke`, `sharded-bracket`, `quota-probe`, and `retries-demo` (at
`DEMO_RPS=600` then `300`) for the multi-process campaign; then
`make evidence RUN_ID=<id>`. Makefile defaults match the tested
configuration.

Raw reports, including per-shard samples, stay gitignored under
`load-results/` and `eval-results/`. The committed evidence —
`evidence/gemini/gemini-2.5-flash/<run-id>-summary.{json,md}` — is the
authoritative numeric source for this write-up: aggregates, retry breakdowns,
configuration, and SHA-256s for every raw source, with no prompts, outputs,
or project identifiers.
