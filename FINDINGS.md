# Gemini 2.5 Flash Findings

## Executive summary

I would approve this integration for a limited production pilot at up to 6 RPS
for this workload, with the current output bound and monitoring enabled. I would
not yet claim the model's absolute capacity or approve an unrestricted rollout:
the highest tested rate, 10 RPS, remained healthy, so the capacity knee was not
reached, and I deliberately omitted a long soak to conserve take-home project
credits.

Within the tested envelope, Gemini 2.5 Flash completed 1,411 measured load
requests and 20 quality cases plus 35 warmups. All measured requests succeeded
with zero recorded retries, all warmups completed successfully, client-side
queue delay stayed low, and median service latency remained stable. In one paired
quality run, the controlled configuration—thinking budget 0 plus a 256-token
output cap—preserved a 10/10 score while using 95% fewer observed output tokens
and completing 59% faster than default thinking/output controls.

## Test environment

The campaign ran on 2026-08-16 under run ID `20260816T230754Z`.

| Setting | Value |
| --- | --- |
| Provider | Google Vertex AI |
| Project | `evertune-tests` |
| Location | `global` |
| Model | `gemini-2.5-flash` |
| Python | 3.12.8 |
| `google-genai` | 2.13.0 |
| `httpx` | 0.28.1 |
| Load temperature | 0 |
| Capacity/retry output limit | 256 tokens |
| Capacity/retry thinking budget | 0 |
| Per-attempt timeout | 60 seconds |
| Capacity-stage retries | 0 |
| Capacity/retry harness concurrency | 64 |

The table above describes the retained `20260816T230754Z` campaign. The
implementation was hardened after that campaign: both registered providers now
default to a 20-second attempt timeout inside a 60-second end-to-end request
deadline, honor provider `Retry-After` guidance, and share a token-bucket retry
budget across requests. The load harness now bounds pending arrivals at four
times configured concurrency by default. These controls are covered by offline
fault-injection tests, but the retained live campaign predates them and should
be rerun before a production decision relies on the new behavior.

The open-loop workload contains five short marketing-assistant questions and a
stable SHA-256 fingerprint:
`ea8a690cb6b3d4189e403cd279a0628d12e20c2d247963b00c2fd0f2b979799a`.
The harness schedules arrivals independently of response completion, so queue
delay reveals when the configured client concurrency can no longer sustain the
offered rate.

To limit credit usage while still observing a progression, I used 30-second
capacity stages at 1, 3, 5, 8, and 10 RPS instead of the Makefile's 60-second
default. I then compared retries off and on for another 30 seconds each at the
highest tested rate. No soak test was run.

## Offline verification

- At campaign time, all 88 unit tests passed. The current hardened implementation
  has 123 passing offline tests; they use fake provider clients and make no live
  LLM calls.
- The synthetic harness completed 10,000 burst-scheduled requests with zero
  failures. Synthetic service latency was p50 1.06 ms and p95 1.22 ms. This
  validates the scheduler and metrics path, not Vertex capacity.
- The one-request Vertex smoke test succeeded with no retry. Its 1.96-second
  latency may include cold connection/authentication effects and is not treated
  as a steady-state performance estimate.

## Output quality

The version 2 golden dataset contains 10 deterministic cases across
classification, structured output, summarization, instruction following, and
grounded refusal. Dataset SHA-256:
`ce990b305480357f887dc6fdf4cbc5a6e6e596af25a4850818bdf2cff443c6a5`.

| Configuration | Passed | Total time | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Default thinking/output controls; temperature 0; retries off | 10/10 | 8.92 s | 433 | 2,135 |
| Thinking 0, max output 256, retries off | 10/10 | 3.66 s | 433 | 106 |

Both configurations passed every category. In this one paired run, the
controlled configuration used 95.0% fewer observed output tokens and completed
58.9% faster. It changed two variables simultaneously and was not repeated, so
the experiment does not isolate the thinking budget from the output cap or
run-to-run variance. The raw artifact also does not separate candidate and
thought token counts.

This is directional evidence that explicit controls are valuable for short,
deterministic tasks, not proof that thinking alone caused the improvement or
should always be disabled. A follow-up factorial experiment should vary thinking
and the output cap independently, repeat each cell, and include more ambiguous
analysis and planning tasks.

## Capacity ramp

Each stage used five warmups, retries off, thinking disabled, a 256-token output
limit, and a concurrency cap of 64.

| Offered RPS | Requests | Success | Achieved RPS | Service p50 | Service p95 | Service p99 | Queue p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 100% | 1.01 | 634 ms | 1,897 ms | 2,411 ms | 2.18 ms |
| 3 | 90 | 100% | 2.95 | 608 ms | 1,628 ms | 3,846 ms | 4.53 ms |
| 5 | 150 | 100% | 4.85 | 611 ms | 1,163 ms | 1,791 ms | 7.01 ms |
| 8 | 240 | 100% | 7.86 | 626 ms | 1,067 ms | 1,717 ms | 6.14 ms |
| 10 | 300 | 100% | 9.75 | 611 ms | 1,241 ms | 1,530 ms | 4.23 ms |

There was no saturation signal through 10 RPS: no request failed, no 429 or 5xx
response occurred, p50 service latency stayed near 0.6 seconds, and queue-delay
p95 stayed below 8 ms. The non-monotonic tail latency is consistent with normal
run-to-run and workload variance; the short low-rate stages contain relatively
few samples, so their p95/p99 values should not be over-interpreted.

The capacity knee is therefore greater than 10 RPS for this specific short-prompt
workload and test window. It was not measured. A provisional 6 RPS pilot limit
keeps 40% distance below the highest validated rate, but that is operational
conservatism rather than proven 40% headroom below actual capacity.

## Retry comparison

The retry experiment used 300 measured requests at 10 RPS per configuration,
plus five warmups each.

| Max retries | Success | Observed retries | Service p50 | Service p95 | Service p99 | Queue p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100% | 0 | 623 ms | 1,042 ms | 2,205 ms | 5.32 ms |
| 4 | 100% | 0 | 573 ms | 1,068 ms | 1,619 ms | 5.90 ms |

No transient error was recorded during the measured phase, so the retry policy
had no opportunity to improve measured availability and added no measured
attempts. The harness clears retry telemetry after warmup; all warmups completed
successfully, but retry activity during the five retry-enabled warmups was not
retained. The latency differences are ordinary between-run variance, not a
retry benefit. Unit tests separately verify the bounded exponential full-jitter
schedule, retryable status classification, retry exhaustion, cancellation, and
retry telemetry.

## Failure modes and operational observations

No measured live request produced a timeout, rate limit, 5xx, provider exception,
or empty-text response during this campaign. That is encouraging but does not
demonstrate those failure modes are absent; it means they were not observed in
the retained measured-phase telemetry. Content correctness outside the ten-case
quality set was not evaluated.

Observed token totals are a billing lower bound. A provider may accept and bill
an attempt that later times out locally without returning usage metadata. The
harness records this limitation rather than presenting incomplete token counts
as exact cost.

## Design decisions and tradeoffs

- One reusable async SDK client shares authentication state and the HTTPX
  connection pool across concurrent requests.
- The provider targets Vertex's stable `v1` API and uses an explicit per-attempt
  deadline.
- System instructions remain separate from user content so Gemini receives the
  intended message roles.
- The shared retry engine owns timing and telemetry, while each provider owns
  error classification. This avoids duplicating backoff logic without pretending
  provider error semantics are universal.
- Both registered providers configure the shared retry engine with a total
  request deadline across attempts and backoff plus a shared token bucket, so
  concurrent failures cannot multiply into an unbounded retry storm. Each
  provider instance owns one process-local budget; separately deployed workers
  have independent buckets. Both providers also honor server retry delays and
  record terminal exhaustion reasons.
- The load harness bounds pending arrivals while retaining their original
  scheduled timestamps, so overload produces visible queue delay without
  unbounded pending-work growth.
- Retries were disabled during the capacity ramp so they could not hide raw
  failure rates or amplify overload.
- Thinking tokens count as output usage because they affect capacity and cost
  even though they are absent from the visible answer.
- Quality and load tooling resolve providers through a registry, allowing the
  same datasets, workloads, Make targets, and result schema to exercise Gemini,
  Together, or another registered implementation.
- Results are local timestamped JSON rather than managed Vertex Experiments.
  That keeps the take-home focused and reproducible without adding cloud-side
  experiment lifecycle complexity. Managed experiment tracking would become
  valuable for a larger, recurring evaluation program.

## Limitations

- Thirty-second stages reveal gross errors and queue growth but not long-term
  quota behavior, regional variance, credential refresh, memory leaks, or daily
  traffic cycles.
- The ramp stopped at 10 RPS before finding a failure or latency knee.
- The workload uses five short questions. Larger contexts, long outputs,
  multimodal inputs, and tool calls could have materially different capacity and
  latency.
- The deterministic quality set is intentionally small. It does not measure
  nuanced helpfulness, broad marketing knowledge, or rare safety behavior.
- Tests ran from one client and one endpoint location. They do not measure
  multi-region behavior or contention from other tenants.
- No direct billing export or server-side quota telemetry was correlated with
  the client metrics.

## Production next steps

Before an unrestricted production rollout, I would:

1. Extend the ramp beyond 10 RPS until a latency, quota, or error knee is
   observed, then choose an operating point against that measured boundary.
2. Run a multi-hour soak at the chosen rate with realistic prompt and output-size
   distributions.
3. Add dashboards and alerts for p95/p99 latency, queue delay, 429/5xx rates,
   retry amplification, token usage, empty/safety-blocked responses, and cost.
4. Replace developer ADC with production identity and configure explicit quota
   ownership.
5. Expand quality coverage for ambiguous analytical work and test whether a
   nonzero thinking budget improves those cases enough to justify its cost.
6. Add adversarial, safety, large-context, cancellation, and regional-failover
   experiments.
7. Correlate client request IDs with Vertex quota and billing telemetry so failed
   or timed-out billed attempts are visible.

## Reproduction and artifacts

The campaign used these commands; the soak target was intentionally omitted:

```bash
make test synthetic-smoke RUN_ID=20260816T230754Z

make provider-smoke \
  RUN_ID=20260816T230754Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=128

make quality \
  RUN_ID=20260816T230754Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=256

make capacity-ramp \
  RUN_ID=20260816T230754Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=256 \
  RAMP_RPS="1 3 5 8 10" RAMP_DURATION_SECONDS=30 RAMP_CONCURRENCY=64

make retry-off retry-on \
  RUN_ID=20260816T230754Z \
  PROVIDER=gemini MODEL=gemini-2.5-flash \
  THINKING_BUDGET=0 MAX_OUTPUT_TOKENS=256 \
  RETRY_RPS=10 RETRY_DURATION_SECONDS=30 RETRY_CONCURRENCY=64 \
  PRODUCTION_RETRIES=4
```

Raw reports are stored locally under:

- `load-results/gemini/gemini-2.5-flash/20260816T230754Z/`
- `load-results/synthetic/local/20260816T230754Z/`
- `eval-results/gemini/gemini-2.5-flash/20260816T230754Z/`

Those directories are intentionally gitignored because reports may contain model
outputs. A sanitized aggregate manifest with raw-artifact SHA-256 checksums is
stored at
`evidence/gemini/gemini-2.5-flash/20260816T230754Z-summary.json`; it contains no
prompts, generated outputs, error messages, credentials, or project identifier.
The manifest and tables above are the reviewed evidence intended for the Git
submission. Raw reports can be supplied separately if independent output-level
inspection is required.
