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

The multi-process generator is simply N copies of the single-process harness
running as separate OS processes, each pacing an equal share of the target
rate; `multi_load.py` only launches them and merges their results. The merge
pools the **raw per-request timings** from every shard and recomputes
percentiles over the combined set. Every stage ran under the same
acceptance rails: fail if more than 1% of requests fail, or if service or
end-to-end p95 exceeds 5 s.

The workload is five short marketing-assistant questions. Combined,
the campaigns issued ~167,000 measured live requests for **$10.98 at list
prices** (a lower bound; failed attempts may not return usage metadata).

Thirty-five offline tests pass in ~5 s (provider behavior,
retry/timeout/budget semantics, open-loop scheduling, shard merging and
sibling reaping, campaign isolation, evidence redaction, cost arithmetic),
and a synthetic two-shard run exercises the orchestrator through real
subprocesses. Live one-request smokes preceded each campaign's billable stages.

## Output quality: 2×2 factorial

The quality experiment tests all four combinations of two settings —
thinking (model default vs. explicitly disabled) and output cap (none vs.
256 tokens) — because problems can hide in the interaction between settings
rather than in either one alone. Each combination ran three times, so a
failure must reproduce before it counts as real. Every run scores the same
ten fixed prompts with mechanical pass/fail validators at temperature 0 (no
human or LLM judging; dataset v2, SHA-256
`ce990b305480357f887dc6fdf4cbc5a6e6e596af25a4850818bdf2cff443c6a5`). The
table shows the first campaign, one row per configuration:

- **Thinking** — the `thinking_budget` setting: `0` disables the model's
  hidden reasoning entirely; `default` lets the model decide how much to
  reason before answering.
- **Output cap** — the `max_output_tokens` limit: `none` leaves it unset,
  `256` hard-caps the response. Hidden thought tokens count against this
  cap — that is the interaction under test.
- **Passed by repeat** — cases passed out of ten, listed separately for
  each of the configuration's three repeats.
- **Time/run** — wall-clock time to score all ten cases once (the range
  across the three repeats).
- **Output tokens/run** — billable output tokens for one full ten-case run,
  including hidden thought tokens.
- **Thought tokens/run** — the hidden-reasoning share of those output
  tokens: never visible in any answer, but billed at the output rate.

| Thinking | Output cap | Passed by repeat | Time/run | Output tokens/run | Thought tokens/run |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 256 | 10/10, 10/10, 10/10 | 2.53–2.74 s | 106 | 0 |
| 0 | none | 10/10, 10/10, 10/10 | 2.46–2.50 s | 106–107 | 0 |
| default | 256 | **9/10, 9/10, 9/10** | 4.29–5.10 s | 1,429–1,445 | 1,349–1,365 |
| default | none | 10/10, 9/10, 10/10 | 6.64–7.03 s | 1,864–2,094 | 1,769–2,003 |

With default thinking and the 256-token cap, the same test case
(`campaign-summary-001`) failed in all six repeats across both days. The
token counts show why: the model spent 241 of the case's 252 output tokens
on hidden reasoning, leaving too little budget for the visible answer, which
then omitted required facts. With thinking disabled, every case passed —
120 of 120 across the two days — at a fraction of the token cost. One
run (default thinking, no cap) failed on a network error rather than a
wrong answer; the evaluator deliberately counts such errors as failures
instead of hiding them, so that entry reflects transport flakiness, not
model quality. The operational rule that falls out: **never set a tight
`max_output_tokens` without also setting an explicit thinking budget** —
zero for this workload.

## Multi-process bracket: the 150–300 RPS region, resolved

The original single-process ramp was clean through a delivered 150 RPS
(service p95 925 ms) but collapsed at the 200-RPS target with 80-second
scheduler lag — the event loop fell behind, not Vertex. The sharded rerun
(four shards, 30 s stages, retries off) settles what the provider does in
that region.

Reading the table — each row is one 30-second stage, pooled across all four
shards:

- **Offered RPS** — the arrival schedule the generator was configured to
  produce for that stage.
- **Delivered RPS** — the arrival rate it measurably emitted. Offered and
  Delivered must roughly match for a row to say anything about Vertex; a
  large gap means the generator, not the provider, was the limit.
- **Overlap** — the fraction of the stage during which all four shards were
  emitting simultaneously. Shards are separate processes that start moments
  apart, and the aggregate rate only exists while all of them are running;
  high overlap means the full rate was actually offered for essentially the
  whole stage.
- **Requests / Failures** — completed requests across all shards, and how
  many ended in an error (with the HTTP status behind each failure).
- **Service p50/p95/p99** — provider call latency (request sent to full
  response received, excluding any client-side queueing), as percentiles
  over the pooled raw samples of every request in the stage.
- **Worst shard sched p95** — the slowest shard's scheduler lag: how late
  requests left the generator relative to their ideal schedule. Single-digit
  milliseconds proves every shard kept pace, so the latency columns describe
  the provider rather than an overloaded client.

| Offered RPS | Delivered RPS | Overlap | Requests | Failures | Service p50/p95/p99 | Worst shard sched p95 |
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

Eight shards this time; the 600-RPS ceiling is a deliberate cap on a shared
project, not a technical limit. Each row is one stage, pooled across all
eight shards:

- **Offered RPS, Delivered RPS, Requests, Failures, Worst shard sched p95**
  — same meanings as in the bracket table.
- **Design** — the stage's shape: "burst" rows are 30-second probe stages
  with retries disabled so raw failure rates stay visible; "sustained" rows
  are 60-second runs under the production retry policy, with no acceptance
  rails (observational).
- **Retries** — retry attempts made under that policy ("off" where retries
  were disabled). In both sustained rows every retried request recovered on
  its first retry, which is why Failures stays zero.
- **Service p50/p95** — as in the bracket table; p99 omitted for width.

| Offered RPS | Design | Delivered RPS | Requests | Failures | Retries | Service p50/p95 | Worst shard sched p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 30 s burst | 389.4 | 12,000 | 0 | off | 724/1,332 ms | 9.41 ms |
| 500 | 30 s burst | 489.2 | 15,000 | 27 (23×`499`, 4×`504`) | off | 706/1,361 ms | 1.10 ms |
| 600 | 30 s burst | 570.5 | 18,000 | 27 (18×`499`, 9×`504`) | off | 705/1,417 ms | 1.16 ms |
| 300 | 60 s sustained | 298.5 | 18,000 | 0 | 5 (`504`) | 706/1,214 ms | 2.12 ms |
| 600 | 60 s sustained | 408.0 | 36,000 | 0 | 5 (1×`429`, 4×transport) | 792/**8,864** ms | 37,261 ms |

- **No throttling up to 600 RPS burst** — 25× the pilot — and no 429 at any
  probe rate. Because capacity is shared and spend-tiered, that reflects
  favorable conditions during the test, not a guarantee; a capacity
  commitment needs Provisioned Throughput.
- **Vertex returns HTTP 499 (`CANCELLED`) sporadically at ≥500 RPS**, a
  status absent from the retry policy's retryable set (408/429/5xx). At
  ~0.17% it barely matters, but production should decide 499 handling
  deliberately; for these idempotent calls I lean retryable.
- Sustained 300 RPS is the cleanest high-rate evidence in the project:
  end-to-end p95 equals service p95 (no queueing) and all five 504s were
  absorbed by one retry each. Sustained 600 completed all 36,000 requests but
  degraded — service p95 8.9 s, end-to-end p95 42.8 s, 128-worker shards
  saturated, delivered rate 408. The same rate was clean for 30 s, so the
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

- **Reuse the tested harness for multi-process load.** Each shard is the
  same single-process generator the test suite covers; the orchestrator
  only launches shards and merges their raw samples. If any shard fails or
  the run is interrupted, every sibling shard is terminated so nothing
  keeps spending unsupervised.
- **Retries off when measuring failures, on when simulating production.**
  The ramp, bracket, and probe disable retries so raw failure rates stay
  visible; the operating-point and sustained runs use the retry
  policy.
- **Committed evidence is machine-generated, redacted, and hashed.** It
  contains numeric aggregates, retry breakdowns, configuration, and a
  SHA-256 for every raw source (including shard reports). It is commited 
  to the repository for historical record.
- **Campaigns cannot mix.** Stage artifacts are never overwritten, and the
  evidence generator rejects shard files that don't match their merged
  report — so reusing a RUN_ID fails fast instead of blending old and new
  data. (An earlier revision also validated overall campaign shape; it
  cost more review surface than the risk it prevented and was cut.)

## Limitations

- **All evidence is short.** The longest runs are five minutes at 24 RPS
  and 60 seconds at 300 RPS. Multi-hour behavior — quota over time,
  credential refresh, connection reuse, memory growth, slow latency drift —
  is unmeasured, and 30-second stages can only reveal abrupt saturation,
  not gradual degradation.
- **One machine generated all the load.** Every shard ran on a single
  10-core host over one network path. At sustained 600 RPS I cannot cleanly
  separate provider slowdown from client resource limits, and the 400-RPS
  stage's 9.4 ms worst-shard scheduler lag hints the host was near its
  comfortable limit with eight shards.
- **Available capacity varies with other customers' traffic.** Pay-as-you-go
  Vertex draws on a capacity pool shared across Google's customers, so the
  throughput available at any moment depends on demand outside our control.
  The clean high-rate stages here describe the pool's state during the test
  window, not a guarantee it will repeat. Throughput baselines also scale
  with organization spend, and a take-home project's spend resembles no
  production organization.
- **The workload is narrow.** Short prompts, small outputs, one model, one
  region. Long contexts, multimodal inputs, tool use, and streaming were
  untested, and time-to-first-token — the number a streaming consumer cares
  about — was never measured.
- **The quality checks are a tripwire, not an assessment.** Ten
  deterministic cases catch regressions and configuration interactions;
  they say nothing about general model quality.

## Production next steps

1. Multi-hour soak at 24 RPS (and a shorter one at ~100 RPS), watching p95
   drift, 429/499/5xx rates, retry amplification, credentials, connections,
   and memory.
2. Price [**Provisioned Throughput**](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/provisioned-throughput) before making any capacity commitment,
   using the token rates measured here as the sizing input. Shared capacity
   is fine for a pilot, but an SLO cannot be built on throughput that other
   customers' traffic can take away.
3. Setup monitoring, logging, and alerting for spikes in error rate and latency
4. Enforce the rule that a tight `max_output_tokens` requires an explicit
   thinking budget.
5. Replace developer ADC with production identity and explicit quota/billing
   ownership; correlate client request IDs with server-side telemetry.
6. Scale the load generator horizontally before claiming anything above 600 RPS;
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
