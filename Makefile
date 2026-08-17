SHELL := /bin/bash
PYTHON ?= .venv/bin/python
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
MODEL ?= gemini-2.5-flash
LOAD_DIR := load-results/gemini/$(MODEL)/$(RUN_ID)
EVAL_DIR := eval-results/gemini/$(MODEL)/$(RUN_ID)
SYNTHETIC_DIR := load-results/synthetic/local/$(RUN_ID)

RAMP_RPS ?= 1 3 5 8 10 15 25 40 60 80 100 125 150 200 250 300 400
RAMP_SECONDS ?= 30
RAMP_MAX_FAILURE_RATE ?= 0.01
RAMP_MAX_P95_MS ?= 5000
RAMP_MAX_E2E_P95_MS ?= 5000
CONCURRENCY ?= 96
RETRY_RPS ?= 40
OPERATING_RPS ?= 24
OPERATING_SECONDS ?= 300
MAX_OUTPUT_TOKENS ?= 256
THINKING_BUDGET ?= 0

# Multi-process stages: the sharded bracket re-tests the single-process
# saturation region with a client-clean generator; the probe hunts for the
# quota/provider ceiling above it. 600 RPS is a deliberate hard stop on this
# shared project, not a technical limit.
BRACKET_RPS ?= 150 200 250 300
BRACKET_SHARDS ?= 4
PROBE_RPS ?= 400 500 600
PROBE_SHARDS ?= 8
SHARD_CONCURRENCY ?= 128
MAX_SHARD_SCHED_P95_MS ?= 10
DEMO_RPS ?= 600
DEMO_SECONDS ?= 60

GEMINI_ARGS = --model "$(MODEL)" --max-output-tokens "$(MAX_OUTPUT_TOKENS)" \
	--thinking-budget "$(THINKING_BUDGET)"
LOAD_ARGS = $(GEMINI_ARGS) --concurrency "$(CONCURRENCY)" \
	--workload load_test_workload.json --temperature 0
MULTI_ARGS = $(GEMINI_ARGS) --shard-concurrency "$(SHARD_CONCURRENCY)" \
	--workload load_test_workload.json --temperature 0 --warmup-requests 5

define RUN_WITH_ENV
set -a; if [ -f .env ]; then source .env || exit 2; fi; set +a;
endef

.PHONY: bootstrap test synthetic-smoke provider-smoke quality-factorial \
	capacity-ramp operating-point retry-comparison evidence \
	sharded-bracket quota-probe retries-demo

bootstrap:
	@test -x "$(PYTHON)" || python3 -m venv .venv
	@"$(PYTHON)" -m pip install -r requirements.txt

test:
	@"$(PYTHON)" -m pytest -q

synthetic-smoke:
	@mkdir -p "$(SYNTHETIC_DIR)"
	@"$(PYTHON)" load_test.py --synthetic --requests 10000 --rps 0 \
		--concurrency 128 --warmup-requests 0 \
		--output "$(SYNTHETIC_DIR)/00-synthetic-smoke.json"

provider-smoke:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) "$(PYTHON)" load_test.py $(LOAD_ARGS) --max-retries 0 \
		--requests 1 --rps 1 --concurrency 1 --warmup-requests 0 \
		--output "$(LOAD_DIR)/01-gemini-smoke.json"

quality-factorial:
	@mkdir -p "$(EVAL_DIR)"
	@$(RUN_WITH_ENV) status=0; \
	for thinking in default 0; do \
		for cap in none 256; do \
			for repeat in 1 2 3; do \
				"$(PYTHON)" quality_eval.py --model "$(MODEL)" \
					--thinking-budget "$$thinking" --max-output-tokens "$$cap" \
					--output "$(EVAL_DIR)/quality-think-$$thinking-cap-$$cap-r$$repeat.json" \
					|| status=1; \
			done; \
		done; \
	done; exit $$status

capacity-ramp:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) set -eu; shopt -s nullglob; \
	previous_reports=("$(LOAD_DIR)"/capacity-*.json); \
	if [ $${#previous_reports[@]} -gt 0 ]; then \
		archive="$(LOAD_DIR)/.superseded-capacity-$$(date -u +%Y%m%dT%H%M%SZ)-$$$$"; \
		mkdir "$$archive"; \
		mv "$${previous_reports[@]}" "$$archive/"; \
	fi; \
	boundary_reached=0; \
	for rps in $(RAMP_RPS); do \
		requests=$$((rps * $(RAMP_SECONDS))); \
		output="$(LOAD_DIR)/capacity-$${rps}rps.json"; \
		tmp_output="$${output}.tmp.$$$$"; \
		test ! -e "$$tmp_output" || exit 2; \
		set +e; \
		"$(PYTHON)" load_test.py $(LOAD_ARGS) --max-retries 0 \
			--requests "$$requests" --rps "$$rps" --warmup-requests 5 \
			--max-failure-rate "$(RAMP_MAX_FAILURE_RATE)" \
			--max-service-p95-ms "$(RAMP_MAX_P95_MS)" \
			--max-end-to-end-p95-ms "$(RAMP_MAX_E2E_P95_MS)" \
			--output "$$tmp_output"; \
		status=$$?; set -e; \
		if [ $$status -eq 0 ] || [ $$status -eq 1 ]; then \
			test -f "$$tmp_output" || exit 2; \
			mv "$$tmp_output" "$$output"; \
		fi; \
		if [ $$status -eq 1 ]; then \
			if "$(PYTHON)" -c 'import json, sys; report = json.load(open(sys.argv[1])); sys.exit(not report["acceptance"]["breaches"])' "$$output"; then \
				echo "Capacity boundary reached at $${rps} RPS; stopping ramp."; \
				boundary_reached=1; break; \
			fi; \
			exit $$status; \
		elif [ $$status -ne 0 ]; then \
			exit $$status; \
		fi; \
	done; \
	if [ $$boundary_reached -ne 1 ]; then \
		echo "No capacity boundary found in configured RAMP_RPS stages." >&2; \
		exit 2; \
	fi

operating-point:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) requests=$$(( $(OPERATING_RPS) * $(OPERATING_SECONDS) )); \
	"$(PYTHON)" load_test.py $(LOAD_ARGS) --max-retries 4 \
		--requests "$$requests" --rps "$(OPERATING_RPS)" --warmup-requests 10 \
		--max-failure-rate "$(RAMP_MAX_FAILURE_RATE)" \
		--max-service-p95-ms "$(RAMP_MAX_P95_MS)" \
		--max-end-to-end-p95-ms "$(RAMP_MAX_E2E_P95_MS)" \
		--output "$(LOAD_DIR)/operating-point-$(OPERATING_RPS)rps.json"

retry-comparison:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) for retries in 0 4; do \
		requests=$$(( $(RETRY_RPS) * $(RAMP_SECONDS) )); \
		"$(PYTHON)" load_test.py $(LOAD_ARGS) --max-retries "$$retries" \
			--requests "$$requests" --rps "$(RETRY_RPS)" --warmup-requests 5 \
			--max-failure-rate 1 \
			--output "$(LOAD_DIR)/retry-$${retries}-$(RETRY_RPS)rps.json" || exit 1; \
	done

# Stops at the first breached stage; a clean top-out is a valid result for
# both multi-process campaigns (unlike capacity-ramp, which must find a
# boundary).
sharded-bracket:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) for rps in $(BRACKET_RPS); do \
		requests=$$((rps * $(RAMP_SECONDS))); \
		set +e; \
		"$(PYTHON)" multi_load.py $(MULTI_ARGS) --max-retries 0 \
			--shards "$(BRACKET_SHARDS)" --total-rps "$$rps" \
			--requests "$$requests" \
			--max-failure-rate "$(RAMP_MAX_FAILURE_RATE)" \
			--max-service-p95-ms "$(RAMP_MAX_P95_MS)" \
			--max-end-to-end-p95-ms "$(RAMP_MAX_E2E_P95_MS)" \
			--max-shard-scheduler-p95-ms "$(MAX_SHARD_SCHED_P95_MS)" \
			--output "$(LOAD_DIR)/bracket-$${rps}rps.json"; \
		status=$$?; set -e; \
		if [ $$status -eq 1 ]; then \
			echo "Bracket breach at $${rps} RPS; stopping."; break; \
		elif [ $$status -ne 0 ]; then \
			exit $$status; \
		fi; \
	done

quota-probe:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) for rps in $(PROBE_RPS); do \
		requests=$$((rps * $(RAMP_SECONDS))); \
		set +e; \
		"$(PYTHON)" multi_load.py $(MULTI_ARGS) --max-retries 0 \
			--shards "$(PROBE_SHARDS)" --total-rps "$$rps" \
			--requests "$$requests" \
			--max-failure-rate "$(RAMP_MAX_FAILURE_RATE)" \
			--max-service-p95-ms "$(RAMP_MAX_P95_MS)" \
			--max-end-to-end-p95-ms "$(RAMP_MAX_E2E_P95_MS)" \
			--max-shard-scheduler-p95-ms "$(MAX_SHARD_SCHED_P95_MS)" \
			--output "$(LOAD_DIR)/probe-$${rps}rps.json"; \
		status=$$?; set -e; \
		if [ $$status -eq 1 ]; then \
			echo "Probe breach at $${rps} RPS; stopping."; break; \
		elif [ $$status -ne 0 ]; then \
			exit $$status; \
		fi; \
	done

# Observational: production retry settings at the breaching (or top) rate,
# with no acceptance rails so the stage always completes and reports.
retries-demo:
	@mkdir -p "$(LOAD_DIR)"
	@$(RUN_WITH_ENV) requests=$$(( $(DEMO_RPS) * $(DEMO_SECONDS) )); \
	"$(PYTHON)" multi_load.py $(MULTI_ARGS) --max-retries 4 \
		--shards "$(PROBE_SHARDS)" --total-rps "$(DEMO_RPS)" \
		--requests "$$requests" --max-failure-rate 1 \
		--max-shard-scheduler-p95-ms 100000 \
		--output "$(LOAD_DIR)/retries-demo-$(DEMO_RPS)rps.json"

evidence:
	@"$(PYTHON)" evidence_summary.py --run-id "$(RUN_ID)" --model "$(MODEL)"
