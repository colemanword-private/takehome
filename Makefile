SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYTHON ?= $(PROJECT_DIR)/.venv/bin/python
ENV_FILE ?= $(PROJECT_DIR)/.env

# Select any provider registered in llm/providers.py. Model defaults live in each
# provider's configuration; set MODEL to pin one explicitly (results are grouped
# under "default" when unset).
PROVIDER ?= gemini
MODEL ?=
PROVIDER_ARGS = --provider "$(PROVIDER)" $(if $(strip $(MODEL)),--model "$(MODEL)")

# Resolve the timestamp once so every target in one Make invocation shares a run.
# Override RUN_ID to group separate invocations into the same experiment campaign.
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_ID := $(RUN_ID)
LOAD_RESULTS_ROOT ?= $(PROJECT_DIR)/load-results
EVAL_RESULTS_ROOT ?= $(PROJECT_DIR)/eval-results

# Keep each provider/model's runs together. Replace characters commonly used in
# model IDs but unsuitable for a single directory component (for example, `/`).
empty :=
space := $(empty) $(empty)
sanitize_path = $(subst $(space),-,$(subst /,-,$(subst :,-,$(subst @,-,$(strip $(1))))))
PROVIDER_PATH = $(call sanitize_path,$(PROVIDER))
MODEL_PATH = $(if $(strip $(MODEL)),$(call sanitize_path,$(MODEL)),default)
LOAD_RESULTS_DIR ?= $(LOAD_RESULTS_ROOT)/$(PROVIDER_PATH)/$(MODEL_PATH)/$(RUN_ID)
EVAL_RESULTS_DIR ?= $(EVAL_RESULTS_ROOT)/$(PROVIDER_PATH)/$(MODEL_PATH)/$(RUN_ID)
SYNTHETIC_RESULTS_DIR ?= $(LOAD_RESULTS_ROOT)/synthetic/local/$(RUN_ID)

# These controls are mapped by the selected provider rather than by this Makefile.
TEMPERATURE ?= 0
MAX_OUTPUT_TOKENS ?= 256
THINKING_BUDGET ?=
MAX_PENDING_REQUESTS ?=
THINKING_BUDGET_ARG = $(if $(strip $(THINKING_BUDGET)),--thinking-budget "$(THINKING_BUDGET)")
MAX_PENDING_REQUESTS_ARG = $(if $(strip $(MAX_PENDING_REQUESTS)),--max-pending-requests "$(MAX_PENDING_REQUESTS)")
PROVIDER_CONTROL_ARGS = \
	--max-output-tokens "$(MAX_OUTPUT_TOKENS)" \
	$(THINKING_BUDGET_ARG)

RAMP_RPS ?= 1 3 5 8 10
RAMP_DURATION_SECONDS ?= 60
RAMP_CONCURRENCY ?= 160
# Cost and safety rails for the knee search. The ramp stops cleanly once the
# measured-request budget is spent (the last stage is truncated to fit; the five
# warmups per stage are not counted), and the first stage breaching either
# threshold aborts the run. Set RAMP_REQUEST_BUDGET empty for an unbounded ramp.
RAMP_REQUEST_BUDGET ?= 20000
RAMP_MAX_FAILURE_RATE ?= 0.01
RAMP_MAX_P95_MS ?= 5000

# Set these only after the capacity ramp identifies a candidate operating point.
RETRY_RPS ?=
RETRY_DURATION_SECONDS ?= 60
RETRY_CONCURRENCY ?= 160
PRODUCTION_RETRIES ?= 4

SOAK_RPS ?=
SOAK_DURATION_SECONDS ?= 900
SOAK_CONCURRENCY ?= 160

# Load optional provider credentials and settings from .env without embedding any
# Gemini-, Vertex-, or Together-specific variables in the experiment workflow.
define LOAD_PROVIDER_ENV
set -a; \
if [ -f "$(ENV_FILE)" ]; then source "$(ENV_FILE)"; fi; \
set +a;
endef

.PHONY: \
	help bootstrap check-python check-provider prepare-results test \
	synthetic-smoke provider-smoke quality quality-baseline quality-controlled \
	quality-factorial capacity-ramp retry-off retry-on soak evidence

# Print available workflows and common overrides. This sends no provider requests.
help:
	@printf '%s\n' \
		'Local verification:' \
		'  make bootstrap              Create/update .venv and result directories.' \
		'  make test                   Run the offline unit test suite.' \
		'  make synthetic-smoke        Exercise only the local load harness.' \
		'  make check-provider         Validate local config and credentials.' \
		'' \
		'Provider experiments (billable):' \
		'  make provider-smoke         Make one low-cost provider request.' \
		'  make quality                Run baseline and controlled quality evals.' \
		'  make quality-factorial      Run thinking x output-cap cells with repeats.' \
		'  make evidence RUN_ID=...    Build the sanitized evidence manifest (free).' \
		'  make capacity-ramp          Run staged load with retries off.' \
		'  make retry-off RETRY_RPS=N  Measure the selected rate without retries.' \
		'  make retry-on RETRY_RPS=N   Repeat it with production retries.' \
		'  make soak SOAK_RPS=N        Hold an operating point for 15 minutes.' \
		'' \
		'Provider selection:' \
		'  PROVIDER=gemini MODEL=gemini-2.5-flash' \
		'  PROVIDER=together MODEL=organization/model' \
		'' \
		'Useful overrides:' \
		'  RAMP_RPS="1 3 5" RAMP_CONCURRENCY=96 RAMP_DURATION_SECONDS=300' \
		'  RAMP_REQUEST_BUDGET=20000  Stop the ramp once this many measured' \
		'                             requests are spent (empty = unbounded).' \
		'  RAMP_MAX_FAILURE_RATE=0.01 RAMP_MAX_P95_MS=5000  Abort the ramp at' \
		'                             the first stage breaching either threshold.' \
		'  MAX_OUTPUT_TOKENS=1024 THINKING_BUDGET=512 TEMPERATURE=0' \
		'  MAX_PENDING_REQUESTS=256  Bound queued load-test arrivals.' \
		'  RUN_ID=20260816T222927Z  Reuse one run directory across invocations.' \
		'' \
		'Results use <root>/<provider>/<model>/<RUN_ID>/ (synthetic uses synthetic/local).' \
		'These scratch roots remain ignored by git. Review and redact artifacts' \
		'before copying curated results into an evidence/ directory.'

# Create the virtual environment if needed, install dependencies, and create result
# roots. This may contact the package index, but it sends no LLM requests.
bootstrap:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || python3 -m venv "$(PROJECT_DIR)/.venv"
	@"$(PYTHON)" -m pip install -r "$(PROJECT_DIR)/requirements.txt"
	@mkdir -p "$(LOAD_RESULTS_ROOT)" "$(EVAL_RESULTS_ROOT)"

# Fail with setup guidance when the configured Python executable is unavailable.
check-python:
	@test -x "$(PYTHON)" || { \
		echo 'Python environment is missing; run `make bootstrap` first.' >&2; \
		exit 1; \
	}

# Delegate configuration and credential validation to the selected provider. This
# performs configuration and credential discovery but sends no inference request.
check-provider: check-python
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	"$(PYTHON)" "$(PROJECT_DIR)/provider_check.py" $(PROVIDER_ARGS)

# Create this invocation's timestamped output directories.
prepare-results:
	@mkdir -p "$(LOAD_RESULTS_DIR)" "$(EVAL_RESULTS_DIR)" "$(SYNTHETIC_RESULTS_DIR)"

# Run the complete offline unit suite. Tests use fakes and send no provider requests.
test: check-python
	@cd "$(PROJECT_DIR)" && "$(PYTHON)" -m pytest -q

# Stress the local scheduler with synthetic responses and save a timestamped report.
# This validates harness behavior, not any external provider's latency or capacity.
synthetic-smoke: check-python prepare-results
	@"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
		--synthetic \
		--requests 10000 \
		--rps 0 \
		--concurrency 128 \
		$(MAX_PENDING_REQUESTS_ARG) \
		--warmup-requests 0 \
		--temperature "$(TEMPERATURE)" \
		--output "$(SYNTHETIC_RESULTS_DIR)/00-synthetic-smoke.json"

# BILLABLE: send one bounded request through the selected provider to verify auth,
# request mapping, and result persistence before starting larger experiments.
provider-smoke: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
		$(PROVIDER_ARGS) \
		$(PROVIDER_CONTROL_ARGS) \
		--max-retries 0 \
		--requests 1 \
		--rps 1 \
		--concurrency 1 \
		$(MAX_PENDING_REQUESTS_ARG) \
		--warmup-requests 0 \
		--temperature "$(TEMPERATURE)" \
		--output "$(LOAD_RESULTS_DIR)/01-$(PROVIDER)-smoke.json"

# BILLABLE: run both quality evaluations to compare environment-configured behavior
# with explicit supported controls for the selected provider. A baseline failure must
# not skip the controlled run, or the paired comparison is lost after billable spend;
# RUN_ID is forwarded so both sub-makes share one run directory.
quality:
	@status=0; \
	$(MAKE) quality-baseline RUN_ID="$(RUN_ID)" || status=1; \
	$(MAKE) quality-controlled RUN_ID="$(RUN_ID)" || status=1; \
	exit $$status

# BILLABLE: evaluate all golden cases with retries disabled and otherwise use the
# selected provider's environment-configured defaults.
quality-baseline: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	"$(PYTHON)" "$(PROJECT_DIR)/quality_eval.py" \
		$(PROVIDER_ARGS) \
		--max-retries 0 \
		--concurrency 2 \
		--output "$(EVAL_RESULTS_DIR)/02-$(PROVIDER)-quality-baseline.json"

# Factorial quality experiment: thinking {default, 0} x output-cap {none, 256},
# FACTORIAL_REPEATS runs per cell. "default" and "none" omit the control so the
# provider's own default applies.
FACTORIAL_REPEATS ?= 3
FACTORIAL_THINKING ?= default 0
FACTORIAL_MAX_OUTPUT ?= none 256

# BILLABLE: run every factorial cell as an independent quality evaluation with
# its own artifact. A failing cell does not skip the remaining cells; the
# combined exit status stays nonzero so the failure is still visible.
quality-factorial: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	status=0; \
	for thinking in $(FACTORIAL_THINKING); do \
		for cap in $(FACTORIAL_MAX_OUTPUT); do \
			for repeat in $$(seq 1 $(FACTORIAL_REPEATS)); do \
				cell_args=""; \
				if [ "$$thinking" != "default" ]; then cell_args="$$cell_args --thinking-budget $$thinking"; fi; \
				if [ "$$cap" != "none" ]; then cell_args="$$cell_args --max-output-tokens $$cap"; fi; \
				output="$(EVAL_RESULTS_DIR)/02-$(PROVIDER)-quality-think-$$thinking-cap-$$cap-r$$repeat.json"; \
				echo "Quality cell thinking=$$thinking cap=$$cap repeat=$$repeat..."; \
				"$(PYTHON)" "$(PROJECT_DIR)/quality_eval.py" \
					$(PROVIDER_ARGS) \
					$$cell_args \
					--max-retries 0 \
					--concurrency 2 \
					--output "$$output" || status=1; \
			done; \
		done; \
	done; \
	exit $$status

# BILLABLE: repeat the golden cases with explicit supported controls and no retries,
# so quality, token, and latency changes remain directly observable.
quality-controlled: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	"$(PYTHON)" "$(PROJECT_DIR)/quality_eval.py" \
		$(PROVIDER_ARGS) \
		$(PROVIDER_CONTROL_ARGS) \
		--max-retries 0 \
		--concurrency 2 \
		--output "$(EVAL_RESULTS_DIR)/03-$(PROVIDER)-quality-controlled.json"

# BILLABLE: step through RAMP_RPS for RAMP_DURATION_SECONDS per stage, with five
# warmups and retries disabled. The run ends at the first stage breaching
# RAMP_MAX_FAILURE_RATE or RAMP_MAX_P95_MS, or cleanly once RAMP_REQUEST_BUDGET
# is spent; inspect the last stage's report either way.
capacity-ramp: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	remaining="$(strip $(RAMP_REQUEST_BUDGET))"; \
	for rps in $(RAMP_RPS); do \
		requests=$$((rps * $(RAMP_DURATION_SECONDS))); \
		if [ -n "$$remaining" ]; then \
			if [ "$$remaining" -le 0 ]; then \
				echo "Stopping: RAMP_REQUEST_BUDGET exhausted before the $$rps RPS stage." >&2; \
				break; \
			fi; \
			if [ "$$requests" -gt "$$remaining" ]; then requests="$$remaining"; fi; \
			remaining=$$((remaining - requests)); \
		fi; \
		output="$(LOAD_RESULTS_DIR)/04-$(PROVIDER)-capacity-$${rps}rps.json"; \
		echo "Running $$requests requests against $(PROVIDER) at $$rps RPS..."; \
		"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
			$(PROVIDER_ARGS) \
			$(PROVIDER_CONTROL_ARGS) \
			--max-retries 0 \
			--max-failure-rate "$(RAMP_MAX_FAILURE_RATE)" \
			--max-service-p95-ms "$(RAMP_MAX_P95_MS)" \
			--requests "$$requests" \
			--rps "$$rps" \
			--concurrency "$(RAMP_CONCURRENCY)" \
			$(MAX_PENDING_REQUESTS_ARG) \
			--warmup-requests 5 \
			--temperature "$(TEMPERATURE)" \
			--output "$$output" || { \
				echo "Stopping: the $$rps RPS stage breached the abort criterion; see $$output." >&2; \
				exit 1; \
			}; \
	done

# BILLABLE: measure RETRY_RPS with retries disabled to establish the unamplified
# failure rate before testing a production retry policy.
retry-off: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	test -n "$(strip $(RETRY_RPS))" || { \
		echo 'Set RETRY_RPS to a rate selected from the capacity ramp.' >&2; \
		exit 1; \
	}; \
	requests=$$(( $(RETRY_RPS) * $(RETRY_DURATION_SECONDS) )); \
	"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
		$(PROVIDER_ARGS) \
		$(PROVIDER_CONTROL_ARGS) \
		--max-retries 0 \
		--requests "$$requests" \
		--rps "$(RETRY_RPS)" \
		--concurrency "$(RETRY_CONCURRENCY)" \
		$(MAX_PENDING_REQUESTS_ARG) \
		--warmup-requests 5 \
		--temperature "$(TEMPERATURE)" \
		--output "$(LOAD_RESULTS_DIR)/05-$(PROVIDER)-retry-off-$(RETRY_RPS)rps.json"

# BILLABLE: repeat RETRY_RPS with PRODUCTION_RETRIES so availability, latency, and
# extra attempts can be compared directly with the retry-off artifact.
retry-on: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	test -n "$(strip $(RETRY_RPS))" || { \
		echo 'Set RETRY_RPS to a rate selected from the capacity ramp.' >&2; \
		exit 1; \
	}; \
	requests=$$(( $(RETRY_RPS) * $(RETRY_DURATION_SECONDS) )); \
	"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
		$(PROVIDER_ARGS) \
		$(PROVIDER_CONTROL_ARGS) \
		--max-retries "$(PRODUCTION_RETRIES)" \
		--requests "$$requests" \
		--rps "$(RETRY_RPS)" \
		--concurrency "$(RETRY_CONCURRENCY)" \
		$(MAX_PENDING_REQUESTS_ARG) \
		--warmup-requests 5 \
		--temperature "$(TEMPERATURE)" \
		--output "$(LOAD_RESULTS_DIR)/06-$(PROVIDER)-retry-on-$(RETRY_RPS)rps.json"

# BILLABLE: hold SOAK_RPS for SOAK_DURATION_SECONDS to expose sustained latency,
# queueing, credential-refresh, or resource problems at the chosen operating point.
soak: check-provider prepare-results
	@set -eu; \
	$(LOAD_PROVIDER_ENV) \
	test -n "$(strip $(SOAK_RPS))" || { \
		echo 'Set SOAK_RPS to roughly 60-70% of the measured capacity knee.' >&2; \
		exit 1; \
	}; \
	requests=$$(( $(SOAK_RPS) * $(SOAK_DURATION_SECONDS) )); \
	"$(PYTHON)" "$(PROJECT_DIR)/load_test.py" \
		$(PROVIDER_ARGS) \
		$(PROVIDER_CONTROL_ARGS) \
		--max-retries "$(PRODUCTION_RETRIES)" \
		--requests "$$requests" \
		--rps "$(SOAK_RPS)" \
		--concurrency "$(SOAK_CONCURRENCY)" \
		$(MAX_PENDING_REQUESTS_ARG) \
		--warmup-requests 5 \
		--temperature "$(TEMPERATURE)" \
		--output "$(LOAD_RESULTS_DIR)/07-$(PROVIDER)-soak-$(SOAK_RPS)rps.json"

# Aggregate one run's raw artifacts into the sanitized, checksummed evidence
# manifest that is safe to commit. Sends no provider requests.
evidence: check-python
	@"$(PYTHON)" "$(PROJECT_DIR)/evidence_manifest.py" \
		--run-id "$(RUN_ID)" \
		--provider-dir "$(PROVIDER_PATH)" \
		--model-dir "$(MODEL_PATH)" \
		--project-dir "$(PROJECT_DIR)"
