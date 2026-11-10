# Invoke from anywhere: make -f /path/to/experiments/hrf_simulations/Makefile help
SHELL := /bin/sh
.DEFAULT_GOAL := help
# Workflow stages must remain sequential even when make itself receives -j.
.NOTPARALLEL:

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT := $(abspath $(dir $(MAKEFILE_PATH))/../..)
SIM_DIR := $(ROOT)/experiments/hrf_simulations
UV_RUN ?= uv run
CLI := $(UV_RUN) python -m experiments.hrf_simulations.cli

N_JOBS ?= 8
STAGE ?= pilot
BIDS_ROOT ?= /Volumes/extdata1/NSD/BIDS
SUBJECT ?= sub-07
MASK ?= $(ROOT)/experiments/hrf_estimation/pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz
CALIBRATION ?= $(SIM_DIR)/calibration/nsd-v1
CALIBRATION_SESSIONS ?= 10 11 12 13 14 15 16 17 18 19
OUTPUT ?= $(SIM_DIR)/outputs/$(STAGE)-v1
REPORT ?= $(SIM_DIR)/reports/$(STAGE)-v1
NOTEBOOK ?= $(SIM_DIR)/results.ipynb
SMOKE_OUTPUT ?= $(SIM_DIR)/outputs/smoke-make

# Optional reductions/controls; leave empty for the configured stage defaults.
SESSIONS ?=
REPLICATES ?=
LIMIT ?=
CASE ?=
SNR ?=
RUN_ARGS ?=
GRID_ARGS = $(if $(strip $(SESSIONS)),--sessions $(SESSIONS)) $(if $(strip $(REPLICATES)),--replicates $(REPLICATES))
CASE_ARGS = $(if $(strip $(LIMIT)),--limit $(LIMIT)) $(if $(strip $(CASE)),--case $(CASE)) $(if $(strip $(SNR)),--snr $(SNR))

# Eight processes, one numerical-library thread each.
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.PHONY: help plan calibrate run report notebook smoke test workflow all

help:
	@printf '%s\n' \
	  'Targets: plan calibrate run report notebook smoke test workflow (alias: all)' \
	  'workflow = calibrate -> run -> report -> notebook; default N_JOBS=8, STAGE=pilot.' \
	  'Inspect first: make -f experiments/hrf_simulations/Makefile plan' \
	  'Small workflow: ... workflow LIMIT=1 REPLICATES=1' \
	  'Overrides: N_JOBS STAGE CALIBRATION OUTPUT REPORT SESSIONS REPLICATES LIMIT CASE SNR RUN_ARGS' \
	  'Use a new OUTPUT for changed code/settings; existing calibration is verified and reused.'

plan:
	cd "$(ROOT)" && $(CLI) plan --stage "$(STAGE)" $(GRID_ARGS)

$(CALIBRATION)/COMPLETE.json:
	cd "$(ROOT)" && $(CLI) calibrate --bids-root "$(BIDS_ROOT)" \
	  --subject "$(SUBJECT)" --mask "$(MASK)" --sessions $(CALIBRATION_SESSIONS) \
	  --output "$(CALIBRATION)"

calibrate: $(CALIBRATION)/COMPLETE.json
	cd "$(ROOT)" && $(UV_RUN) python -c 'import sys; from experiments.hrf_simulations.io import verify; verify(sys.argv[1])' "$(CALIBRATION)"

run:
	cd "$(ROOT)" && $(CLI) run --calibration "$(CALIBRATION)" --output "$(OUTPUT)" \
	  --bids-root "$(BIDS_ROOT)" --subject "$(SUBJECT)" --mask "$(MASK)" \
	  --stage "$(STAGE)" --n-jobs "$(N_JOBS)" $(GRID_ARGS) $(CASE_ARGS) $(RUN_ARGS)

report:
	cd "$(ROOT)" && $(CLI) report --input "$(OUTPUT)" --output "$(REPORT)"

notebook:
	cd "$(ROOT)" && HRF_SIM_REPORT="$(REPORT)" $(UV_RUN) python -m experiments.hrf_simulations.render \
	  --report "$(REPORT)" --output "$(NOTEBOOK)" --source "$(SIM_DIR)/results.py"

smoke:
	cd "$(ROOT)" && $(CLI) smoke --output "$(SMOKE_OUTPUT)"

test:
	cd "$(ROOT)" && $(UV_RUN) python -m pytest experiments/hrf_simulations tests/test_notebook_workflow.py -q

workflow: calibrate run report notebook

all: workflow
