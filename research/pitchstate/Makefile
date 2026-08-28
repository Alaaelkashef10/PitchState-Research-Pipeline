PYTHON ?= python3
PYTHONPATH := src
CONFIG := configs/smoke.toml
OUTPUT := outputs/smoke.json

.PHONY: help smoke test validate-manifest audit-manifest

help:
	@echo "make smoke              Run the deterministic synthetic pipeline"
	@echo "make test               Run the standard-library test suite"
	@echo "make validate-manifest Validate the example dataset manifest"
	@echo "make audit-manifest    Run the match/clip leakage audit"

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pitchstate.cli smoke --config $(CONFIG) --output $(OUTPUT)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

validate-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pitchstate.cli manifest validate --path data/manifests/example.json

audit-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pitchstate.cli manifest audit --path data/manifests/example.json