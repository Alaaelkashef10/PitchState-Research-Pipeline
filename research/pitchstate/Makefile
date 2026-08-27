PYTHON ?= python3
PYTHONPATH := src
CONFIG := configs/smoke.toml
OUTPUT := outputs/smoke.json

.PHONY: help smoke test validate-manifest

help:
	@echo "make smoke              Run the deterministic synthetic pipeline"
	@echo "make test               Run the standard-library test suite"
	@echo "make validate-manifest Validate the example dataset manifest"

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pitchstate.cli smoke --config $(CONFIG) --output $(OUTPUT)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

validate-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pitchstate.cli manifest validate --path data/manifests/example.json