PYTHON = .venv/bin/python

.PHONY: setup generate demo test clean-demo

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

generate:
	$(PYTHON) scripts/generate_data.py data/raw

demo: clean-demo generate
	$(PYTHON) -m src.pipeline run

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

clean-demo:
	rm -f data/raw/*.csv data/processed/*.parquet data/quarantine/*.csv
	rm -f logs/reports/*.json logs/pipeline.log state/manifest.json
	rm -f models/*.joblib models/*.json

