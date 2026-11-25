PYTHON ?= python

.PHONY: help check test lint docker-config etl dbt stage1 analytics stage2 train infer stage3 serve tunnel stage4 agent-demo all

help:
	@echo "LakePulse-AI Makefile targets:"
	@echo "  make check          - Run Stage 0 environment self-check"
	@echo "  make test           - Run pytest unit tests"
	@echo "  make lint           - Run ruff linter"
	@echo "  make docker-config  - Validate docker-compose.yml syntax"
	@echo "  make etl            - Run Stage 1 M5 bootstrap & PySpark ODS ETL"
	@echo "  make dbt            - Run Stage 1 dbt build, quality gates & publish"
	@echo "  make stage1         - Run complete Stage 1 pipeline (etl + dbt)"
	@echo "  make analytics      - Run Stage 2 diagnostics & rolling feature engineering"
	@echo "  make stage2         - Alias for Stage 2 analytics & feature engineering"
	@echo "  make train          - Run Stage 3 model training & evaluation"
	@echo "  make infer          - Run Stage 3 distributed @pandas_udf inference"
	@echo "  make stage3         - Run complete Stage 3 pipeline (train + infer)"
	@echo "  make serve          - Start FastAPI read-only SQL & OpenAPI gateway"
	@echo "  make tunnel         - Start Cloudflare Quick Tunnel & sync openapi_tools.json"
	@echo "  make stage4         - Export Dify OpenAPI tools & run Agent closed-loop demo"
	@echo "  make all            - Run complete Stage 1 -> Stage 4 pipeline"

check:
	$(PYTHON) scripts/check_env.py

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check src tests scripts

docker-config:
	docker compose -f docker/docker-compose.yml config --quiet

etl:
	$(PYTHON) scripts/run_etl.py

dbt:
	$(PYTHON) scripts/run_dbt.py

stage1: etl dbt

analytics:
	$(PYTHON) scripts/run_analytics.py

stage2: analytics

train:
	$(PYTHON) scripts/run_train.py

infer:
	$(PYTHON) scripts/run_infer.py

stage3: train infer

serve:
	$(PYTHON) -m uvicorn lakepulse.api.app:create_app --factory --host 127.0.0.1 --port 8000

tunnel:
	$(PYTHON) scripts/start_tunnel.py

stage4:
	$(PYTHON) scripts/demo_agent_workflow.py

agent-demo: stage4

all: stage1 stage2 stage3 stage4
