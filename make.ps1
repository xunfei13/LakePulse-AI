param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    if ($env:LAKEPULSE_PYTHON -and (Test-Path $env:LAKEPULSE_PYTHON)) {
        return $env:LAKEPULSE_PYTHON
    }
    $condaPython = "D:\miniconda\envs\lakepulse\python.exe"
    if (Test-Path $condaPython) {
        return $condaPython
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    throw "Python executable not found. Create conda env 'lakepulse' or set LAKEPULSE_PYTHON."
}

switch ($Target.ToLowerInvariant()) {
    "help" {
        Write-Output "LakePulse-AI PowerShell Runner (make.ps1):"
        Write-Output "  .\make.ps1 check          - Run Stage 0 environment self-check"
        Write-Output "  .\make.ps1 test           - Run pytest unit tests"
        Write-Output "  .\make.ps1 lint           - Run ruff linter"
        Write-Output "  .\make.ps1 docker-config  - Validate docker/docker-compose.yml"
        Write-Output "  .\make.ps1 etl            - Run Stage 1 M5 bootstrap & PySpark ODS ETL"
        Write-Output "  .\make.ps1 dbt            - Run Stage 1 dbt build, quality gates & publish"
        Write-Output "  .\make.ps1 stage1         - Run complete Stage 1 pipeline (etl + dbt)"
        Write-Output "  .\make.ps1 analytics      - Run Stage 2 diagnostics & rolling features"
        Write-Output "  .\make.ps1 stage2         - Alias for Stage 2 analytics & features"
        Write-Output "  .\make.ps1 train          - Run Stage 3 model training & evaluation"
        Write-Output "  .\make.ps1 infer          - Run Stage 3 distributed @pandas_udf inference"
        Write-Output "  .\make.ps1 stage3         - Run complete Stage 3 pipeline (train + infer)"
        Write-Output "  .\make.ps1 serve          - Start FastAPI read-only SQL & OpenAPI gateway"
        Write-Output "  .\make.ps1 tunnel         - Start Cloudflare Quick Tunnel & sync OpenAPI JSON"
        Write-Output "  .\make.ps1 stage4         - Export Dify OpenAPI tools & run Agent closed-loop"
        Write-Output "  .\make.ps1 all            - Run complete Stage 1 -> Stage 4 pipeline"
        exit 0
    }
    "check" {
        $py = Resolve-PythonExe
        & $py scripts/check_env.py
        exit $LASTEXITCODE
    }
    "test" {
        $py = Resolve-PythonExe
        & $py -m pytest -v
        exit $LASTEXITCODE
    }
    "lint" {
        $py = Resolve-PythonExe
        & $py -m ruff check src tests scripts
        exit $LASTEXITCODE
    }
    "docker-config" {
        docker compose -f docker/docker-compose.yml config --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Output "[PASS] docker/docker-compose.yml config is valid."
        }
        exit $LASTEXITCODE
    }
    "etl" {
        $py = Resolve-PythonExe
        & $py scripts/run_etl.py
        exit $LASTEXITCODE
    }
    "dbt" {
        $py = Resolve-PythonExe
        & $py scripts/run_dbt.py
        exit $LASTEXITCODE
    }
    "stage1" {
        $py = Resolve-PythonExe
        & $py scripts/run_etl.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_dbt.py
        exit $LASTEXITCODE
    }
    "analytics" {
        $py = Resolve-PythonExe
        & $py scripts/run_analytics.py
        exit $LASTEXITCODE
    }
    "stage2" {
        $py = Resolve-PythonExe
        & $py scripts/run_analytics.py
        exit $LASTEXITCODE
    }
    "train" {
        $py = Resolve-PythonExe
        & $py scripts/run_train.py
        exit $LASTEXITCODE
    }
    "infer" {
        $py = Resolve-PythonExe
        & $py scripts/run_infer.py
        exit $LASTEXITCODE
    }
    "stage3" {
        $py = Resolve-PythonExe
        & $py scripts/run_train.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_infer.py
        exit $LASTEXITCODE
    }
    "serve" {
        $py = Resolve-PythonExe
        & $py -m uvicorn lakepulse.api.app:create_app --factory --host 127.0.0.1 --port 8000
        exit $LASTEXITCODE
    }
    "tunnel" {
        $py = Resolve-PythonExe
        & $py scripts/start_tunnel.py
        exit $LASTEXITCODE
    }
    "stage4" {
        $py = Resolve-PythonExe
        & $py scripts/demo_agent_workflow.py
        exit $LASTEXITCODE
    }
    "agent-demo" {
        $py = Resolve-PythonExe
        & $py scripts/demo_agent_workflow.py
        exit $LASTEXITCODE
    }
    "all" {
        $py = Resolve-PythonExe
        & $py scripts/run_etl.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_dbt.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_analytics.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_train.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/run_infer.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $py scripts/demo_agent_workflow.py
        exit $LASTEXITCODE
    }
    default {
        Write-Error "Unknown target '$Target'. Run '.\make.ps1 help' for usage."
        exit 2
    }
}
