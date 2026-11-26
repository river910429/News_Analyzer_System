# Deploy the whole stack to the local Kubernetes cluster (Docker Desktop).
#   powershell -ExecutionPolicy Bypass -File k8s\deploy.ps1          # deploy / update
#   powershell -ExecutionPolicy Bypass -File k8s\deploy.ps1 -NoKeda  # without autoscaling
param([switch]$NoKeda, [string]$KedaVersion = "2.16.1")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "1/5 Building images with docker compose"
docker compose build api-gateway nlp-worker mcp-server frontend

Write-Host "2/5 Namespace, secret (from .env) and DB init script"
kubectl apply -f k8s/namespace.yaml
kubectl -n financial-rag create secret generic rag-env --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
kubectl -n financial-rag create configmap db-init --from-file=init.sql=init.sql --dry-run=client -o yaml | kubectl apply -f -

Write-Host "3/5 Infrastructure and services"
kubectl apply -k k8s

if (-not $NoKeda) {
    Write-Host "4/5 KEDA $KedaVersion"
    kubectl get namespace keda 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        kubectl apply --server-side -f "https://github.com/kedacore/keda/releases/download/v$KedaVersion/keda-$KedaVersion.yaml"
        kubectl -n keda rollout status deployment/keda-operator --timeout=300s
    }
    kubectl -n financial-rag apply -f k8s/scaledobject.yaml
} else {
    Write-Host "4/5 Skipping KEDA"
}

Write-Host "5/5 Waiting for the API (first start loads bge-m3)"
kubectl -n financial-rag rollout restart deployment/api-gateway deployment/nlp-worker | Out-Null
kubectl -n financial-rag rollout status deployment/api-gateway --timeout=600s
Write-Host ""
Write-Host "Frontend:   http://localhost:30173"
Write-Host "API:        http://localhost:30080/health"
Write-Host "MCP server: http://localhost:30801/mcp"
