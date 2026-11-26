# Import the merged model into Ollama with the base model's chat template (tool calling included),
# quantized to q4_K_M like the stock qwen2.5:7b, so the comparison is like for like.
param(
    [string]$Name = "qwen2.5-fin:7b",
    [string]$Base = "qwen2.5:7b"
)
$ErrorActionPreference = "Stop"
$merged = Resolve-Path "$PSScriptRoot\out\merged"
$modelfile = "$PSScriptRoot\out\Modelfile"

# Same TEMPLATE / PARAMETER lines as the base model; only FROM changes.
$lines = ollama show $Base --modelfile | Where-Object { $_ -notmatch '^\s*FROM\s' -and $_ -notmatch '^#' }
@("FROM $merged") + $lines | Set-Content -Encoding utf8 $modelfile

ollama create $Name -q q4_K_M -f $modelfile
ollama list | Select-String $Name
