# Convert the merged model to GGUF, quantize it to Q4_K_M (the quantization of the stock
# qwen2.5 7B in Ollama, so the comparison is like for like) and import it into Ollama with the
# base model's chat template, which carries the tool-calling format.
#
# llama.cpp runs in its official Docker image; Ollama's own safetensors import only offers
# int4/int8/fp4/fp8, not Q4_K_M.
param(
    [string]$Name = "qwen2.5-fin:7b",
    [string]$Base = "qwen2.5",            # = qwen2.5:latest, the 7B the agent uses by default
    [string]$Image = "ghcr.io/ggml-org/llama.cpp:full"
)
$ErrorActionPreference = "Stop"
$out = Resolve-Path "$PSScriptRoot\out"
if (-not (Test-Path "$out\merged\config.json")) { throw "run lora/merge.py first" }

if (-not (Test-Path "$out\fin-f16.gguf")) {
    Write-Host "1/3 merged safetensors -> GGUF f16"
    docker run --rm -v "${out}:/models" $Image --convert /models/merged --outtype f16 --outfile /models/fin-f16.gguf
    if ($LASTEXITCODE -ne 0) { throw "conversion failed" }
}
if (-not (Test-Path "$out\fin-q4_k_m.gguf")) {
    Write-Host "2/3 f16 -> Q4_K_M"
    docker run --rm -v "${out}:/models" $Image --quantize /models/fin-f16.gguf /models/fin-q4_k_m.gguf Q4_K_M
    if ($LASTEXITCODE -ne 0) { throw "quantization failed" }
}

Write-Host "3/3 Ollama model $Name (template and parameters from $Base)"
$template = ollama show $Base --modelfile
if ($LASTEXITCODE -ne 0) { throw "base model $Base not found; check 'ollama list'" }
$lines = $template -split "`r?`n" | Where-Object { $_ -notmatch '^\s*FROM\s' -and $_ -notmatch '^#' }
# UTF-8 without BOM ("-Encoding utf8" adds one, which breaks the FROM line) and LF line endings:
# the TEMPLATE block is multi-line, and CRLF there puts "\r\n" into every ChatML turn
# ("<|im_start|>system\r\n..."), a prompt format the model never saw in training. The first
# evaluation of the fine-tuned model ran with that template and answered with "\r\n" throughout.
$text = ((@("FROM ./fin-q4_k_m.gguf") + $lines) -join "`n") + "`n"
[IO.File]::WriteAllText("$out\Modelfile", $text, (New-Object Text.UTF8Encoding $false))
Push-Location $out
try { ollama create $Name -f Modelfile } finally { Pop-Location }
ollama list | Select-String $Name
Write-Host "The f16 GGUF and merged/ folder can be deleted once the evaluation is done (~30 GB)."
