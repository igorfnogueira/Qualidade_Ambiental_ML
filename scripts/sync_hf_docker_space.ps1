# Atualiza hf_docker_space/ a partir da raiz do projeto (para copiar no clone do Hugging Face Space).
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Dst = Join-Path $ProjectRoot "hf_docker_space"

New-Item -ItemType Directory -Force -Path $Dst | Out-Null
foreach ($dir in @("qa_api", "web", "artifacts")) {
    $p = Join-Path $Dst $dir
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}
Copy-Item (Join-Path $ProjectRoot "Dockerfile") $Dst -Force
Copy-Item (Join-Path $ProjectRoot "requirements.api.txt") $Dst -Force
Copy-Item (Join-Path $ProjectRoot "qa_api") (Join-Path $Dst "qa_api") -Recurse -Force
Copy-Item (Join-Path $ProjectRoot "web") (Join-Path $Dst "web") -Recurse -Force
Copy-Item (Join-Path $ProjectRoot "artifacts") (Join-Path $Dst "artifacts") -Recurse -Force
Write-Host "Sincronizado: $Dst"
