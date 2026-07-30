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

# README mínimo exigido pelo Hugging Face Spaces (sdk: docker). Não usar como docs do GitHub.
$earth = [char]::ConvertFromUtf32(0x1F30D)
$hfReadme = @"
---
title: Qualidade Ambiental (IA)
emoji: $earth
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Web UI + FastAPI for environmental quality classification. See the GitHub repository README for full documentation.
"@
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $Dst "README.md"), $hfReadme.Replace("`r`n", "`n"), $utf8NoBom)

Write-Host "Sincronizado: $Dst"
