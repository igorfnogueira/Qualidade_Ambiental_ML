# Push do clone local do Space para o Hugging Face (nao commite tokens).
# Uso interativo: cd Documents\HF_Qualidade_Ambiental_ML; git push origin main
# Uso nao-interativo (PowerShell):
#   $env:HF_TOKEN = "hf_seu_token_write"
#   .\scripts\push_hf_space.ps1
$ErrorActionPreference = "Stop"
$CloneRoot = Join-Path $env:USERPROFILE "Documents\HF_Qualidade_Ambiental_ML"
if (-not (Test-Path (Join-Path $CloneRoot ".git"))) {
    Write-Error "Clone nao encontrado: $CloneRoot. Clone o Space em Documents primeiro."
}
Set-Location $CloneRoot
$remote = (git remote get-url origin).Trim()
if ($remote -notmatch "huggingface\.co/spaces/([^/]+)/([^./]+)") {
    Write-Error "Remote inesperado: $remote"
}
$hfUser = $Matches[1]
$repoName = $Matches[2]

$rawToken = [string]$env:HF_TOKEN
$token = if ($rawToken) { $rawToken.Trim() } else { "" }
if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host @"
Defina HF_TOKEN (token Write do Hugging Face) ou faça push manual:

  cd `"$CloneRoot`"
  git push origin main

Login: username HF ($hfUser). Senha: token hf_... (nao use senha da conta).

Dica: sem espacos dentro das aspas ao definir `$env:HF_TOKEN.
"@
    exit 1
}

$pushUrl = "https://${hfUser}:${token}@huggingface.co/spaces/${hfUser}/${repoName}.git"
git push $pushUrl main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Push concluido. Confira Files e Logs no Space."
