Param(
  [string]$Version = "v0.1.0",
  [string]$Dest = "$HOME/.local/bin",
  [string]$Owner = "coding-agent-search",
  [string]$Repo = "coding-agent-search",
  [string]$Checksum = "",
  [switch]$EasyMode
)

$ErrorActionPreference = "Stop"
$os = "windows"
$arch = if ([Environment]::Is64BitProcess) { "x86_64" } else { "x86" }
$zip = "coding-agent-search-$Version-$os-$arch.zip"
$url = "https://github.com/$Owner/$Repo/releases/download/$Version/$zip"
$tmp = New-TemporaryFile | Split-Path

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile "$tmp/$zip"

if ($Checksum -ne "") {
  $hash = Get-FileHash "$tmp/$zip" -Algorithm SHA256
  if ($hash.Hash.ToLower() -ne $Checksum.ToLower()) {
    Write-Error "Checksum mismatch"
    exit 1
  }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$tmp/$zip", $tmp)

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item "$tmp/coding-agent-search.exe" "$Dest" -Force

Write-Host "Installed to $Dest/coding-agent-search.exe"
if ($EasyMode) {
  $path = [Environment]::GetEnvironmentVariable("PATH", "User")
  if (-not $path.Contains($Dest)) {
    [Environment]::SetEnvironmentVariable("PATH", "$path;$Dest", "User")
    Write-Host "Added $Dest to PATH (User)"
  }
}
