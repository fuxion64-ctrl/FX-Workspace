# Used by the setup wizard before it updates an installed copy: closes FX Workspace in that folder.
# First politely - the app is asked to quit (POST /api/shell/quit, since closing its window only hides it in the
# tray); its launcher then stops the server and llama-server by itself - then, after 20 s, whatever of that copy is
# still running (the server under run.bat, a hung process).
# 1.4.8: the quit request goes to the port the app really uses (data\run\ports.json - it moves off 8000 when that is
# taken), and a llama-server left behind by a copy that crashed or hung is ended too, also when the models folder
# (AGENTIC_HOME) is on another drive - it kept the engine's files locked, so the update couldn't replace them. A copy
# that hung while closing (10 Nov 2026) is ended here as well: the update never needs anyone to open Task Manager.
param([Parameter(Mandatory = $true)][string]$App)
$App = $App.TrimEnd('\')
$Homes = @($App)
$custom = [Environment]::GetEnvironmentVariable('AGENTIC_HOME', 'User')
if ($custom) { $Homes += $custom.TrimEnd('\') }

function Mine { Get-CimInstance Win32_Process -Filter "name='pythonw.exe' or name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$App*" } }
function Engines { Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
    Where-Object { $p = $_.ExecutablePath; $p -and ($Homes | Where-Object { $p -like "$_\*" }) } }

$ports = @(8000)
try {
    $run = Get-Content -LiteralPath (Join-Path $App 'data\run\ports.json') -Raw -ErrorAction Stop | ConvertFrom-Json
    if ($run.app -and [int]$run.app -ne 8000) { $ports = @([int]$run.app) + $ports }
} catch {}
foreach ($port in $ports) {
    try { Invoke-WebRequest -UseBasicParsing -Method Post -TimeoutSec 3 -Uri "http://127.0.0.1:$port/api/shell/quit" | Out-Null } catch {}
}

Get-Process 'FX Workspace' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$App*" } |
    ForEach-Object { [void]$_.CloseMainWindow() }
for ($i = 0; $i -lt 40 -and (Mine | Where-Object { $_.CommandLine -like '*launcher.pyw*' }); $i++) { Start-Sleep -Milliseconds 500 }
Mine | ForEach-Object { & taskkill.exe /PID $_.ProcessId /T /F | Out-Null }
Get-Process 'FX Workspace' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$App*" } | Stop-Process -Force
Engines | ForEach-Object { & taskkill.exe /PID $_.ProcessId /T /F | Out-Null }
Start-Sleep -Milliseconds 800            # Windows lets go of the files a moment after the processes end
exit 0
