# FX Workspace - first-time setup (run install.bat, which runs this).
#
# 1. Python: finds Python 3.11 or 3.12 (offers to install 3.11 with winget when there is none).
# 2. .venv with the pinned packages from requirements.txt.
# 3. local\ (or AGENTIC_HOME): llama.cpp and models live here (paths.py).
# 4. llama.cpp: the newest Windows build for this PC - CUDA 12.4 with an NVIDIA GPU, else Vulkan, else CPU -
#    from github.com/ggml-org/llama.cpp, checked against GitHub's SHA-256. Asks first.
# 5. A starter chat model (Qwen3 4B, ~2.4 GB, from huggingface.co/Qwen), checked by SHA-256. Asks first.
# 6. models.ini from models.example.ini, with every .gguf in local\models added.
# 7. A desktop shortcut. Asks first.
# 8. Tidy up (update_cleanup.py): packages an older version needed and this one doesn't, __pycache__, old
#    models.ini backups. pip keeps no download cache (it grew by hundreds of MB with every update).
# Run it again any time: what's already there is kept.
#
# Unattended: install.ps1 -Yes [-SkipLlama] [-SkipModel] [-SkipShortcut] [-Images]  (answers every question itself;
#             the ~22 GB image models only with -Images)
# -Update (the setup wizard, updating): also brings the app's own llama.cpp (local\llama.cpp) to the newest build -
#             replaced in place, the old build's files removed; kept as it is if the new one doesn't start.
# -Quiet:     plain line-by-line output, no download progress bars (the setup wizard shows each line).
param([switch]$Yes, [switch]$SkipLlama, [switch]$SkipModel, [switch]$SkipShortcut, [switch]$Images,
      [switch]$Update, [switch]$Quiet)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PIP_NO_CACHE_DIR = '1'          # pip's cache is never reused here - it only grew with each update
$env:PIP_PROGRESS_BAR = if ($Quiet) { 'off' } else { 'on' }
$env:PYTHONUNBUFFERED = '1'          # each line shows as it happens, also on the wizard's progress page
Set-Location -LiteralPath $PSScriptRoot
$App = $PSScriptRoot
$HomeDir = if ($env:AGENTIC_HOME) { $env:AGENTIC_HOME } else { Join-Path $App 'local' }
$Headers = @{ 'User-Agent' = 'agentic-workspace-installer' }

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }
function Step($text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }
function Ask($question, $default = 'Y', [switch]$Skip) {
    if ($Skip) { Say "$question - skipped"; return $false }
    if ($Yes) { Say "$question - yes"; return $true }
    $hint = if ($default -eq 'Y') { '[Y/n]' } else { '[y/N]' }
    $a = Read-Host "$question $hint"
    if ([string]::IsNullOrWhiteSpace($a)) { $a = $default }
    return $a.Trim().ToUpper().StartsWith('Y')
}
function Download($url, $dest, $sha256) {
    # curl.exe (in Windows 10 and 11) resumes a broken download and shows progress.
    Say "   $url"
    $show = @(if ($Quiet) { '-sS' })                 # an array, even of one: splatting a string passes its letters
    & curl.exe -L --fail --retry 3 @show -C - -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    if ($sha256) {
        $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLower()
        if ($got -ne $sha256.ToLower()) { Remove-Item -LiteralPath $dest -Force; throw "Checksum mismatch for $dest (deleted)." }
        Say '   SHA-256 verified.' 'Green'
    }
}

Say 'FX Workspace - setup' 'White'

# ---- 0. A downloaded source folder: the setup wizard is the better way -------------------------
# install.bat + run.bat are for working on the code. Someone who downloaded the source zip gets the real app from
# the setup wizard: its own window (no browser), a Start-menu entry, uninstall, and Update now in Settings > About.
if (-not $Yes -and -not (Test-Path (Join-Path $App '.git')) -and -not (Test-Path (Join-Path $App 'desktop-app'))) {
    Say 'This is the source code. The setup wizard installs FX Workspace as a normal Windows app - its own window,' 'Yellow'
    Say 'a Start-menu entry and updates from inside the app.' 'Yellow'
    if (Ask 'Download the newest setup wizard from GitHub and run it instead (recommended)?') {
        $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/fuxion64-ctrl/FX-Workspace/releases/latest' -Headers $Headers
        $a = $rel.assets | Where-Object { $_.name -match '^FX-Workspace-Setup-[\d.]+\.exe$' } | Select-Object -First 1
        if (-not $a -or $a.digest -notmatch '^sha256:(.+)$') { throw 'The newest release has no setup wizard with a checksum - get it from github.com/fuxion64-ctrl/FX-Workspace/releases.' }
        $setup = Join-Path $env:TEMP $a.name
        Download $a.browser_download_url $setup $Matches[1]
        Start-Process -FilePath $setup
        Say 'The setup wizard is open. You can delete this folder once it has finished.' 'Green'
        exit 0
    }
}

Say "App folder:   $App"
Say "Models/tools: $HomeDir  (set AGENTIC_HOME to use another drive)"

# ---- 1. Python ---------------------------------------------------------------------------------
# (1.4.8) Setup no longer stops at "install Python, then run install.bat again": a missing Python was installed with
# winget and then setup ended with an error anyway (the new Python isn't on this process's PATH), and a PC without
# winget got only a link. Now Python is looked for everywhere it lands - the py launcher, PATH, the registry, the
# usual folders - and when there is none, python.org's own installer is fetched, checked (its Python Software
# Foundation signature), installed for this user only (no admin) and used straight away.
Step 'Python'
function Test-Python($exe, [string[]]$pre = @()) {
    # A real Python 3.11/3.12 - not the Microsoft Store stub (WindowsApps\python.exe opens the Store and prints nothing).
    try {
        if (-not $exe) { return $false }
        if ($exe -like '*\WindowsApps\*') { return $false }
        $ver = & $exe @pre -c "import sys, venv; print('%d.%d' % sys.version_info[:2])" 2>$null
        return ($LASTEXITCODE -eq 0) -and ($ver -in '3.11', '3.12')
    } catch { return $false }
}
function Find-Python {
    foreach ($v in '3.11', '3.12') {
        foreach ($py in @('py', (Join-Path $env:WINDIR 'py.exe'), (Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'))) {
            if ((Get-Command $py -ErrorAction SilentlyContinue) -and (Test-Python $py @("-$v"))) { return @($py, "-$v") }
        }
    }
    $cands = @()
    foreach ($v in '3.11', '3.12') {
        foreach ($root in 'HKCU:', 'HKLM:', 'HKLM:\SOFTWARE\WOW6432Node') {
            $key = if ($root -eq 'HKLM:\SOFTWARE\WOW6432Node') { "$root\Python\PythonCore\$v\InstallPath" } else { "$root\SOFTWARE\Python\PythonCore\$v\InstallPath" }
            try { $p = (Get-ItemProperty -Path $key -ErrorAction Stop).'(default)'; if ($p) { $cands += (Join-Path $p 'python.exe') } } catch {}
        }
        $n = $v -replace '\.', ''
        $cands += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$n\python.exe"), (Join-Path $env:ProgramFiles "Python$n\python.exe")
    }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $cands = @($onPath.Source) + $cands }
    foreach ($c in $cands) { if ((Test-Path -LiteralPath $c) -and (Test-Python $c)) { return @($c) } }
    return $null
}
function Install-Python {
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'amd64' }
    $name = "python-3.11.9-$arch.exe"                      # the last 3.11 release with a Windows installer
    $setup = Join-Path $env:TEMP $name
    try {
        Download "https://www.python.org/ftp/python/3.11.9/$name" $setup $null
        $sig = Get-AuthenticodeSignature -FilePath $setup
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
            Remove-Item -LiteralPath $setup -Force
            throw "the downloaded installer isn't signed by the Python Software Foundation (deleted)"
        }
        Say '   Signed by the Python Software Foundation - verified.' 'Green'
        Say '   Installing Python 3.11 for this user (no admin needed) ...'
        $p = Start-Process -FilePath $setup -Wait -PassThru -ArgumentList @('/quiet', 'InstallAllUsers=0', 'PrependPath=0',
            'Include_test=0', 'Include_doc=0', 'Include_launcher=1', 'InstallLauncherAllUsers=0', 'Shortcuts=0')
        Remove-Item -LiteralPath $setup -Force -ErrorAction SilentlyContinue
        if ($p.ExitCode -ne 0) { throw "the Python installer ended with code $($p.ExitCode)" }
        return $true
    } catch {
        Say "   python.org's installer didn't work: $_" 'Yellow'
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Say '   Trying winget instead ...'
        winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
        return ($LASTEXITCODE -eq 0)
    }
    return $false
}
$Py = Find-Python
if (-not $Py) {
    Say 'Python 3.11 or 3.12 was not found.' 'Yellow'
    if (Ask 'Download and install Python 3.11 from python.org now (about 25 MB, for this user only)?') {
        $null = Install-Python
        $Py = Find-Python                                    # found by its folder or the registry - no restart needed
    }
    if (-not $Py) {
        Say 'Python could not be installed automatically. Install Python 3.11 from https://www.python.org/downloads/' 'Yellow'
        Say 'and run this setup again - nothing else is needed.' 'Yellow'
        exit 1
    }
}
Say "Using: $($Py -join ' ')" 'Green'

# ---- 2. The environment ------------------------------------------------------------------------
Step 'Python packages'
$VenvPy = Join-Path $App '.venv\Scripts\python.exe'
# (1.4.8) A .venv whose Python no longer runs (the Python it was made from was removed or moved) is made again.
if ((Test-Path $VenvPy) -and -not (Test-Python $VenvPy)) {
    Say 'The Python environment is broken (its Python was removed or moved) - making it again ...' 'Yellow'
    Remove-Item -LiteralPath (Join-Path $App '.venv') -Recurse -Force
}
if (-not (Test-Path $VenvPy)) {
    Say 'Creating .venv ...'
    $PyArgs = @($Py | Select-Object -Skip 1)
    & $Py[0] @PyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}
& $VenvPy -m pip install --disable-pip-version-check --upgrade pip | Out-Null
& $VenvPy -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    # (1.4.8) A dropped connection or a busy mirror: once more before giving up.
    Say 'Installing the packages failed - trying once more ...' 'Yellow'
    Start-Sleep -Seconds 3
    & $VenvPy -m pip install --disable-pip-version-check --retries 5 --timeout 60 -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Installing the packages failed (see above) - check the internet connection and run setup again.' }
Say 'Packages installed.' 'Green'
& $VenvPy update_cleanup.py packages          # what an older version needed and this one doesn't

# ---- 3. Folders --------------------------------------------------------------------------------
$LlamaDir = Join-Path $HomeDir 'llama.cpp'
$ModelsDir = Join-Path $HomeDir 'models'
New-Item -ItemType Directory -Force -Path $LlamaDir, $ModelsDir | Out-Null

# ---- 4. llama.cpp ------------------------------------------------------------------------------
Step 'llama.cpp (runs the chat models)'
# The newest Windows build of one kind (cuda-12.4 / vulkan): @{ release; zips; build = its b-number }.
function Find-LlamaBuild($kind) {
    $releases = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers $Headers
    foreach ($r in $releases) {
        $main = $r.assets | Where-Object { $_.name -match "^llama-b(\d+)-bin-win-$([regex]::Escape($kind))-x64\.zip$" } | Select-Object -First 1
        if (-not $main) { continue }
        $build = [int]($main.name -replace '^llama-b(\d+)-.*$', '$1')
        $zips = @($main)
        if ($kind -like 'cuda*') {
            $rt = $r.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-$kind-x64.zip" } | Select-Object -First 1
            if ($rt) { $zips += $rt }
        }
        return @{ release = $r; zips = $zips; build = $build }
    }
    throw "No Windows $kind build found in the recent llama.cpp releases."
}
# (1.4.8) The GPUs a build can use. 10 Nov 2026: an update ran out of disk space while unpacking NVIDIA's runtime
# (cudart/cublas), the new build still "started", and every model ran on the CPU at 4 tokens/s. A CUDA build must
# list a CUDA device, or it isn't installed.
function Get-LlamaDevices($exe) {
    $ErrorActionPreference = 'Continue'
    try { $out = & $exe --list-devices 2>&1 | ForEach-Object { "$_" } | Out-String } catch { return $null }
    if ($out -notmatch 'Available devices') { return $null }       # an older build: can't tell
    return @([regex]::Matches($out, '(?m)^\s*((?:CUDA|Vulkan)\d+:.*)$') | ForEach-Object { $_.Groups[1].Value })
}
function Test-NvidiaGpu {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    & nvidia-smi -L 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}
# Room for the downloads and what they unpack to (about 2.5 times the zips), on the drive they land on.
function Assert-Room($pick, $where) {
    $need = [long](($pick.zips | Measure-Object -Property size -Sum).Sum * 2.5)
    $drive = (Get-Item -LiteralPath $where).PSDrive
    if ($drive -and $drive.Free -lt $need) {
        throw ("not enough disk space on {0}: (llama.cpp needs about {1:N1} GB free, there is {2:N1} GB)" -f
               $drive.Name, ($need / 1GB), ($drive.Free / 1GB))
    }
}
# Downloads release zips (checked by their SHA-256) and unpacks them into $dest, after checking there is room.
function Get-ZipsInto($pick, $dest) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Assert-Room $pick (Split-Path -Parent $dest)
    foreach ($a in $pick.zips) {
        $zip = Join-Path $HomeDir $a.name
        $sha = if ($a.digest -match '^sha256:(.+)$') { $Matches[1] } else { $null }
        Download $a.browser_download_url $zip $sha
        Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
        Remove-Item -LiteralPath $zip -Force
    }
}
# Downloads and unpacks a build into $dest (empty), with llama-server.exe and its DLLs at its top level.
function Get-LlamaInto($pick, $dest) {
    Get-ZipsInto $pick $dest
    # Some builds unpack into a subfolder: bring llama-server.exe and its DLLs up.
    $exe = Get-ChildItem -LiteralPath $dest -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
    if ($exe -and $exe.DirectoryName -ne $dest) { Get-ChildItem -LiteralPath $exe.DirectoryName | Move-Item -Destination $dest -Force }
    if (-not (Test-Path (Join-Path $dest 'llama-server.exe'))) { throw 'llama-server.exe was not in the download.' }
    if (($pick.zips | Where-Object { $_.name -like 'llama-*cuda*' }) -and (Test-NvidiaGpu)) {
        $dev = Get-LlamaDevices (Join-Path $dest 'llama-server.exe')
        if ($dev -ne $null -and -not ($dev | Where-Object { $_ -like 'CUDA*' })) {
            throw 'the CUDA build cannot see the GPU (NVIDIA runtime files missing or damaged - is the disk full?)'
        }
    }
}
# The build number llama-server reports - "version: 0.6.0-dev (build 11541, commit ...)" now, "version: 6912
# (abc123)" in older builds - or 0.
function Get-LlamaVersion($exe) {
    $ErrorActionPreference = 'Continue'            # it prints to stderr, which 'Stop' would turn into an error
    try { $out = & $exe --version 2>&1 | ForEach-Object { "$_" } | Out-String } catch { return 0 }
    if ($out -match '\bbuild\s+(\d+)') { return [int]$Matches[1] }
    if ($out -match 'version:\s*(\d+)\s*\(') { return [int]$Matches[1] }
    return 0
}

$Server = Join-Path $LlamaDir 'llama-server.exe'
if (Test-Path $Server) {
    if ($Update -and -not $SkipLlama) {
        # Same kind as before (a CUDA build has ggml-cuda.dll beside it). Only the files at the top of
        # local\llama.cpp are the build's - PrismML's build in local\llama.cpp\prism stays as it is.
        $kind = if (Test-Path (Join-Path $LlamaDir 'ggml-cuda.dll')) { 'cuda-12.4' } else { 'vulkan' }
        $have = Get-LlamaVersion $Server
        try { $pick = Find-LlamaBuild $kind } catch { $pick = $null; Say "Couldn't check for a newer llama.cpp ($($_.Exception.Message)) - keeping build $have." 'Yellow' }
        if ($pick -and $have -and $pick.build -le $have) {
            Say "llama.cpp is up to date (build $have)." 'Green'
        } elseif ($pick) {
            Say "Updating llama.cpp: build $have -> $($pick.build)"
            $new = Join-Path $HomeDir 'llama.cpp.new'
            if (Test-Path $new) { Remove-Item -LiteralPath $new -Recurse -Force }
            try {
                Get-LlamaInto $pick $new
                if (-not (Get-LlamaVersion (Join-Path $new 'llama-server.exe'))) { throw 'the new build did not start' }
                Get-ChildItem -LiteralPath $LlamaDir -File | Remove-Item -Force
                Get-ChildItem -LiteralPath $new | Move-Item -Destination $LlamaDir -Force
                Say "llama.cpp updated to build $($pick.build); the old build's files were removed." 'Green'
            } catch {
                Say "llama.cpp wasn't updated ($($_.Exception.Message)) - build $have stays." 'Yellow'
            } finally {
                if (Test-Path $new) { Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue }
            }
        }
    } else {
        Say "Already installed in $LlamaDir." 'Green'
    }
    # (1.4.8) A CUDA build that can't see the GPU gets NVIDIA's runtime again - the one its own release ships.
    if (-not $SkipLlama -and (Test-Path (Join-Path $LlamaDir 'ggml-cuda.dll')) -and (Test-NvidiaGpu)) {
        $dev = Get-LlamaDevices $Server
        if ($dev -ne $null -and -not ($dev | Where-Object { $_ -like 'CUDA*' })) {
            Say 'llama.cpp cannot see the GPU (its NVIDIA runtime files are missing or damaged) - repairing ...' 'Yellow'
            $fix = Join-Path $HomeDir 'llama.cpp.runtime'
            try {
                $b = Get-LlamaVersion $Server
                $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b$b" -Headers $Headers
                $rt = $rel.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip' } | Select-Object -First 1
                if (-not $rt) { throw "release b$b has no CUDA runtime to repair it with" }
                if (Test-Path $fix) { Remove-Item -LiteralPath $fix -Recurse -Force }
                Get-ZipsInto @{ zips = @($rt) } $fix
                Get-ChildItem -LiteralPath $fix -Recurse -Filter '*.dll' | Copy-Item -Destination $LlamaDir -Force
                $dev = Get-LlamaDevices $Server
                if ($dev | Where-Object { $_ -like 'CUDA*' }) { Say "Repaired: $($dev[0])" 'Green' }
                else { Say 'The repair did not help - reinstall llama.cpp (delete local\llama.cpp and run setup again).' 'Yellow' }
            } catch {
                Say "Repair failed: $($_.Exception.Message)" 'Yellow'
            } finally {
                if (Test-Path $fix) { Remove-Item -LiteralPath $fix -Recurse -Force -ErrorAction SilentlyContinue }
            }
        }
    }
} elseif (Ask 'Download the newest llama.cpp for this PC?' -Skip:$SkipLlama) {
    $nvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($nvidia) { & nvidia-smi -L 2>$null | Out-Null; $nvidia = ($LASTEXITCODE -eq 0) }
    $kind = if ($nvidia) { 'cuda-12.4' } else { 'vulkan' }
    Say ("GPU: " + $(if ($nvidia) { 'NVIDIA - the CUDA 12.4 build' } else { 'not NVIDIA - the Vulkan build (AMD, Intel, or any GPU)' }))
    $pick = Find-LlamaBuild $kind
    Say "Release: $($pick.release.tag_name)"
    Get-LlamaInto $pick $LlamaDir
    Say "llama.cpp installed in $LlamaDir." 'Green'
} else {
    Say "Skipped. Put llama-server.exe (and its DLLs) in $LlamaDir later - or use Cloud mode with an API key."
}

# ---- 5. A starter model ------------------------------------------------------------------------
Step 'A chat model'
$have = @(Get-ChildItem -LiteralPath $ModelsDir -Filter '*.gguf' -ErrorAction SilentlyContinue)
if ($have.Count) {
    Say "Found $($have.Count) model file(s) in $ModelsDir." 'Green'
} elseif (Ask 'Download a starter model - Qwen3 4B (2.4 GB, runs on 4 GB+ of VRAM)?' -Skip:$SkipModel) {
    Download 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf' `
        (Join-Path $ModelsDir 'Qwen3-4B-Q4_K_M.gguf') '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
} else {
    Say "Skipped. Put any .gguf model in $ModelsDir, then press Settings > Models > Scan for models."
}

# ---- 5b. Image generation (optional) -----------------------------------------------------------
Step 'Image generation (optional)'
$ImagesDir = Join-Path $HomeDir 'Images'
$gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if (Test-Path (Join-Path $ImagesDir 'manifest.json')) {
    Say "Already set up in $ImagesDir." 'Green'
} elseif (-not $gpu) {
    Say 'Skipped: the image engine needs an NVIDIA GPU (CUDA).'
} elseif (Ask 'Set up local image generation - FLUX.2 klein, Z-Image-Turbo and Krea 2 (about 22 GB, every file checksum-verified)?' 'N' -Skip:($Yes -and -not $Images)) {
    New-Item -ItemType Directory -Force -Path $ImagesDir | Out-Null
    & $VenvPy image_setup.py $ImagesDir
    if ($LASTEXITCODE -ne 0) { Say 'Image setup finished with errors - run it again later: .venv\Scripts\python.exe image_setup.py' 'Yellow' }
} else {
    Say 'Skipped. Later: .venv\Scripts\python.exe image_setup.py   (Settings > Images explains the models)'
}

# ---- 6. models.ini -----------------------------------------------------------------------------
Step 'Model list (models.ini)'
# (1.4.8) core.modpath first: model_scan lives in llm\ since the folders split, and this step failed on every setup.
& $VenvPy -c "import core.modpath, model_scan, json; p = model_scan.ensure_ini(); r = model_scan.scan(); print('models.ini:', p); print('added:', ', '.join(a['name'] for a in r['added']) or 'nothing new')"

# ---- 7. Shortcut -------------------------------------------------------------------------------
Step 'Shortcut'
$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'FX Workspace.lnk'
if (-not (Test-Path $lnk) -and (Ask 'Put a shortcut on the desktop?' -Skip:$SkipShortcut)) {
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = Join-Path $App 'run.bat'
    $s.WorkingDirectory = $App
    $s.WindowStyle = 7
    $s.Save()
    Say 'Shortcut created.' 'Green'
}

# ---- 8. Tidy up --------------------------------------------------------------------------------
Step 'Tidying up'
& $VenvPy update_cleanup.py tidy

Write-Host ''
Say 'Done. Start the app with run.bat (or the desktop shortcut).' 'Green'
Say 'Optional: Settings > Voice has a one-click setup; images: .venv\Scripts\python.exe image_setup.py (see README).'
Say 'Cloud models: put API keys in keys.local.bat (created on the first start).'
exit 0          # the setup wizard reads the exit code; a probe that failed on the way (py, nvidia-smi) isn't a failure
