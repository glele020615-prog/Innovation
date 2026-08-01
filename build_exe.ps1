Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonExe = $null
$pythonArgs = @()

if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = 'python'
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = 'py'
    $pythonArgs = @('-3')
}
else {
    throw 'python or py not found. Install Python or run this inside a configured environment.'
}

function Test-PyInstaller {
    & $pythonExe @pythonArgs -m PyInstaller --version | Out-Null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-PyInstaller)) {
    Write-Host 'PyInstaller not found. Installing...'
    & $pythonExe @pythonArgs -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw 'pip upgrade failed.'
    }
    & $pythonExe @pythonArgs -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller installation failed.'
    }
}

$appName = 'ADDiagnosisSystem'
$candidateScripts = @(
    Get-ChildItem -Path $scriptDir -Filter '*.py' | Where-Object {
    $_.Name -notin @(
        'Data_Pre.py',
        'Read_Write_JSON.py',
        'SelfAttentionBlock_ST_GCN.py',
        'feature_extraction_page.py',
        'individual_report_page.py',
        'predict.py',
        'viewer.py',
        'explainability_utils.py',
        'report_explain_utils.py'
    )
    }
)

if ($candidateScripts.Count -ne 1) {
    throw "Unable to determine the entry script. Found $($candidateScripts.Count) candidates in the project root."
}

$entryScript = $candidateScripts[0].FullName

$dataArgs = @(
    '--add-data', 'assets;assets',
    '--add-data', 'scan_history.json;.',
    '--add-data', 'best_acc_model_fold_1.pt;.',
    '--add-data', 'label_order_jian.node;.',
    '--add-data', 'Node_AAL116.node;.'
)

$pyinstallerArgs = @(
    '-m', 'PyInstaller',
    '--noconfirm',
    '--clean',
    '--onedir',
    '--noconsole',
    '--name', $appName,
    '--distpath', (Join-Path $scriptDir 'dist'),
    '--workpath', (Join-Path $scriptDir 'build'),
    '--specpath', $scriptDir,
    '--paths', $scriptDir
) + $dataArgs + @($entryScript)

Write-Host "Building: $entryScript"
& $pythonExe @pythonArgs @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller build failed.'
}

Write-Host ''
Write-Host 'Build complete. Executable path:'
$outputDir = Join-Path -Path (Join-Path -Path $scriptDir -ChildPath 'dist') -ChildPath $appName
Write-Host (Join-Path -Path $outputDir -ChildPath "$appName.exe")
