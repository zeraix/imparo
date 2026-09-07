function Resolve-ComputeSanitizer {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$NvccPath)

    foreach ($name in @(
        "compute-sanitizer", "compute-sanitizer.exe", "compute-sanitizer.bat"
    )) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -and (Test-Path -LiteralPath $command.Source)) {
            return (Resolve-Path -LiteralPath $command.Source).Path
        }
    }

    $nvccItem = Get-Item -LiteralPath $NvccPath -ErrorAction Stop
    $toolkitRoot = Split-Path -Parent $nvccItem.Directory.FullName
    $toolkitExecutable = Join-Path $toolkitRoot `
        "compute-sanitizer/compute-sanitizer.exe"
    if (Test-Path -LiteralPath $toolkitExecutable) {
        return (Resolve-Path -LiteralPath $toolkitExecutable).Path
    }

    throw "compute-sanitizer was not found in PATH or the CUDA toolkit containing $NvccPath"
}
