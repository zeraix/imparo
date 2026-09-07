param(
    [Parameter(Mandatory = $true)][string]$CubinW4,
    [Parameter(Mandatory = $true)][string]$SymbolW4,
    [Parameter(Mandatory = $true)][uint32]$DynamicSharedW4,
    [Parameter(Mandatory = $true)][string]$CubinW8,
    [Parameter(Mandatory = $true)][string]$SymbolW8,
    [Parameter(Mandatory = $true)][uint32]$DynamicSharedW8,
    [string]$Nvcc = "nvcc",
    [string]$Policy = "config/kernel-lab-policy.toml",
    [string]$OutputDirectory = "artifacts/kernel-lab/rms-q8-sm86",
    [ValidateSet("all", "measure", "sanitizer")][string]$Mode = "all",
    [double]$ProjectedE2EImprovement = [double]::NaN
)

$ErrorActionPreference = "Stop"
if ($env:IMPARO_CUDA_KERNEL_LAB -ne "1") {
    throw "set IMPARO_CUDA_KERNEL_LAB=1 to build the isolated Phase A2 harness"
}
. (Join-Path $PSScriptRoot "kernel_lab_tools.ps1")

function Read-PolicyRaw([string]$Text, [string]$Name) {
    $escaped = [regex]::Escape($Name)
    $matches = [regex]::Matches($Text, "(?m)^\s*$escaped\s*=\s*([^#\r\n]+?)\s*$")
    if ($matches.Count -ne 1) {
        throw "policy key '$Name' must occur exactly once"
    }
    return $matches[0].Groups[1].Value.Trim()
}

function Read-PolicyInt([string]$Text, [string]$Name) {
    return [int](Read-PolicyRaw $Text $Name)
}

function Read-PolicyDouble([string]$Text, [string]$Name) {
    return [double]::Parse(
        (Read-PolicyRaw $Text $Name),
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Read-PolicyBool([string]$Text, [string]$Name) {
    $raw = Read-PolicyRaw $Text $Name
    if ($raw -eq "true") { return $true }
    if ($raw -eq "false") { return $false }
    throw "policy key '$Name' must be true or false"
}

function Read-PolicyString([string]$Text, [string]$Name) {
    $raw = Read-PolicyRaw $Text $Name
    if ($raw -notmatch '^"(.*)"$') {
        throw "policy key '$Name' must be a quoted string"
    }
    return $Matches[1]
}

function Add-Violation([Collections.Generic.List[string]]$List, [bool]$Condition,
                       [string]$Message) {
    if (-not $Condition) { $List.Add($Message) }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$policyPath = if ([IO.Path]::IsPathRooted($Policy)) {
    $Policy
} else {
    Join-Path $repo $Policy
}
$policyText = Get-Content $policyPath -Raw

$schema = Read-PolicyInt $policyText "schema"
$targetSm = Read-PolicyInt $policyText "target_sm"
$width = Read-PolicyInt $policyText "width"
$tokens = Read-PolicyInt $policyText "n_tok"
$warmup = Read-PolicyInt $policyText "warmup"
$pairs = Read-PolicyInt $policyText "abba_baab_pairs"
$minSamples = Read-PolicyInt $policyText "min_samples_per_route"
$launchesPerSample = Read-PolicyInt $policyText "launches_per_sample"
$eps = Read-PolicyDouble $policyText "eps"
if ($schema -ne 1 -or $targetSm -ne 86 -or $width -ne 2560 -or $tokens -ne 512) {
    throw "the standalone harness implements only policy schema1 SM86 2560x512"
}
$policyDecision = Read-PolicyString $policyText "decision"
$policyInterleaved = Read-PolicyString $policyText "interleaved"
$policyProduction = Read-PolicyBool $policyText "production_authority"
if ($policyDecision -ne "gate-a" -or $policyInterleaved -ne "abba-baab" `
    -or $policyProduction) {
    throw "policy must remain non-production Gate A with ABBA/BAAB scheduling"
}
foreach ($required in @(
    "same_device_required", "same_shape_required", "same_stream_required",
    "cuda_events_required", "canary_required", "sanitizer_required"
)) {
    if (-not (Read-PolicyBool $policyText $required)) {
        throw "policy must require $required"
    }
}
foreach ($forbidden in @(
    "allow_signing", "allow_installer", "allow_receipt_update",
    "allow_graph_integration"
)) {
    if (Read-PolicyBool $policyText $forbidden) {
        throw "Phase A2 policy must set $forbidden=false"
    }
}
if ($warmup -lt 0 -or $pairs -le 0 -or $launchesPerSample -le 0 `
    -or 2 * $pairs -lt $minSamples) {
    throw "policy does not provide its declared minimum sample count"
}

$source = Join-Path $repo "crates/imparo-cuda/native/tests/kernel_lab_rms_q8.cu"
$catalog = Get-Content (Join-Path $repo "crates/imparo-cuda/cuda-sm.json") -Raw |
    ConvertFrom-Json
if ($catalog.sms -notcontains 86) {
    throw "cuda-sm.json does not declare SM86"
}
$output = Join-Path $repo $OutputDirectory
New-Item -ItemType Directory -Force -Path $output | Out-Null
$exe = Join-Path $output "kernel_lab_rms_q8_sm86.exe"
$result = Join-Path $output "result.json"
$decisionPath = Join-Path $output "gate-a-decision.json"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhereCommand = Get-Command "vswhere.exe" -ErrorAction SilentlyContinue
    $vswhere = if ($vswhereCommand -and $vswhereCommand.Source) {
        $vswhereCommand.Source
    } else {
        @("ProgramFiles(x86)", "ProgramFiles") |
            ForEach-Object {
                $programFilesRoot = [Environment]::GetEnvironmentVariable($_)
                if ($programFilesRoot) {
                    Join-Path $programFilesRoot "Microsoft Visual Studio\Installer\vswhere.exe"
                }
            } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Select-Object -First 1
    }
    if (-not $vswhere) {
        throw "Visual Studio Installer discovery tool vswhere.exe not found"
    }
    $vs = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vs) { throw "Visual Studio C++ Build Tools not found" }
    $vcvars = Join-Path $vs "VC/Auxiliary/Build/vcvars64.bat"
    $lines = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
    foreach ($line in $lines) {
        $split = $line.IndexOf('=')
        if ($split -gt 0) {
            [Environment]::SetEnvironmentVariable(
                $line.Substring(0, $split), $line.Substring($split + 1), "Process"
            )
        }
    }
}

$nvccCommand = (Get-Command $Nvcc -ErrorAction Stop).Source
$zeroSha = "0" * 64
$nvccArgs = @(
    "-O3", "-std=c++17", "--use_fast_math", "-lineinfo",
    "-cudart=shared", "-Xcompiler", "/MD",
    "-DIMPARO_CUDA_KERNEL_LAB=1",
    "-DIMPARO_CUDA_BACKEND_ABI=$($catalog.backend_abi)",
    "-DIMPARO_CUDA_BUILD_SHA256=\`"$zeroSha\`"",
    "--generate-code", "arch=compute_86,code=[sm_86,compute_86]",
    $source, "-o", $exe
)
& $nvccCommand @nvccArgs
if ($LASTEXITCODE -ne 0) {
    throw "NVCC failed with exit code $LASTEXITCODE"
}

$sanitizerPassed = $false
$sanitizerLogs = [ordered]@{}
if ($Mode -in @("all", "sanitizer")) {
    $sanitizer = Resolve-ComputeSanitizer -NvccPath $nvccCommand
    $sanitizerPassed = $true
    foreach ($tool in @("memcheck", "initcheck", "racecheck", "synccheck")) {
        $toolLog = Join-Path $output "compute-sanitizer-$tool.log"
        $toolOutput = & $sanitizer --tool $tool --error-exitcode 86 `
            $exe $CubinW4 $SymbolW4 $CubinW8 $SymbolW8 `
            $DynamicSharedW4 $DynamicSharedW8 $eps 0 1 1 2>&1
        $toolExit = $LASTEXITCODE
        [IO.File]::WriteAllLines($toolLog, [string[]]$toolOutput)
        $sanitizerLogs[$tool] = [ordered]@{
            path = $toolLog
            exit_code = $toolExit
            pass = $toolExit -eq 0
        }
        if ($toolExit -ne 0) { $sanitizerPassed = $false }
    }
    if ($Mode -eq "sanitizer") {
        if (-not $sanitizerPassed) {
            throw "one or more compute-sanitizer tools failed; see $output"
        }
        Write-Host "LAB-A sanitizer results: $output"
        exit 0
    }
}

$runOutput = & $exe $CubinW4 $SymbolW4 $CubinW8 $SymbolW8 `
    $DynamicSharedW4 $DynamicSharedW8 $eps $warmup $pairs $launchesPerSample
$runExit = $LASTEXITCODE
if ($runExit -ne 0) {
    if ($null -ne $runOutput) {
        [IO.File]::WriteAllText($result, [string]$runOutput + "`n")
    }
    throw "LAB-A harness target failed with exit code $runExit"
}
if ($runOutput.Count -ne 1) {
    throw "LAB-A harness must emit exactly one JSON line"
}
[IO.File]::WriteAllText($result, [string]$runOutput + "`n")
$data = [string]$runOutput | ConvertFrom-Json
$violations = [Collections.Generic.List[string]]::new()

Add-Violation $violations ($data.schema -eq 1) "result schema mismatch"
Add-Violation $violations ($data.phase -eq "A2") "result phase mismatch"
Add-Violation $violations (-not $data.production_enabled) "production must remain disabled"
Add-Violation $violations ($data.target_sm -eq $targetSm) "target SM mismatch"
Add-Violation $violations ($data.width -eq $width -and $data.n_tok -eq $tokens) `
    "shape mismatch"
Add-Violation $violations ($data.same_primary_context -and $data.same_stream) `
    "native and Triton must share context and stream"
Add-Violation $violations ($data.separate_outputs) "routes must use separate outputs"
Add-Violation $violations ($data.timing_clock -eq "cuda-events") `
    "timing must use CUDA events"
Add-Violation $violations ($data.interleaved_schedule -eq "abba-baab") `
    "timing must be ABBA/BAAB interleaved"
Add-Violation $violations ($data.launches_per_sample -eq $launchesPerSample) `
    "launches-per-sample mismatch"
Add-Violation $violations $data.structural_ok "structural checks failed"
Add-Violation $violations ($data.all_canary_errors -le `
    (Read-PolicyInt $policyText "max_canary_errors")) "canary corruption"
Add-Violation $violations $sanitizerPassed "compute-sanitizer did not pass"

$correctVariants = 0
$fastVariants = 0
foreach ($variant in $data.variants) {
    $prefix = "warps=$($variant.warps)"
    $variantCorrect = $true
    $checks = @(
        @(($variant.normalized_vs_native.max_abs -le `
          (Read-PolicyDouble $policyText "max_normalized_abs_vs_native")),
          "$prefix normalized max_abs"),
        @(($variant.normalized_vs_native.max_rel -le `
          (Read-PolicyDouble $policyText "max_normalized_rel_vs_native")),
          "$prefix normalized max_rel"),
        @(($variant.q8_scales_vs_native.max_abs -le `
          (Read-PolicyDouble $policyText "max_q8_scale_abs_vs_native")),
          "$prefix Q8 scale max_abs"),
        @(($variant.q8_value_max_abs -le `
          (Read-PolicyInt $policyText "max_q8_value_abs_vs_native")),
          "$prefix Q8 value max_abs"),
        @(($variant.q8_byte_agreement -ge `
          (Read-PolicyDouble $policyText "min_q8_byte_agreement")),
          "$prefix Q8 byte agreement"),
        @(($variant.q8_dequant.max_abs -le `
          (Read-PolicyDouble $policyText "max_q8_dequant_abs")),
          "$prefix Q8 dequant max_abs"),
        @(($variant.normalized_vs_native.non_finite -le `
          (Read-PolicyInt $policyText "max_non_finite")),
          "$prefix non-finite normalized values"),
        @(($variant.resources.registers_per_thread -le `
          (Read-PolicyInt $policyText "max_registers_per_thread")),
          "$prefix register ceiling"),
        @((($variant.resources.static_shared_bytes + $variant.dynamic_shared_bytes) -le `
          (Read-PolicyInt $policyText "max_static_shared_bytes")),
          "$prefix total shared-memory ceiling"),
        @(($variant.resources.local_bytes -le `
          (Read-PolicyInt $policyText "max_local_memory_bytes")),
          "$prefix local-memory ceiling"),
        @((($variant.timing.native_samples_us.Count -ge $minSamples) -and `
          ($variant.timing.triton_samples_us.Count -ge $minSamples)),
          "$prefix sample count"),
        @(($variant.timing.native_cv -le `
          (Read-PolicyDouble $policyText "max_native_cv")),
          "$prefix native timing CV"),
        @(($variant.timing.triton_cv -le `
          (Read-PolicyDouble $policyText "max_candidate_cv")),
          "$prefix candidate timing CV"),
        @(($variant.timing.paired_mad_fraction -le `
          (Read-PolicyDouble $policyText "max_paired_mad_fraction")),
          "$prefix paired MAD")
    )
    foreach ($check in $checks) {
        if (-not $check[0]) {
            $variantCorrect = $false
            $violations.Add([string]$check[1])
        }
    }
    if ($variantCorrect) { ++$correctVariants }
    if ($variant.timing.speedup_median -ge `
        (Read-PolicyDouble $policyText "min_kernel_speedup_ratio")) {
        ++$fastVariants
    }
}
Add-Violation $violations ($correctVariants -eq $data.variants.Count) `
    "not all variants passed correctness/noise/resource gates"
$kernelFloorMet = $fastVariants -gt 0
$projectedFloorMet = -not [double]::IsNaN($ProjectedE2EImprovement) -and `
    $ProjectedE2EImprovement -ge `
        (Read-PolicyDouble $policyText "min_projected_e2e_improvement")
Add-Violation $violations ($kernelFloorMet -or $projectedFloorMet) `
    "candidate reached neither kernel nor projected end-to-end floor"

$decision = [ordered]@{
    schema = 1
    decision = "gate-a-candidate"
    production_authority = $false
    final_gate_a_decision = $false
    candidate_admissible = $violations.Count -eq 0
    policy = $policyPath
    result = $result
    sanitizer = $sanitizerLogs
    projected_e2e_improvement = if ([double]::IsNaN($ProjectedE2EImprovement)) {
        $null
    } else {
        $ProjectedE2EImprovement
    }
    launches_per_sample = $launchesPerSample
    correct_variants = $correctVariants
    fast_variants = $fastVariants
    kernel_floor_met = $kernelFloorMet
    projected_e2e_floor_met = $projectedFloorMet
    violations = [string[]]$violations
}
[IO.File]::WriteAllText(
    $decisionPath, ($decision | ConvertTo-Json -Depth 8) + "`n"
)
Write-Host "LAB-A result: $result"
Write-Host "LAB-A decision: $decisionPath"
if (-not $decision.candidate_admissible) {
    throw "Gate A candidate is inadmissible; see $decisionPath"
}
