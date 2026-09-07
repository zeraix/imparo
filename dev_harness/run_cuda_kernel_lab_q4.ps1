param(
    [string]$Metadata = "program-packs/lab/q4-q8-mmq-sm86/run-b/lab-metadata.json",
    [string]$VariantMetadata = "",
    [string]$ExpansionConfigId = "",
    [string]$ContractionConfigId = "",
    [string]$Policy = "config/kernel-lab-policy.toml",
    [string]$Nvcc = "nvcc",
    [string]$Cuobjdump = "cuobjdump",
    [string]$OutputDirectory = "artifacts/kernel-lab/q4-q8-mmq-sm86/run-b",
    [ValidateSet("preflight", "smoke", "measure", "sanitizer", "all")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"
if ($env:IMPARO_CUDA_KERNEL_LAB -ne "1") {
    throw "set IMPARO_CUDA_KERNEL_LAB=1 to build the isolated Phase A2 harness"
}
. (Join-Path $PSScriptRoot "kernel_lab_tools.ps1")
. (Join-Path $PSScriptRoot "kernel_lab_q4_contract.ps1")
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $repo $Path
}

$policyPath = Resolve-RepoPath $Policy
$policyText = Get-Content -LiteralPath $policyPath -Raw
if ((Read-Q4PolicyInt $policyText "schema") -ne 1 -or
    (Read-Q4PolicyString $policyText "phase") -ne "A2" -or
    (Read-Q4PolicyString $policyText "decision") -ne "gate-a") {
    throw "Q4 runner requires schema1 Phase A2 Gate-A policy"
}
$variantPairMode = -not [string]::IsNullOrWhiteSpace($VariantMetadata)
$hasExpansionId = -not [string]::IsNullOrWhiteSpace($ExpansionConfigId)
$hasContractionId = -not [string]::IsNullOrWhiteSpace($ContractionConfigId)
if (($variantPairMode -and (-not $hasExpansionId -or
        -not $hasContractionId)) -or
    (-not $variantPairMode -and ($hasExpansionId -or $hasContractionId))) {
    throw "variant-pair mode requires metadata plus both expansion/contraction config IDs"
}
$metadataPath = Resolve-RepoPath $(if ($variantPairMode) {
    $VariantMetadata
} else {
    $Metadata
})
$artifacts = if ($variantPairMode) {
    $variantContract = @{
        MetadataPath = $metadataPath
        PolicyText = $policyText
        ExpansionConfigId = $ExpansionConfigId
        ContractionConfigId = $ContractionConfigId
    }
    @(Assert-Q4VariantPairContract @variantContract)
} else {
    $baselineContract = @{
        MetadataPath = $metadataPath
        PolicyText = $policyText
    }
    @(Assert-Q4MetadataContract @baselineContract)
}
$cuobjdumpCommand = (Get-Command $Cuobjdump -ErrorAction Stop).Source
$policySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $policyPath).
    Hash.ToLowerInvariant()
$metadataSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath).
    Hash.ToLowerInvariant()
$cuobjdumpVersionLines = & $cuobjdumpCommand --version
if ($LASTEXITCODE -ne 0) { throw "cuobjdump --version failed" }
$cuobjdumpVersion =
    [string]::Join([Environment]::NewLine, [string[]]$cuobjdumpVersionLines).
        Trim()
$preflightRecords = @()
foreach ($artifact in $artifacts) {
    $lines = & $cuobjdumpCommand -elf $artifact.Cubin
    if ($LASTEXITCODE -ne 0) { throw "cuobjdump failed for $($artifact.Shape)" }
    $inspection = [string]::Join([Environment]::NewLine, [string[]]$lines)
    Assert-Q4CubinInspection -Text $inspection -Symbol $artifact.Symbol -Registers $artifact.Registers -Threads $artifact.Threads
    $preflightRecords += [ordered]@{
        shape = $artifact.Shape
        shape_id = $artifact.ShapeId
        config_id = $artifact.ConfigId
        rationale = $artifact.Rationale
        symbol = $artifact.Symbol
        cubin = $artifact.Cubin
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact.Cubin).Hash.ToLowerInvariant()
        cubin_bytes = $artifact.CubinBytes
        sm = 86
        visible_arguments = Read-Q4PolicyInt $policyText "q4_visible_arguments"
        hidden_arguments = Read-Q4PolicyInt $policyText "q4_hidden_arguments"
        kparam_bytes = Read-Q4PolicyInt $policyText "q4_kparam_bytes"
        threads = $artifact.Threads
        block_m = $artifact.BlockM
        block_n = $artifact.BlockN
        registers_per_thread = $artifact.Registers
        local_memory_bytes = 0
        static_shared_bytes = 0
        dynamic_shared_bytes = $artifact.DynamicSharedBytes
    }
}

$output = Resolve-RepoPath $OutputDirectory
New-Item -ItemType Directory -Force -Path $output | Out-Null
$preflightPath = Join-Path $output "preflight.json"
$preflight = [ordered]@{
    schema = 1
    phase = "A2"
    candidate = "q4_0_x_q8_1_mmq_epilogue0"
    production_authority = $false
    metadata_kparam_preflight = $true
    selection_mode = $(if ($variantPairMode) { "bounded-variant-pair" } else { "baseline-run-b" })
    policy_sha256 = $policySha
    metadata = (Resolve-Path -LiteralPath $metadataPath).Path
    metadata_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath).Hash.ToLowerInvariant()
    artifacts = $preflightRecords
    passed = $true
}
[IO.File]::WriteAllText(
    $preflightPath,
    ($preflight | ConvertTo-Json -Depth 8) + [Environment]::NewLine
)
if ($Mode -eq "preflight") {
    Write-Host "LAB-A Q4 preflight: $preflightPath"
    exit 0
}

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
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "Visual Studio C++ Build Tools not found" }
    $vcvars = Join-Path $vs "VC/Auxiliary/Build/vcvars64.bat"
    $vcCommand = 'call "' + $vcvars + '" >nul && set'
    $lines = & cmd.exe /d /s /c $vcCommand
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
$nvccVersionLines = & $nvccCommand --version
if ($LASTEXITCODE -ne 0) { throw "nvcc --version failed" }
$nvccVersion =
    [string]::Join([Environment]::NewLine, [string[]]$nvccVersionLines).Trim()
$source = Join-Path $repo "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu"
$exe = Join-Path $output "kernel_lab_q4_q8_sm86.exe"
$zeroSha = "0" * 64
$quotedZeroSha = '"' + $zeroSha + '"'
$catalog = Get-Content (Join-Path $repo "crates/imparo-cuda/cuda-sm.json") -Raw |
    ConvertFrom-Json
if ($catalog.backend_abi -ne 26 -or $catalog.sms -notcontains 86) {
    throw "Q4 lab requires backend ABI26 with SM86 declared"
}
$nvccArgs = @(
    "-O3", "-std=c++17", "--use_fast_math", "-lineinfo",
    "-cudart=shared", "-Xcompiler", "/MD",
    "-DIMPARO_CUDA_KERNEL_LAB=1",
    "-DIMPARO_CUDA_BACKEND_ABI=$($catalog.backend_abi)",
    "-DIMPARO_CUDA_BUILD_SHA256=$quotedZeroSha",
    "--generate-code", "arch=compute_86,code=[sm_86,compute_86]",
    $source, "-o", $exe
)
& $nvccCommand @nvccArgs
if ($LASTEXITCODE -ne 0) {
    throw "NVCC Q4 lab compile failed with exit code $LASTEXITCODE"
}
$sourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).
    Hash.ToLowerInvariant()
$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).
    Hash.ToLowerInvariant()
$expansion = @($artifacts | Where-Object {
    $_.ShapeId -eq "k2560-m10240"
})[0]
$contraction = @($artifacts | Where-Object {
    $_.ShapeId -eq "k10240-m2560"
})[0]
$arguments = @(
    $expansion.Cubin, $expansion.Symbol,
    $contraction.Cubin, $contraction.Symbol
)
$policyWarmup = Read-Q4PolicyInt $policyText "warmup"
$policyPairs = Read-Q4PolicyInt $policyText "abba_baab_pairs"
$policyLaunches = Read-Q4PolicyInt $policyText "launches_per_sample"
$kernelGeometryArguments = @(
    $expansion.BlockM, $expansion.BlockN, $expansion.Threads,
    $expansion.DynamicSharedBytes,
    $contraction.BlockM, $contraction.BlockN, $contraction.Threads,
    $contraction.DynamicSharedBytes
)
$formalArguments = @($arguments) + @(
    $policyWarmup, $policyPairs, $policyLaunches
) + $kernelGeometryArguments
$smokeArguments = @($arguments) + @(1, 1, 1) + $kernelGeometryArguments

$sanitizerPassed = $false
$sanitizerLogs = [ordered]@{}
if ($Mode -in @("sanitizer", "all")) {
    $sanitizerPassed = $true
    $sanitizer = Resolve-ComputeSanitizer -NvccPath $nvccCommand
    foreach ($tool in @("memcheck", "initcheck", "racecheck", "synccheck")) {
        $log = Join-Path $output "compute-sanitizer-$tool.log"
        $toolOutput = & $sanitizer --tool $tool --error-exitcode 86 $exe @smokeArguments 2>&1
        $exitCode = $LASTEXITCODE
        [IO.File]::WriteAllLines($log, [string[]]$toolOutput)
        $sanitizerLogs[$tool] = [ordered]@{
            path = $log
            exit_code = $exitCode
            pass = $exitCode -eq 0
        }
        if ($exitCode -ne 0) { $sanitizerPassed = $false }
    }
    if ($Mode -eq "sanitizer") {
        if (-not $sanitizerPassed) {
            throw "one or more Q4 compute-sanitizer tools failed"
        }
        Write-Host "LAB-A Q4 sanitizer results: $output"
        exit 0
    }
}

$runArguments = if ($Mode -eq "smoke") { $smokeArguments } else { $formalArguments }
$runOutput = & $exe @runArguments
$runExit = $LASTEXITCODE
$resultPath = Join-Path $output "result.json"
if ($null -ne $runOutput) {
    [IO.File]::WriteAllText(
        $resultPath,
        [string]::Join([Environment]::NewLine, [string[]]$runOutput) +
            [Environment]::NewLine
    )
}
if ($runExit -notin @(0, 3)) {
    throw "LAB-A Q4 harness failed with exit code $runExit"
}
if ($runOutput.Count -ne 1) {
    throw "LAB-A Q4 harness must emit exactly one JSON line"
}
$data = [string]$runOutput | ConvertFrom-Json
$resultSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).
    Hash.ToLowerInvariant()
$violations = [Collections.Generic.List[string]]::new()
foreach ($violation in @(Get-Q4ResultViolations -Data $data -PolicyText $policyText)) {
    $violations.Add($violation)
}
$device = $data.device
if ($null -eq $device -or $device.sm -ne 86 -or $device.sm_count -ne 30 -or
    [string]::IsNullOrWhiteSpace([string]$device.uuid) -or
    [string]::IsNullOrWhiteSpace([string]$device.name) -or
    [string]::IsNullOrWhiteSpace([string]$device.driver_version) -or
    [string]::IsNullOrWhiteSpace([string]$device.cuda_runtime_version)) {
    $violations.Add("complete runtime device identity missing")
}
# Raw max_rel remains diagnostic because near-zero native outputs make it
# unstable. Formal numerical admission uses max_abs, RMS and
# abs(error)/max(abs(native),1.0).
$numericPolicyStatus =
    Read-Q4PolicyString $policyText "q4_numeric_policy_status"
if ($numericPolicyStatus -ne "frozen-from-cpu-oracle") {
    $violations.Add("Q4 numerical policy is provisional pending CPU oracle")
}
$correctnessViolationCount = $violations.Count
$measurementViolations = [Collections.Generic.List[string]]::new()

$timingPresent = $true
$timingEvidenceCases = 0
$resourceEvidenceCases = 0
$fastCases = 0
foreach ($case in @($data.cases)) {
    if ($case.n_tok -ne 512) {
        if ($null -ne $case.timing -or $null -ne $case.artifact_resources) {
            $measurementViolations.Add(
                "tail case $($case.shape_id)/$($case.n_tok) must not carry timing/resources"
            )
        }
        continue
    }
    ++$timingEvidenceCases
    $expectedArtifact = if ($case.shape_id -eq "k2560-m10240") {
        $expansion
    } elseif ($case.shape_id -eq "k10240-m2560") {
        $contraction
    } else {
        $null
    }
    $timing = $case.timing
    $timingOk = $null -ne $timing -and
        $timing.clock -eq "cuda_event" -and
        $timing.schedule -eq "ABBA_BAAB" -and
        $timing.warmup -eq $policyWarmup -and
        $timing.pairs -eq $policyPairs -and
        $timing.launches_per_sample -eq $policyLaunches -and
        $timing.samples_per_route -eq 2 * $policyPairs -and
        $timing.native_samples_us.Count -eq 2 * $policyPairs -and
        $timing.triton_samples_us.Count -eq 2 * $policyPairs -and
        $timing.native_first_launch_us -gt 0 -and
        $timing.triton_first_launch_us -gt 0 -and
        $timing.native_median_us -gt 0 -and
        $timing.triton_median_us -gt 0 -and
        $timing.native_cv -le
            (Read-Q4PolicyDouble $policyText "max_native_cv") -and
        $timing.triton_cv -le
            (Read-Q4PolicyDouble $policyText "max_candidate_cv") -and
        $timing.paired_mad_fraction -le
            (Read-Q4PolicyDouble $policyText "max_paired_mad_fraction")
    if ($timingOk) {
        foreach ($sample in @($timing.native_samples_us) +
            @($timing.triton_samples_us)) {
            if ($sample -le 0 -or [double]::IsNaN([double]$sample) -or
                [double]::IsInfinity([double]$sample)) {
                $timingOk = $false
            }
        }
    }
    if (-not $timingOk) {
        $timingPresent = $false
    }

    $resource = $case.artifact_resources
    $resourceOk = $null -ne $resource
    if ($resourceOk) {
        $rawCheckpoints = @(
            $resource.cuda_mem_free_before_buffers,
            $resource.cuda_mem_free_after_buffers,
            $resource.cuda_mem_free_before_module,
            $resource.cuda_mem_free_after_module,
            $resource.cuda_mem_free_after_first_launches,
            $resource.cuda_mem_free_after_timing
        )
        $resourceOk = $null -ne $expectedArtifact -and
            $resource.cubin_bytes -eq $expectedArtifact.CubinBytes -and
            $resource.module_load_wall_us -gt 0 -and
            $resource.cuda_mem_total_bytes -gt 0 -and
            ($rawCheckpoints | Where-Object {
                $_ -lt 0 -or $_ -gt $resource.cuda_mem_total_bytes
            }).Count -eq 0 -and
            $resource.cuda_mem_free_after_buffers -le
                $resource.cuda_mem_free_before_buffers -and
            $resource.cuda_mem_free_after_module -le
                $resource.cuda_mem_free_before_module -and
            $resource.cuda_mem_free_after_first_launches -le
                $resource.cuda_mem_free_after_module -and
            $resource.cuda_mem_free_after_timing -le
                $resource.cuda_mem_free_after_first_launches -and
            $resource.observed_buffer_delta_bytes -ge 0 -and
            $resource.observed_module_delta_bytes -ge 0 -and
            $resource.observed_first_launch_delta_bytes -ge 0 -and
            $resource.observed_timing_delta_bytes -ge 0 -and
            $resource.observed_peak_delta_bytes -ge
                $resource.observed_buffer_delta_bytes -and
            $resource.workspace_logical_bytes -eq
                (Read-Q4PolicyInt $policyText "q4_workspace_bytes") -and
            $resource.output_buffer_count -eq
                (Read-Q4PolicyInt $policyText "q4_output_buffer_count")
        $expectedWeights = [int64]$case.n_out * ($case.n_in / 32) * 18
        $expectedQ8 = [int64]($case.n_in / 128) * 512 * 144
        $expectedOutput = [int64]$case.n_out * 512 * 4
        $expectedGuarded = $expectedWeights + $expectedQ8 +
            2 * $expectedOutput +
            (Read-Q4PolicyInt $policyText "q4_workspace_bytes") + 2560
        $nativeResource = $resource.native_function
        $tritonResource = $resource.triton_function
        $resourceOk = $resourceOk -and
            $resource.weights_logical_bytes -eq $expectedWeights -and
            $resource.q8_logical_bytes -eq $expectedQ8 -and
            $resource.output_logical_bytes_per_buffer -eq $expectedOutput -and
            $resource.guarded_allocation_bytes -eq $expectedGuarded -and
            $resource.observed_buffer_delta_bytes -ge $expectedGuarded -and
            $nativeResource.registers_per_thread -gt 0 -and
            $nativeResource.registers_per_thread -le
                (Read-Q4PolicyInt $policyText "max_registers_per_thread") -and
            $nativeResource.static_shared_bytes -eq 0 -and
            $nativeResource.local_bytes -eq 0 -and
            $nativeResource.max_threads_per_block -ge
                (Read-Q4PolicyInt $policyText "q4_native_launch_threads") -and
            $nativeResource.binary_version -eq 86 -and
            $nativeResource.launch_threads -eq
                (Read-Q4PolicyInt $policyText "q4_native_launch_threads") -and
            $nativeResource.launch_dynamic_shared_bytes -eq
                (Read-Q4PolicyInt $policyText "q4_native_dynamic_shared_bytes") -and
            $tritonResource.registers_per_thread -eq
                $expectedArtifact.Registers -and
            $tritonResource.static_shared_bytes -eq
                (Read-Q4PolicyInt $policyText "q4_static_shared_bytes") -and
            $tritonResource.local_bytes -eq
                (Read-Q4PolicyInt $policyText "q4_local_memory_bytes") -and
            $tritonResource.max_threads_per_block -ge
                $expectedArtifact.Threads -and
            $tritonResource.binary_version -eq 86 -and
            $tritonResource.launch_threads -eq
                $expectedArtifact.Threads -and
            $tritonResource.block_m -eq
                $expectedArtifact.BlockM -and
            $tritonResource.block_n -eq
                $expectedArtifact.BlockN -and
            $tritonResource.launch_dynamic_shared_bytes -eq
                $expectedArtifact.DynamicSharedBytes
    }
    if ($resourceOk) { ++$resourceEvidenceCases }
    $caseCorrect = $case.comparison.non_finite -eq 0 -and
        $case.comparison.max_abs -le
            (Read-Q4PolicyDouble $policyText "q4_max_abs_vs_native") -and
        $case.comparison.rms -le
            (Read-Q4PolicyDouble $policyText "q4_max_rms_vs_native") -and
        $case.comparison.max_normalized_rel -le
            (Read-Q4PolicyDouble $policyText "q4_max_normalized_rel_vs_native") -and
        $case.native_input_mismatches -eq 0 -and
        $case.triton_input_mismatches -eq 0 -and
        $case.native_padding_errors -eq 0 -and
        $case.triton_padding_errors -eq 0 -and
        $case.canary_errors -eq 0 -and
        $null -ne $case.post_timing_comparison -and
        $case.post_timing_comparison.non_finite -eq 0 -and
        $case.post_timing_comparison.max_abs -le
            (Read-Q4PolicyDouble $policyText "q4_max_abs_vs_native") -and
        $case.post_timing_comparison.rms -le
            (Read-Q4PolicyDouble $policyText "q4_max_rms_vs_native") -and
        $case.post_timing_comparison.max_normalized_rel -le
            (Read-Q4PolicyDouble $policyText "q4_max_normalized_rel_vs_native") -and
        $case.post_timing_native_input_mismatches -eq 0 -and
        $case.post_timing_triton_input_mismatches -eq 0 -and
        $case.post_timing_native_padding_errors -eq 0 -and
        $case.post_timing_triton_padding_errors -eq 0 -and
        $case.post_timing_canary_errors -eq 0
    if ($caseCorrect -and $timingOk -and $resourceOk -and
        $timing.median_speedup_native_over_triton -ge
            (Read-Q4PolicyDouble $policyText "min_kernel_speedup_ratio")) {
        ++$fastCases
    }
}
$formalRequest = $data.measurement_request.warmup -eq $policyWarmup -and
    $data.measurement_request.pairs -eq $policyPairs -and
    $data.measurement_request.launches_per_sample -eq $policyLaunches -and
    $data.measurement_request.samples_per_route -eq 2 * $policyPairs -and
    $data.measurement_request.formal_contract
if ($timingEvidenceCases -ne 2 -or -not $formalRequest) {
    $timingPresent = $false
}
if (-not $timingPresent) {
    $measurementViolations.Add(
        "formal CUDA-event ABBA/BAAB timing/noise evidence missing"
    )
}
if ($resourceEvidenceCases -ne 2) {
    $measurementViolations.Add(
        "cold module/first-launch/VRAM/function-resource evidence missing"
    )
}
if ($fastCases -eq 0) {
    $measurementViolations.Add("Q4 kernel speedup floor not met")
}

$decisionPath = Join-Path $output "gate-a-q4-decision.json"
$correctnessAdmissible = $correctnessViolationCount -eq 0
$measurementAdmissible = $correctnessAdmissible -and
    $measurementViolations.Count -eq 0
foreach ($violation in $measurementViolations) { $violations.Add($violation) }
if (-not $sanitizerPassed) {
    $violations.Add("four-tool sanitizer evidence missing")
}
$candidateAdmissible = $measurementAdmissible -and $sanitizerPassed
$builderProvenance = [ordered]@{
    triton_source_commit =
        Read-Q4PolicyString $policyText "q4_builder_triton_commit"
    python = Read-Q4PolicyString $policyText "q4_builder_python"
    triton = Read-Q4PolicyString $policyText "q4_builder_triton"
    torch = Read-Q4PolicyString $policyText "q4_builder_torch"
    ptxas = Read-Q4PolicyString $policyText "q4_builder_ptxas"
    cuobjdump = Read-Q4PolicyString $policyText "q4_builder_cuobjdump"
    semantic_recipe_sha256 =
        Read-Q4PolicyString $policyText "q4_builder_recipe_sha256"
    image_id = Read-Q4PolicyString $policyText "q4_builder_image_id"
    linux_amd64_manifest =
        Read-Q4PolicyString $policyText "q4_builder_linux_amd64_manifest"
    oci_config = Read-Q4PolicyString $policyText "q4_builder_oci_config"
}
$decision = [ordered]@{
    schema = 1
    decision = "gate-a-candidate"
    candidate = "q4_0_x_q8_1_mmq_epilogue0"
    production_authority = $false
    final_gate_a_decision = $false
    candidate_admissible = $candidateAdmissible
    correctness_admissible = $correctnessAdmissible
    measurement_admissible = $measurementAdmissible
    selection_mode = $(if ($variantPairMode) {
        "bounded-variant-pair"
    } else {
        "baseline-run-b"
    })
    selected_pair = @($artifacts | ForEach-Object {
        [ordered]@{
            shape_id = $_.ShapeId
            config_id = $_.ConfigId
            rationale = $_.Rationale
            block_m = $_.BlockM
            block_n = $_.BlockN
            threads = $_.Threads
            dynamic_shared_bytes = $_.DynamicSharedBytes
            registers_per_thread = $_.Registers
        }
    })
    kernel_speed_floor_ratio =
        (Read-Q4PolicyDouble $policyText "min_kernel_speedup_ratio")
    projected_e2e = [ordered]@{
        required_improvement =
            (Read-Q4PolicyDouble $policyText "min_projected_e2e_improvement")
        evidence = $null
        admissible = $false
        gates_kernel_candidate = $false
    }
    policy = $policyPath
    provenance = [ordered]@{
        policy_sha256 = $policySha
        metadata_sha256 = $metadataSha
        variant_allowlist_rationale =
            (Read-Q4PolicyString $policyText "q4_variant_allowlist_rationale")
        cubins = @($preflightRecords | ForEach-Object {
            [ordered]@{
                shape = $_.shape
                sha256 = $_.sha256
                bytes = (Get-Item -LiteralPath $_.cubin).Length
            }
        })
        harness_source_sha256 = $sourceSha
        harness_exe_sha256 = $exeSha
        result_json_sha256 = $resultSha
        device = $device
        local_toolchain = [ordered]@{
            nvcc = $nvccVersion
            cuobjdump = $cuobjdumpVersion
            backend_abi = $catalog.backend_abi
            target_sm = 86
        }
        builder = $builderProvenance
    }
    numeric_policy_status = $numericPolicyStatus
    metadata_preflight = $preflightPath
    result = $resultPath
    sanitizer = $sanitizerLogs
    timing_evidence_present = $timingPresent
    timing_evidence_cases = $timingEvidenceCases
    resource_evidence_cases = $resourceEvidenceCases
    fast_cases = $fastCases
    violations = [string[]]$violations
}
[IO.File]::WriteAllText(
    $decisionPath,
    ($decision | ConvertTo-Json -Depth 8) + [Environment]::NewLine
)
Write-Host "LAB-A Q4 result: $resultPath"
Write-Host "LAB-A Q4 decision: $decisionPath"
if ($Mode -eq "smoke" -and $correctnessViolationCount -ne 0) {
    throw "Q4 C0 correctness/preflight policy failed; see $decisionPath"
}
if ($Mode -eq "measure" -and -not $measurementAdmissible) {
    throw "Gate A Q4 measurement is inadmissible; see $decisionPath"
}
if ($Mode -eq "all" -and -not $candidateAdmissible) {
    throw "Gate A Q4 candidate is inadmissible; see $decisionPath"
}
