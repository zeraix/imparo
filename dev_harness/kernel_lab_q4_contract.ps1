$ErrorActionPreference = "Stop"

function Read-Q4PolicyRaw([string]$Text, [string]$Name) {
    $matches = [regex]::Matches(
        $Text,
        "(?m)^\s*$([regex]::Escape($Name))\s*=\s*([^#\r\n]+?)\s*$"
    )
    if ($matches.Count -ne 1) {
        throw "policy key '$Name' must occur exactly once"
    }
    return $matches[0].Groups[1].Value.Trim()
}

function Read-Q4PolicyInt([string]$Text, [string]$Name) {
    return [int](Read-Q4PolicyRaw $Text $Name)
}

function Read-Q4PolicyDouble([string]$Text, [string]$Name) {
    return [double]::Parse(
        (Read-Q4PolicyRaw $Text $Name),
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Read-Q4PolicyString([string]$Text, [string]$Name) {
    $raw = Read-Q4PolicyRaw $Text $Name
    if ($raw -notmatch '^"(.*)"$') {
        throw "policy key '$Name' must be a quoted string"
    }
    return $Matches[1]
}

function Read-Q4PolicyIntArray([string]$Text, [string]$Name) {
    $raw = Read-Q4PolicyRaw $Text $Name
    if ($raw -notmatch '^\[(.*)\]$') {
        throw "policy key '$Name' must be an integer array"
    }
    if ([string]::IsNullOrWhiteSpace($Matches[1])) { return @() }
    return @($Matches[1].Split(',') | ForEach-Object { [int]$_.Trim() })
}

function Read-Q4PolicyStringArray([string]$Text, [string]$Name) {
    $raw = Read-Q4PolicyRaw $Text $Name
    if ($raw -notmatch '^\[(.*)\]$') {
        throw "policy key '$Name' must be a string array"
    }
    if ([string]::IsNullOrWhiteSpace($Matches[1])) { return @() }
    $values = @()
    foreach ($entry in $Matches[1].Split(',')) {
        $trimmed = $entry.Trim()
        if ($trimmed -notmatch '^"(.*)"$') {
            throw "policy key '$Name' contains a non-string entry"
        }
        $values += $Matches[1]
    }
    return [string[]]$values
}

function Assert-Q4CubinInspection {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Symbol,
        [Parameter(Mandatory = $true)][int]$Registers,
        [int]$Threads = 128
    )
    if ($Text -notmatch '(?m)^64bit elf:.*\bsm=86\b') {
        throw "cuobjdump target is not exact SM86"
    }
    if ($Text -notmatch [regex]::Escape(".nv.info.$Symbol")) {
        throw "cuobjdump does not contain required symbol $Symbol"
    }
    if ($Text -notmatch 'EIATTR_CBANK_PARAM_SIZE[\s\S]*?Value:\s+0x48\b') {
        throw "KPARAM constant-bank size is not 0x48"
    }
    $parameters = [regex]::Matches(
        $Text,
        'EIATTR_KPARAM_INFO[\s\S]*?Ordinal\s*:\s*0x([0-9a-f]+)' +
        '\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)',
        [Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if ($parameters.Count -ne 10) {
        throw "KPARAM must contain exactly 10 parameters, found $($parameters.Count)"
    }
    $expectedOffsets = @(0, 8, 16, 24, 32, 40, 44, 48, 56, 64)
    $expectedSizes = @(8, 8, 8, 8, 8, 4, 4, 4, 8, 8)
    $seen = @{}
    foreach ($parameter in $parameters) {
        $ordinal = [Convert]::ToInt32($parameter.Groups[1].Value, 16)
        $offset = [Convert]::ToInt32($parameter.Groups[2].Value, 16)
        $size = [Convert]::ToInt32($parameter.Groups[3].Value, 16)
        if ($ordinal -lt 0 -or $ordinal -ge 10 -or $seen.ContainsKey($ordinal)) {
            throw "KPARAM ordinal set is not exactly 0..9"
        }
        if ($offset -ne $expectedOffsets[$ordinal] -or
            $size -ne $expectedSizes[$ordinal]) {
            throw "KPARAM ordinal $ordinal has unexpected offset/size"
        }
        $seen[$ordinal] = $true
    }
    $threadsHex = "0x{0:x}" -f $Threads
    $threadPattern = 'EIATTR_REQNTID[\s\S]*?Value:\s+' +
        [regex]::Escape($threadsHex) + '\s+0x1\s+0x1'
    if ($Text -notmatch $threadPattern) {
        throw "REQNTID is not ${Threads}x1x1"
    }
    $registerPattern =
        'EIATTR_REGCOUNT[\s\S]*?register count:\s*' + [regex]::Escape("$Registers")
    if ($Text -notmatch $registerPattern) {
        throw "register count does not match metadata/policy"
    }
    if ($Text -notmatch 'EIATTR_FRAME_SIZE[\s\S]*?frame size:\s*0x0\b') {
        throw "local frame size is not zero"
    }
    $memoryPattern = '(?m)^\.text\.' + [regex]::Escape($Symbol) +
        '\r?\nbar\s*=\s*\d+\s+reg\s*=\s*' +
        [regex]::Escape("$Registers") + '\s+lmem\s*=\s*0\s+smem\s*=\s*0\s*$'
    if ($Text -notmatch $memoryPattern) {
        throw "cuobjdump reports non-zero local/static shared memory"
    }
}

function Assert-Q4MetadataContract {
    param(
        [Parameter(Mandatory = $true)][string]$MetadataPath,
        [Parameter(Mandatory = $true)][string]$PolicyText
    )
    $resolvedMetadata = (Resolve-Path -LiteralPath $MetadataPath).Path
    $metadataSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedMetadata).
        Hash.ToLowerInvariant()
    if ($metadataSha -ne (Read-Q4PolicyString $PolicyText "q4_metadata_sha256")) {
        throw "run-b metadata SHA256 mismatch"
    }
    $metadata = Get-Content -LiteralPath $resolvedMetadata -Raw | ConvertFrom-Json
    if ($metadata.schema -ne 1 -or $metadata.phase -ne "A2" -or
        $metadata.production_enabled -or $metadata.target -ne "cuda:86:32" -or
        $metadata.candidate -ne (Read-Q4PolicyString $PolicyText "q4_candidate")) {
        throw "run-b metadata identity mismatch"
    }
    if ($metadata.builder.image_id -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_image_id") -or
        $metadata.builder.python -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_python") -or
        $metadata.builder.triton -ne "3.8.0-source-pin" -or
        $metadata.builder.torch -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_torch")) {
        throw "run-b builder identity mismatch"
    }
    $names = @(
        "w_qs", "w_d", "x_qs", "x_d", "y", "n_tok", "out_stride",
        "numeric_stream_grid"
    )
    if ($metadata.launch_abi.Count -ne
        (Read-Q4PolicyInt $PolicyText "q4_visible_arguments")) {
        throw "visible launch ABI count mismatch"
    }
    for ($index = 0; $index -lt $names.Count; ++$index) {
        if ($metadata.launch_abi[$index].ordinal -ne $index -or
            $metadata.launch_abi[$index].name -ne $names[$index]) {
            throw "visible launch ABI ordinal/name mismatch at $index"
        }
    }
    if ($metadata.hidden_launch_abi.Count -ne
        (Read-Q4PolicyInt $PolicyText "q4_hidden_arguments") -or
        $metadata.hidden_launch_abi[0].ordinal -ne 8 -or
        $metadata.hidden_launch_abi[0].name -ne "global_scratch" -or
        -not $metadata.hidden_launch_abi[0].argument_required -or
        $metadata.hidden_launch_abi[0].allocation_bytes -ne 0 -or
        $metadata.hidden_launch_abi[1].ordinal -ne 9 -or
        $metadata.hidden_launch_abi[1].name -ne "profile_scratch" -or
        -not $metadata.hidden_launch_abi[1].argument_required -or
        $metadata.hidden_launch_abi[1].allocation_bytes -ne 0) {
        throw "hidden launch ABI mismatch"
    }
    $root = Split-Path -Parent $resolvedMetadata
    $contracts = @(
        @{
            Shape = Read-Q4PolicyString $PolicyText "q4_expansion_shape"
            Symbol = Read-Q4PolicyString $PolicyText "q4_expansion_symbol"
            Sha = Read-Q4PolicyString $PolicyText "q4_expansion_cubin_sha256"
        },
        @{
            Shape = Read-Q4PolicyString $PolicyText "q4_contraction_shape"
            Symbol = Read-Q4PolicyString $PolicyText "q4_contraction_symbol"
            Sha = Read-Q4PolicyString $PolicyText "q4_contraction_cubin_sha256"
        }
    )
    $validated = @()
    foreach ($contract in $contracts) {
        $matching = @($metadata.variants | Where-Object {
            $_.shape_id -eq $contract.Shape
        })
        if ($matching.Count -ne 1) {
            throw "metadata must contain one variant for $($contract.Shape)"
        }
        $variant = $matching[0]
        if ($variant.symbol -ne $contract.Symbol -or
            $variant.module_sha256 -ne $contract.Sha -or
            $variant.block.Count -ne 3 -or
            $variant.block[0] -ne (Read-Q4PolicyInt $PolicyText "q4_threads") -or
            $variant.block[1] -ne 1 -or $variant.block[2] -ne 1 -or
            $variant.dynamic_shared_bytes -ne
                (Read-Q4PolicyInt $PolicyText "q4_dynamic_shared_bytes") -or
            $variant.resources.registers_per_thread -ne
                (Read-Q4PolicyInt $PolicyText "q4_registers_per_thread") -or
            $variant.resources.local_memory_bytes -ne
                (Read-Q4PolicyInt $PolicyText "q4_local_memory_bytes") -or
            $variant.resources.global_scratch_bytes -ne 0 -or
            $variant.resources.profile_scratch_bytes -ne 0) {
            throw "metadata variant contract mismatch for $($contract.Shape)"
        }
        $module = [IO.Path]::GetFullPath((Join-Path $root $variant.module))
        $rootPrefix = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
        if (-not $module.StartsWith(
            $rootPrefix, [StringComparison]::OrdinalIgnoreCase
        ) -or -not (Test-Path -LiteralPath $module -PathType Leaf)) {
            throw "metadata module escapes run-b or is missing"
        }
        $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $module).
            Hash.ToLowerInvariant()
        if ($actualSha -ne $contract.Sha) {
            throw "cubin SHA256 mismatch for $($contract.Shape)"
        }
        if ((Get-Item -LiteralPath $module).Length -ne
            (Read-Q4PolicyInt $PolicyText "q4_cubin_bytes")) {
            throw "cubin byte-size mismatch for $($contract.Shape)"
        }
        $validated += [pscustomobject]@{
            Shape = $contract.Shape
            ShapeId = ([string]$variant.shape_id -replace '-n512$', '')
            ConfigId = "baseline-run-b"
            Rationale = "exact run-b baseline"
            Symbol = $contract.Symbol
            Cubin = $module
            CubinBytes = [int64](Get-Item -LiteralPath $module).Length
            Registers = [int]$variant.resources.registers_per_thread
            DynamicSharedBytes = [int]$variant.dynamic_shared_bytes
            Threads = Read-Q4PolicyInt $PolicyText "q4_triton_launch_threads"
            BlockM = Read-Q4PolicyInt $PolicyText "q4_triton_block_m"
            BlockN = Read-Q4PolicyInt $PolicyText "q4_triton_block_n"
        }
    }
    return $validated
}

function Assert-Q4VariantPairContract {
    param(
        [Parameter(Mandatory = $true)][string]$MetadataPath,
        [Parameter(Mandatory = $true)][string]$PolicyText,
        [Parameter(Mandatory = $true)][string]$ExpansionConfigId,
        [Parameter(Mandatory = $true)][string]$ContractionConfigId
    )
    $resolvedMetadata = (Resolve-Path -LiteralPath $MetadataPath).Path
    $metadataSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedMetadata).
        Hash.ToLowerInvariant()
    if ($metadataSha -ne
        (Read-Q4PolicyString $PolicyText "q4_variants_metadata_sha256")) {
        throw "variants-b metadata SHA256 mismatch"
    }
    $metadata = Get-Content -LiteralPath $resolvedMetadata -Raw |
        ConvertFrom-Json
    if ($metadata.schema -ne 1 -or $metadata.phase -ne "A2" -or
        $metadata.production_enabled -or $metadata.target -ne "cuda:86:32" -or
        $metadata.candidate -ne
            (Read-Q4PolicyString $PolicyText "q4_candidate") -or
        $metadata.milestone -ne "bounded-variant-resource-screen" -or
        -not $metadata.matrix.bounded -or
        -not $metadata.matrix.no_cartesian_expansion -or
        $metadata.matrix.requested_configs -ne 4 -or
        $metadata.matrix.requested_modules -ne 8 -or
        $metadata.matrix.surviving_modules -ne 8 -or
        $metadata.matrix.rejected_modules -ne 0 -or
        $metadata.survivors.Count -ne 8) {
        throw "variants-b bounded matrix identity mismatch"
    }
    if ($metadata.builder.image_id -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_image_id") -or
        $metadata.builder.python -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_python") -or
        $metadata.builder.triton -ne "3.8.0-source-pin" -or
        $metadata.builder.torch -ne
            (Read-Q4PolicyString $PolicyText "q4_builder_torch")) {
        throw "variants-b builder identity mismatch"
    }
    $names = @(
        "w_qs", "w_d", "x_qs", "x_d", "y", "n_tok", "out_stride",
        "numeric_stream_grid"
    )
    if ($metadata.launch_abi.Count -ne 8) {
        throw "variants-b visible ABI count mismatch"
    }
    for ($index = 0; $index -lt $names.Count; ++$index) {
        if ($metadata.launch_abi[$index].ordinal -ne $index -or
            $metadata.launch_abi[$index].name -ne $names[$index]) {
            throw "variants-b visible ABI mismatch at $index"
        }
    }
    if ($metadata.hidden_launch_abi.Count -ne 2 -or
        $metadata.hidden_launch_abi[0].ordinal -ne 8 -or
        $metadata.hidden_launch_abi[0].name -ne "global_scratch" -or
        -not $metadata.hidden_launch_abi[0].argument_required -or
        $metadata.hidden_launch_abi[0].allocation_bytes -ne 0 -or
        $metadata.hidden_launch_abi[1].ordinal -ne 9 -or
        $metadata.hidden_launch_abi[1].name -ne "profile_scratch" -or
        -not $metadata.hidden_launch_abi[1].argument_required -or
        $metadata.hidden_launch_abi[1].allocation_bytes -ne 0) {
        throw "variants-b hidden ABI mismatch"
    }
    $selections = @(
        @{
            Shape = "k2560-m10240"
            NIn = 2560
            NOut = 10240
            ConfigId = $ExpansionConfigId
            AllowedKey = "q4_variant_allowed_expansion"
        },
        @{
            Shape = "k10240-m2560"
            NIn = 10240
            NOut = 2560
            ConfigId = $ContractionConfigId
            AllowedKey = "q4_variant_allowed_contraction"
        }
    )
    $root = Split-Path -Parent $resolvedMetadata
    $validated = @()
    foreach ($selection in $selections) {
        $allowed = @(
            Read-Q4PolicyStringArray $PolicyText $selection.AllowedKey
        )
        if ($allowed.Count -ne 4 -or
            $allowed -notcontains $selection.ConfigId) {
            throw "variant config_id is not policy-allowed for $($selection.Shape)"
        }
        $matching = @($metadata.survivors | Where-Object {
            $_.shape_id -eq $selection.Shape -and
            $_.config_id -eq $selection.ConfigId
        })
        if ($matching.Count -ne 1) {
            throw "variants-b must contain exactly one selected survivor for $($selection.Shape)"
        }
        $variant = $matching[0]
        if ($variant.rejected -or $variant.n_in -ne $selection.NIn -or
            $variant.n_out -ne $selection.NOut -or $variant.n_tok -ne 512 -or
            $variant.block.Count -ne 3 -or
            $variant.block[0] -ne 32 * $variant.num_warps -or
            $variant.block[1] -ne 1 -or $variant.block[2] -ne 1 -or
            $variant.tile.rows -le 0 -or $variant.tile.tokens -le 0 -or
            128 % $variant.tile.rows -ne 0 -or
            128 % $variant.tile.tokens -ne 0 -or
            $variant.resources.registers_per_thread -le 0 -or
            $variant.resources.registers_per_thread -gt
                (Read-Q4PolicyInt $PolicyText "max_registers_per_thread") -or
            $variant.resources.dynamic_shared_bytes -lt 0 -or
            $variant.resources.dynamic_shared_bytes -gt
                (Read-Q4PolicyInt $PolicyText "max_dynamic_shared_bytes") -or
            $variant.resources.local_memory_bytes -ne 0 -or
            $variant.resources.global_scratch_bytes -ne 0 -or
            $variant.resources.profile_scratch_bytes -ne 0 -or
            [string]::IsNullOrWhiteSpace([string]$variant.symbol) -or
            [string]::IsNullOrWhiteSpace([string]$variant.rationale)) {
            throw "selected variant resource/geometry contract mismatch for $($selection.Shape)"
        }
        $module = [IO.Path]::GetFullPath((Join-Path $root $variant.module))
        $rootPrefix = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
        if (-not $module.StartsWith(
            $rootPrefix, [StringComparison]::OrdinalIgnoreCase
        ) -or -not (Test-Path -LiteralPath $module -PathType Leaf)) {
            throw "selected variant module escapes variants-b or is missing"
        }
        $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $module).
            Hash.ToLowerInvariant()
        $actualBytes = (Get-Item -LiteralPath $module).Length
        if ($actualSha -ne $variant.module_sha256 -or
            $actualBytes -ne $variant.module_bytes) {
            throw "selected variant module hash/size mismatch for $($selection.Shape)"
        }
        $validated += [pscustomobject]@{
            Shape = "$($selection.Shape)-n512"
            ShapeId = $selection.Shape
            ConfigId = [string]$variant.config_id
            Rationale = [string]$variant.rationale
            Symbol = [string]$variant.symbol
            Cubin = $module
            CubinBytes = [int64]$actualBytes
            Registers = [int]$variant.resources.registers_per_thread
            DynamicSharedBytes = [int]$variant.resources.dynamic_shared_bytes
            Threads = [int]$variant.block[0]
            BlockM = [int]$variant.tile.rows
            BlockN = [int]$variant.tile.tokens
        }
    }
    return $validated
}

function Get-Q4ResultViolations {
    param(
        [Parameter(Mandatory = $true)]$Data,
        [Parameter(Mandatory = $true)][string]$PolicyText
    )
    $violations = [Collections.Generic.List[string]]::new()
    function Check($Condition, [string]$Message) {
        # Missing JSON properties arrive as $null. Treat them as false so a
        # malformed result becomes a named violation instead of aborting the
        # validator during Boolean parameter binding.
        if (-not [bool]$Condition) { $violations.Add($Message) }
    }
    Check ($Data.schema -eq 1) "result schema mismatch"
    Check ($Data.phase -eq "A2") "result phase mismatch"
    Check (-not $Data.production_enabled) "production must remain disabled"
    Check ($Data.target_sm -eq 86) "result target is not SM86"
    Check ($Data.same_primary_context -and $Data.same_stream) "context/stream mismatch"
    Check ($Data.visible_arguments -eq
        (Read-Q4PolicyInt $PolicyText "q4_visible_arguments")) "visible ABI mismatch"
    Check ($Data.hidden_arguments -eq
        (Read-Q4PolicyInt $PolicyText "q4_hidden_arguments")) "hidden ABI mismatch"
    Check (-not $Data.policy_admissible) "harness must not claim policy authority"
    Check (-not $Data.formal_pass) "C0 harness must not claim formal pass"
    Check (-not $Data.metadata_kparam_preflight) "harness must not claim wrapper preflight"
    Check ($Data.debug_exact_triton_native_required -eq $false) "bitwise comparison must be diagnostic-only"
    Check ($Data.PSObject.Properties.Name -contains
        "debug_exact_triton_native_pass") "bitwise diagnostic result missing"
    $required = @(Read-Q4PolicyIntArray $PolicyText "q4_require_tail_tokens")
    $actual = @($Data.tail_tokens | ForEach-Object { [int]$_ })
    Check (($actual -join ',') -eq ($required -join ',')) "tail-token vector mismatch"
    Check ($Data.structural_ok) "harness structural checks failed"
    Check ($Data.cases.Count -eq 2 * $required.Count) "case count mismatch"
    $seen = @{}
    foreach ($case in @($Data.cases)) {
        $shape = [string]$case.shape_id
        $isExpand = $shape -eq "k2560-m10240"
        $isContract = $shape -eq "k10240-m2560"
        Check ($isExpand -or $isContract) "unknown result shape $shape"
        $expectedIn = if ($isExpand) { 2560 } else { 10240 }
        $expectedOut = if ($isExpand) { 10240 } else { 2560 }
        $token = [int]$case.n_tok
        $key = "$shape/$token"
        Check (-not $seen.ContainsKey($key)) "duplicate result case $key"
        $seen[$key] = $true
        Check ($required -contains $token) "unexpected token count in $key"
        Check ($case.n_in -eq $expectedIn -and $case.n_out -eq $expectedOut) "shape mismatch in $key"
        $padding = Read-Q4PolicyInt $PolicyText "q4_output_stride_padding"
        $expectedStride = $expectedOut + $(if ($token -eq 512) { 0 } else { $padding })
        Check ($case.out_stride -eq $expectedStride) "output stride mismatch in $key"
        $logical = [int]([math]::Ceiling($expectedOut / 128.0) *
            [math]::Ceiling($token / 128.0))
        $waves = [math]::Ceiling($logical / 30.0)
        $efficiency = [math]::Floor(100 * $logical / (30 * $waves))
        $expectedRoute = if ($token -eq 512) { "full-tile" } else { "physical-stream-k" }
        $expectedPhysical = if ($token -eq 512) { $logical } elseif (
            $efficiency -ge 90
        ) { $logical } else { 30 }
        $expectedEfficiency = if ($token -eq 512) {
            if ($isExpand) { 96 } else { 88 }
        } else { $efficiency }
        $expectedGrid = if ($isContract -and $token -eq 512) { 30 } else {
            $expectedPhysical
        }
        Check ($case.native_route -eq $expectedRoute) "native route mismatch in $key"
        Check ($case.native_tiles.rows -eq 128 -and
            $case.native_tiles.tokens -eq 128) "native tile mismatch in $key"
        Check ($case.logical_tiles -eq $logical) "logical tile mismatch in $key"
        Check ($case.physical_tiles -eq $expectedPhysical) "physical tile mismatch in $key"
        Check ($case.efficiency -eq $expectedEfficiency) "efficiency mismatch in $key"
        $expectedSeams = if ($expectedRoute -eq "full-tile") {
            $expectedEfficiency -lt 90
        } else {
            $expectedPhysical -gt 0 -and ($logical % $expectedPhysical) -ne 0
        }
        Check ($case.numeric_seams -eq $expectedSeams) "numeric seam flag mismatch in $key"
        Check (-not $case.fused_epilogue) "unexpected fused epilogue in $key"
        Check ($case.numeric_stream_grid -eq $expectedGrid) "numeric grid mismatch in $key"
        Check ($case.workspace_bytes -eq
            (Read-Q4PolicyInt $PolicyText "q4_workspace_bytes")) "workspace mismatch in $key"
        Check ($case.comparison.finite -eq $token * $expectedOut) "finite count mismatch in $key"
        Check ($case.comparison.non_finite -le
            (Read-Q4PolicyInt $PolicyText "q4_max_non_finite")) "non-finite output in $key"
        Check ($case.comparison.max_abs -le
            (Read-Q4PolicyDouble $PolicyText "q4_max_abs_vs_native")) "max_abs mismatch in $key"
        Check ($case.comparison.rms -le
            (Read-Q4PolicyDouble $PolicyText "q4_max_rms_vs_native")) "RMS mismatch in $key"
        Check ($case.comparison.max_normalized_rel -le
            (Read-Q4PolicyDouble $PolicyText "q4_max_normalized_rel_vs_native")) "normalized relative mismatch in $key"
        $inputLimit = Read-Q4PolicyInt $PolicyText "q4_max_input_mismatches"
        Check ($case.native_input_mismatches -le $inputLimit) "native input mutation in $key"
        Check ($case.triton_input_mismatches -le $inputLimit) "Triton input mutation in $key"
        $paddingLimit = Read-Q4PolicyInt $PolicyText "q4_max_padding_errors"
        Check ($case.native_padding_errors -le $paddingLimit) "native padding error in $key"
        Check ($case.triton_padding_errors -le $paddingLimit) "Triton padding error in $key"
        Check ($case.canary_errors -le
            (Read-Q4PolicyInt $PolicyText "q4_max_canary_errors")) "canary error in $key"
        foreach ($postField in @(
            "post_timing_comparison",
            "post_timing_native_input_mismatches",
            "post_timing_triton_input_mismatches",
            "post_timing_native_padding_errors",
            "post_timing_triton_padding_errors",
            "post_timing_canary_errors"
        )) {
            Check ($case.PSObject.Properties.Name -contains $postField) "post-timing field $postField missing in $key"
        }
        if ($token -eq 512) {
            Check ($null -ne $case.post_timing_comparison) "post-timing comparison missing in $key"
            Check ($case.post_timing_comparison.finite -eq
                $token * $expectedOut) "post-timing finite count mismatch in $key"
            Check ($case.post_timing_comparison.non_finite -le
                (Read-Q4PolicyInt $PolicyText "q4_max_non_finite")) "post-timing non-finite output in $key"
            Check ($case.post_timing_comparison.max_abs -le
                (Read-Q4PolicyDouble $PolicyText "q4_max_abs_vs_native")) "post-timing max_abs mismatch in $key"
            Check ($case.post_timing_comparison.rms -le
                (Read-Q4PolicyDouble $PolicyText "q4_max_rms_vs_native")) "post-timing RMS mismatch in $key"
            Check ($case.post_timing_comparison.max_normalized_rel -le
                (Read-Q4PolicyDouble $PolicyText "q4_max_normalized_rel_vs_native")) "post-timing normalized relative mismatch in $key"
        } else {
            Check ($null -eq $case.post_timing_comparison) "tail case carries post-timing comparison in $key"
        }
        Check ($case.post_timing_native_input_mismatches -eq $inputLimit) "post-timing native input mutation in $key"
        Check ($case.post_timing_triton_input_mismatches -eq $inputLimit) "post-timing Triton input mutation in $key"
        Check ($case.post_timing_native_padding_errors -eq $paddingLimit) "post-timing native padding error in $key"
        Check ($case.post_timing_triton_padding_errors -eq $paddingLimit) "post-timing Triton padding error in $key"
        Check ($case.post_timing_canary_errors -eq
            (Read-Q4PolicyInt $PolicyText "q4_max_canary_errors")) "post-timing canary error in $key"
        if ($token -eq 512) {
            $oracle = $case.cpu_oracle
            $expectedSamples = if ($isExpand) { 4 } else { 8 }
            Check ($null -ne $oracle -and $oracle.samples -eq $expectedSamples) "CPU oracle sample mismatch in $key"
            Check ($oracle.native_vs_strict_float.max_abs -le
                (Read-Q4PolicyDouble $PolicyText "q4_oracle_strict_native_max_abs")) "native strict oracle mismatch in $key"
            Check ($oracle.native_vs_strict_float.bitwise_different -eq 0) "native strict oracle bit mismatch in $key"
            Check ($oracle.triton_vs_strict_float.max_abs -le
                (Read-Q4PolicyDouble $PolicyText "q4_oracle_strict_triton_max_abs")) "Triton strict oracle mismatch in $key"
            $f64Limit = Read-Q4PolicyDouble $PolicyText "q4_oracle_f64_max_abs"
            Check ($oracle.native_vs_f64.max_abs -le $f64Limit) "native f64 oracle mismatch in $key"
            Check ($oracle.triton_vs_f64.max_abs -le $f64Limit) "Triton f64 oracle mismatch in $key"
            Check ($oracle.native_vs_strict_float.non_finite -eq 0 -and
                $oracle.triton_vs_strict_float.non_finite -eq 0 -and
                $oracle.native_vs_f64.non_finite -eq 0 -and
                $oracle.triton_vs_f64.non_finite -eq 0) "non-finite CPU oracle value in $key"
            if ($isExpand) {
                Check ($oracle.seam_104_samples -eq 0 -and
                    $oracle.seam_208_samples -eq 0 -and
                    $oracle.no_seam_samples -eq 4) "expansion oracle seam counts mismatch"
                Check (-not $oracle.wrong_grid_gpu_launched) "expansion must not launch wrong-grid mutation"
            } else {
                $mutationMinimum =
                    Read-Q4PolicyInt $PolicyText "q4_mutation_min_bitwise_different"
                Check ($oracle.seam_104_samples -eq 2 -and
                    $oracle.seam_208_samples -eq 2 -and
                    $oracle.no_seam_samples -eq 4) "contraction oracle seam counts mismatch"
                Check ($oracle.wrong_grid_bitwise_different -ge 1) "wrong-grid mutation was not detected"
                Check ($oracle.seam_minus_one_bitwise_different -ge
                    $mutationMinimum) "seam-1 CPU mutation was not detected"
                Check ($oracle.seam_plus_one_bitwise_different -ge
                    $mutationMinimum) "seam+1 CPU mutation was not detected"
                Check (($oracle.directed_seam_tiles -join ',') -eq "2,5") "directed seam tile vector mismatch"
                Check (($oracle.adjacent_no_seam_tiles -join ',') -eq "1,3,4,6") "adjacent no-seam vector mismatch"
                Check (($oracle.directed_block_boundaries -join ',') -eq "103,104,207,208") "directed block boundary vector mismatch"
                Check ($oracle.wrong_grid_gpu_launched) "wrong-grid GPU mutation was not launched"
                Check ($oracle.wrong_grid_gpu_vs_correct.bitwise_different -ge
                    $mutationMinimum -and
                    $oracle.wrong_grid_gpu_vs_correct.max_abs -gt 0) "wrong-grid GPU mutation did not differ from correct Triton"
                Check ($oracle.wrong_grid_gpu_vs_native.bitwise_different -ge
                    $mutationMinimum -and
                    $oracle.wrong_grid_gpu_vs_native.max_abs -gt 0) "wrong-grid GPU mutation did not differ from native"
                Check ($oracle.wrong_grid_gpu_vs_correct.non_finite -eq 0 -and
                    $oracle.wrong_grid_gpu_vs_native.non_finite -eq 0) "wrong-grid GPU mutation produced non-finite output"
                Check ($oracle.wrong_grid_gpu_canary_errors -eq 0) "wrong-grid GPU mutation corrupted canary"
                Check ($oracle.wrong_grid_gpu_input_mismatches -eq 0) "wrong-grid GPU mutation changed input"
                Check ($oracle.wrong_grid_gpu_padding_errors -eq 0) "wrong-grid GPU mutation corrupted padding"
            }
        } else {
            Check ($null -eq $case.cpu_oracle) "CPU oracle must be restricted to evidence n_tok=512"
        }
    }
    foreach ($shape in @("k2560-m10240", "k10240-m2560")) {
        foreach ($token in $required) {
            Check ($seen.ContainsKey("$shape/$token")) "missing result case $shape/$token"
        }
    }
    return [string[]]$violations
}
