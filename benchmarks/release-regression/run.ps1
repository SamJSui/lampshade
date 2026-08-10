param(
    [long[]]$Items = @(1000000, 10000000, 100000000),
    [ValidateSet('reduce_sum', 'sort_bounded16', 'sort_full_width', 'exclusive_scan', 'compact_50')]
    [string[]]$Workloads = @('reduce_sum', 'sort_bounded16', 'sort_full_width', 'exclusive_scan', 'compact_50'),
    [ValidateRange(1, 20)]
    [int]$Processes = 3,
    [string]$Backend,
    [ValidateRange(0, 100)]
    [double]$ThresholdPercent = 2.0,
    [string]$OutputPath,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Backend) {
    $Backend = if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::OSX
    )) { 'metal' } else { 'vulkan' }
}

$benchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $benchmarkRoot '..\..')).Path
$targetRoot = Join-Path $repoRoot 'target\release-regression'
$safeRepoRoot = $repoRoot.Replace('\', '/')
$baselineVersion = '0.7.0'
if (-not $OutputPath) {
    $OutputPath = Join-Path $benchmarkRoot 'results\latest.json'
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}
if ($Quick) {
    $Items = @(1000000)
    $Processes = 1
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'Cannot compute the median of an empty collection.' }
    $middle = [math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 0) { return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0 }
    return $sorted[$middle]
}

function Get-Sha256Hex {
    param([byte[]]$Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-SamplingConfig {
    param([long]$ItemCount)
    if ($Quick) { return @{ Warmups = 1; WarmupMs = 0; Samples = 3 } }
    if ($ItemCount -ge 100000000) { return @{ Warmups = 2; WarmupMs = 2000; Samples = 7 } }
    return @{ Warmups = 4; WarmupMs = 2000; Samples = 11 }
}

function Build-Runner {
    param([string]$Name, [string]$Manifest, [string]$TargetDirectory)
    Write-Host "Building $Name..."
    $previousTarget = $env:CARGO_TARGET_DIR
    try {
        $env:CARGO_TARGET_DIR = $TargetDirectory
        & cargo build --release --locked --manifest-path $Manifest
        if ($LASTEXITCODE -ne 0) { throw "Failed to build $Name." }
    }
    finally { $env:CARGO_TARGET_DIR = $previousTarget }
}

function Invoke-Runner {
    param(
        [string]$Executable, [string]$Version, [string]$Revision,
        [long]$ItemCount, [string]$Workload, [int]$Warmups,
        [long]$WarmupMs, [int]$Samples, [int]$ProcessIndex
    )
    $names = @(
        'WGPU_BACKEND', 'MASSIVELY_BENCH_ITEMS', 'MASSIVELY_BENCH_WORKLOAD',
        'MASSIVELY_BENCH_WARMUPS', 'MASSIVELY_BENCH_WARMUP_MS',
        'MASSIVELY_BENCH_SAMPLES', 'MASSIVELY_BENCH_PROCESS_INDEX',
        'MASSIVELY_BENCH_IMPLEMENTATION_VERSION', 'MASSIVELY_BENCH_IMPLEMENTATION_REVISION'
    )
    $previous = @{}
    foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:WGPU_BACKEND = $Backend
        $env:MASSIVELY_BENCH_ITEMS = [string]$ItemCount
        $env:MASSIVELY_BENCH_WORKLOAD = $Workload
        $env:MASSIVELY_BENCH_WARMUPS = [string]$Warmups
        $env:MASSIVELY_BENCH_WARMUP_MS = [string]$WarmupMs
        $env:MASSIVELY_BENCH_SAMPLES = [string]$Samples
        $env:MASSIVELY_BENCH_PROCESS_INDEX = [string]$ProcessIndex
        $env:MASSIVELY_BENCH_IMPLEMENTATION_VERSION = $Version
        $env:MASSIVELY_BENCH_IMPLEMENTATION_REVISION = $Revision
        $output = @(& $Executable 2>&1)
        if ($LASTEXITCODE -ne 0) { throw (($output | ForEach-Object ToString) -join [Environment]::NewLine) }
        $json = $output | ForEach-Object ToString | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
        if (-not $json) { throw 'Runner completed without emitting a JSON result.' }
        return $json | ConvertFrom-Json
    }
    finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
    }
}

$candidateManifest = Join-Path $repoRoot 'benchmarks\massively-comparison\wgpu-primitives-runner\Cargo.toml'
$baselineManifest = Join-Path $benchmarkRoot 'published-runner\Cargo.toml'
$candidateTarget = Join-Path $targetRoot 'checkout'
$baselineTarget = Join-Path $targetRoot 'published'
Build-Runner 'checkout runner' $candidateManifest $candidateTarget
Build-Runner "crates.io $baselineVersion runner" $baselineManifest $baselineTarget

$suffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$candidateExecutable = Join-Path $candidateTarget "release\wgpu-primitives-massively-comparison-runner$suffix"
$baselineExecutable = Join-Path $baselineTarget "release\wgpu-primitives-release-baseline-runner$suffix"
$revision = (& git -c "safe.directory=$safeRepoRoot" -C $repoRoot rev-parse HEAD).Trim()
$dirty = @(& git -c "safe.directory=$safeRepoRoot" -C $repoRoot status --porcelain).Count -gt 0
$metadata = cargo metadata --no-deps --format-version 1 --manifest-path (Join-Path $repoRoot 'Cargo.toml') | ConvertFrom-Json
$candidateVersion = $metadata.packages[0].version
$sourceRoots = @(
    'Cargo.toml', 'src', 'benchmarks/massively-comparison/common',
    'benchmarks/massively-comparison/wgpu-primitives-runner', 'benchmarks/release-regression'
)
[string[]]$sourcePaths = @(& git -c "safe.directory=$safeRepoRoot" -C $repoRoot ls-files --cached --others --exclude-standard -- @sourceRoots)
[Array]::Sort($sourcePaths, [StringComparer]::Ordinal)
$utf8 = [System.Text.UTF8Encoding]::new($false)
$sourceFiles = [System.Collections.Generic.List[object]]::new()
$manifestLines = [System.Collections.Generic.List[string]]::new()
foreach ($sourcePath in $sourcePaths) {
    $content = [System.IO.File]::ReadAllText((Join-Path $repoRoot $sourcePath)).Replace("`r`n", "`n").Replace("`r", "`n")
    $hash = Get-Sha256Hex $utf8.GetBytes($content)
    $sourceFiles.Add([pscustomobject][ordered]@{ path = $sourcePath; sha256 = $hash })
    $manifestLines.Add("$hash  $sourcePath`n")
}
$sourceManifest = Get-Sha256Hex $utf8.GetBytes(($manifestLines -join ''))
$runs = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[object]]::new()

foreach ($itemCount in $Items) {
    if ($itemCount -lt 1 -or $itemCount -gt [uint32]::MaxValue) { throw "Item count out of range: $itemCount" }
    $sampling = Get-SamplingConfig $itemCount
    foreach ($workload in $Workloads) {
        for ($processIndex = 1; $processIndex -le $Processes; $processIndex++) {
            $sources = @(
                @{ Source = 'published'; Executable = $baselineExecutable; Version = "crates.io-$baselineVersion"; Revision = "v$baselineVersion" },
                @{ Source = 'checkout'; Executable = $candidateExecutable; Version = "working-tree-$candidateVersion"; Revision = $revision }
            )
            if ($processIndex % 2 -eq 0) { [array]::Reverse($sources) }
            foreach ($source in $sources) {
                Write-Host "$($source.Source) $workload items=$itemCount process=$processIndex"
                try {
                    $result = Invoke-Runner $source.Executable $source.Version $source.Revision $itemCount $workload $sampling.Warmups $sampling.WarmupMs $sampling.Samples $processIndex
                    $runs.Add([pscustomobject][ordered]@{ source = $source.Source; result = $result })
                }
                catch {
                    $failures.Add([pscustomobject][ordered]@{ source = $source.Source; workload = $workload; items = $itemCount; process_index = $processIndex; error = $_.Exception.Message })
                }
            }
        }
    }
}

$aggregates = [System.Collections.Generic.List[object]]::new()
$groups = $runs | Group-Object -Property { "$($_.source)|$($_.result.config.workload)|$($_.result.config.items)" }
foreach ($group in $groups) {
    $first = $group.Group[0]
    $medians = @($group.Group | ForEach-Object { [double]$_.result.median_ms })
    $adapterKeys = @($group.Group | ForEach-Object {
        $adapter = $_.result.adapter
        "$($adapter.name)|$($adapter.vendor)|$($adapter.device)|$($adapter.device_type)|$($adapter.backend)"
    } | Select-Object -Unique)
    $aggregates.Add([pscustomobject][ordered]@{
        source = $first.source; workload = $first.result.config.workload; items = [long]$first.result.config.items
        process_medians_ms = $medians; median_of_process_medians_ms = Get-Median $medians
        adapter = [pscustomobject][ordered]@{
            name = $first.result.adapter.name; vendor = $first.result.adapter.vendor
            device = $first.result.adapter.device; device_type = $first.result.adapter.device_type
            backend = $first.result.adapter.backend
        }
        adapter_key = $adapterKeys[0]; adapter_consistent = $adapterKeys.Count -eq 1
    })
}

$comparisons = [System.Collections.Generic.List[object]]::new()
$cases = $aggregates | Group-Object -Property { "$($_.workload)|$($_.items)" }
foreach ($case in $cases) {
    $published = $case.Group | Where-Object source -eq 'published' | Select-Object -First 1
    $checkout = $case.Group | Where-Object source -eq 'checkout' | Select-Object -First 1
    if ($null -eq $published -or $null -eq $checkout) { continue }
    $change = ([double]$checkout.median_of_process_medians_ms / [double]$published.median_of_process_medians_ms - 1.0) * 100.0
    $adapterMatch = $published.adapter_consistent -and $checkout.adapter_consistent -and $published.adapter_key -eq $checkout.adapter_key
    $comparisons.Add([pscustomobject][ordered]@{
        workload = $published.workload; items = $published.items
        published_ms = $published.median_of_process_medians_ms; checkout_ms = $checkout.median_of_process_medians_ms
        change_percent = $change; adapter_match = $adapterMatch
        passed = $adapterMatch -and $change -le ($ThresholdPercent + 1e-9)
    })
}

$expected = $Items.Count * $Workloads.Count
$adaptersPassed = @($comparisons | Where-Object { -not $_.adapter_match }).Count -eq 0
$regressionPassed = [bool]$Quick -or @($comparisons | Where-Object { -not $_.passed }).Count -eq 0
$gatePassed = $failures.Count -eq 0 -and $comparisons.Count -eq $expected -and $adaptersPassed -and $regressionPassed
$artifact = [pscustomobject][ordered]@{
    schema_version = 1; generated_at_utc = [DateTime]::UtcNow.ToString('o')
    host = @{ hostname = [Environment]::MachineName; architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() }
    baseline = @{ source = 'crates.io'; version = $baselineVersion }
    candidate = @{
        version = $candidateVersion; revision = $revision; dirty = $dirty
        source_manifest_algorithm = "SHA-256 of ordered '<LF-normalized file SHA-256>  <path>\n' entries"
        source_manifest_sha256 = $sourceManifest; source_files = $sourceFiles
    }
    config = @{ backend = $Backend; items = $Items; workloads = $Workloads; processes = $Processes; threshold_percent = $ThresholdPercent; quick = [bool]$Quick }
    methodology = @{ timing = 'identical public resident API and completion boundary per source'; aggregation = 'median of independent process medians'; gate = 'candidate increase must not exceed threshold_percent' }
    runs = $runs; failures = $failures; aggregates = $aggregates; comparisons = $comparisons
    gate_evaluated = -not [bool]$Quick; gate_passed = $gatePassed
}
$directory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$artifact | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8
$displayComparisons = $comparisons | Select-Object workload, items, published_ms, checkout_ms, change_percent, @{
    Name = 'gate'
    Expression = { if ($Quick) { 'n/a' } elseif ($_.passed) { 'pass' } else { 'FAIL' } }
}
$displayComparisons | Format-Table -AutoSize
Write-Host "Machine-readable results: $OutputPath"
if (-not $gatePassed) { throw 'Release regression gate failed.' }
