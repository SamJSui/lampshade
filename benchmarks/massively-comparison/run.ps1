param(
    [long[]]$Items = @(1000000, 10000000, 100000000),
    [ValidateSet('reduce_sum', 'sort_bounded16', 'sort_full_width', 'exclusive_scan', 'compact_50')]
    [string[]]$Workloads = @('reduce_sum', 'sort_bounded16', 'sort_full_width', 'exclusive_scan', 'compact_50'),
    [ValidateRange(1, 20)]
    [int]$Processes = 3,
    [string]$Backend = 'vulkan',
    [string]$OutputPath,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$benchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $benchmarkRoot '..\..')).Path
$safeRepoRoot = $repoRoot.Replace('\', '/')
$targetRoot = Join-Path $repoRoot 'target\massively-comparison'

if (-not $OutputPath) {
    $OutputPath = Join-Path $benchmarkRoot 'results\latest.json'
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}
if ($Quick) {
    $Items = @(1000000)
    $Workloads = @('reduce_sum', 'sort_bounded16', 'sort_full_width', 'exclusive_scan', 'compact_50')
    $Processes = 1
}

function Get-Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) {
        throw 'Cannot compute the median of an empty collection.'
    }
    $middle = [math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 0) {
        return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
    }
    return $sorted[$middle]
}

function Get-SamplingConfig {
    param([long]$ItemCount)
    if ($Quick) {
        return @{ Warmups = 1; WarmupMs = 0; Samples = 3 }
    }
    if ($ItemCount -ge 100000000) {
        return @{ Warmups = 2; WarmupMs = 2000; Samples = 7 }
    }
    return @{ Warmups = 4; WarmupMs = 2000; Samples = 11 }
}

function Build-Runner {
    param([string]$Name, [string]$Manifest, [string]$TargetDirectory)
    Write-Host "Building $Name..."
    $previousTarget = $env:CARGO_TARGET_DIR
    try {
        $env:CARGO_TARGET_DIR = $TargetDirectory
        & cargo build --release --locked --manifest-path $Manifest
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build $Name."
        }
    }
    finally {
        $env:CARGO_TARGET_DIR = $previousTarget
    }
}

function Invoke-Runner {
    param(
        [string]$Executable,
        [string]$Implementation,
        [string]$ImplementationVersion,
        [string]$ImplementationRevision,
        [long]$ItemCount,
        [string]$Workload,
        [int]$Warmups,
        [long]$WarmupMs,
        [int]$Samples,
        [int]$ProcessIndex
    )
    $variables = @(
        'WGPU_BACKEND',
        'MASSIVELY_BENCH_ITEMS',
        'MASSIVELY_BENCH_WORKLOAD',
        'MASSIVELY_BENCH_WARMUPS',
        'MASSIVELY_BENCH_WARMUP_MS',
        'MASSIVELY_BENCH_SAMPLES',
        'MASSIVELY_BENCH_PROCESS_INDEX',
        'MASSIVELY_BENCH_IMPLEMENTATION_VERSION',
        'MASSIVELY_BENCH_IMPLEMENTATION_REVISION'
    )
    $previous = @{}
    foreach ($variable in $variables) {
        $previous[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
    }

    try {
        $env:WGPU_BACKEND = $Backend
        $env:MASSIVELY_BENCH_ITEMS = [string]$ItemCount
        $env:MASSIVELY_BENCH_WORKLOAD = $Workload
        $env:MASSIVELY_BENCH_WARMUPS = [string]$Warmups
        $env:MASSIVELY_BENCH_WARMUP_MS = [string]$WarmupMs
        $env:MASSIVELY_BENCH_SAMPLES = [string]$Samples
        $env:MASSIVELY_BENCH_PROCESS_INDEX = [string]$ProcessIndex
        $env:MASSIVELY_BENCH_IMPLEMENTATION_VERSION = $ImplementationVersion
        $env:MASSIVELY_BENCH_IMPLEMENTATION_REVISION = $ImplementationRevision

        Write-Host "$Implementation $Workload items=$ItemCount process=$ProcessIndex"
        $output = @(& $Executable 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine)
        }
        $jsonLine = $output |
            ForEach-Object { $_.ToString() } |
            Where-Object { $_.TrimStart().StartsWith('{') } |
            Select-Object -Last 1
        if (-not $jsonLine) {
            throw 'Runner completed without emitting a JSON result.'
        }
        return $jsonLine | ConvertFrom-Json
    }
    finally {
        foreach ($variable in $variables) {
            [Environment]::SetEnvironmentVariable($variable, $previous[$variable], 'Process')
        }
    }
}

$primitivesManifest = Join-Path $benchmarkRoot 'wgpu-primitives-runner\Cargo.toml'
$massivelyManifest = Join-Path $benchmarkRoot 'massively-runner\Cargo.toml'
$primitivesTarget = Join-Path $targetRoot 'wgpu-primitives'
$massivelyTarget = Join-Path $targetRoot 'massively'
Build-Runner 'wgpu-primitives runner' $primitivesManifest $primitivesTarget
Build-Runner 'Massively runner' $massivelyManifest $massivelyTarget

$executableSuffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$primitivesExecutable = Join-Path $primitivesTarget "release\wgpu-primitives-massively-comparison-runner$executableSuffix"
$massivelyExecutable = Join-Path $massivelyTarget "release\massively-comparison-runner$executableSuffix"
$repoRevision = (& git -c "safe.directory=$safeRepoRoot" -C $repoRoot rev-parse HEAD).Trim()
$repoStatus = & git -c "safe.directory=$safeRepoRoot" -C $repoRoot status --porcelain
$repoDirty = $null -ne $repoStatus -and @($repoStatus).Count -gt 0
$packageMetadata = cargo metadata --no-deps --format-version 1 --manifest-path (Join-Path $repoRoot 'Cargo.toml') | ConvertFrom-Json
$packageVersion = $packageMetadata.packages[0].version
$massivelyVersion = '0.96.0'
$massivelyRevision = 'ef9de55190529be98203aca207edab9d560d312e'

$runs = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[object]]::new()
foreach ($itemCount in $Items) {
    if ($itemCount -le 0 -or $itemCount -gt [uint32]::MaxValue) {
        throw "Item count must be between 1 and $([uint32]::MaxValue), got $itemCount."
    }
    $sampling = Get-SamplingConfig $itemCount
    foreach ($workload in $Workloads) {
        for ($processIndex = 1; $processIndex -le $Processes; $processIndex++) {
            $implementations = @(
                @{
                    Name = 'wgpu-primitives'; Executable = $primitivesExecutable;
                    Version = "wgpu-primitives-$packageVersion"; Revision = $repoRevision
                },
                @{
                    Name = 'massively'; Executable = $massivelyExecutable;
                    Version = $massivelyVersion; Revision = $massivelyRevision
                }
            )
            foreach ($implementation in $implementations) {
                try {
                    $runs.Add((Invoke-Runner `
                        $implementation.Executable `
                        $implementation.Name `
                        $implementation.Version `
                        $implementation.Revision `
                        $itemCount `
                        $workload `
                        $sampling.Warmups `
                        $sampling.WarmupMs `
                        $sampling.Samples `
                        $processIndex))
                }
                catch {
                    Write-Warning "$($implementation.Name) failed for $workload/$itemCount/process $processIndex`: $($_.Exception.Message)"
                    $failures.Add([pscustomobject][ordered]@{
                        implementation = $implementation.Name
                        workload = $workload
                        items = $itemCount
                        process_index = $processIndex
                        error = $_.Exception.Message
                    })
                }
            }
        }
    }
}
if ($runs.Count -eq 0) {
    throw 'All benchmark runs failed.'
}

$aggregates = [System.Collections.Generic.List[object]]::new()
$groups = $runs | Group-Object -Property {
    "$($_.implementation)|$($_.config.workload)|$($_.config.items)"
}
foreach ($group in $groups) {
    $first = $group.Group[0]
    $processMedians = @($group.Group | ForEach-Object { [double]$_.median_ms })
    $aggregateMedian = Get-Median $processMedians
    $aggregates.Add([pscustomobject][ordered]@{
        implementation = $first.implementation
        workload = $first.config.workload
        items = [long]$first.config.items
        process_medians_ms = $processMedians
        median_of_process_medians_ms = $aggregateMedian
        throughput_items_per_second = [double]$first.config.items / ($aggregateMedian / 1000.0)
        memory = $first.memory
    })
}

$comparisons = [System.Collections.Generic.List[object]]::new()
$cases = $aggregates | Group-Object -Property { "$($_.workload)|$($_.items)" }
foreach ($case in $cases) {
    $primitives = $case.Group | Where-Object implementation -eq 'wgpu-primitives' | Select-Object -First 1
    $massively = $case.Group | Where-Object implementation -eq 'massively' | Select-Object -First 1
    if ($null -ne $primitives -and $null -ne $massively) {
        $comparisons.Add([pscustomobject][ordered]@{
            workload = $primitives.workload
            items = $primitives.items
            wgpu_primitives_ms = $primitives.median_of_process_medians_ms
            massively_ms = $massively.median_of_process_medians_ms
            wgpu_primitives_speedup = $massively.median_of_process_medians_ms / $primitives.median_of_process_medians_ms
        })
    }
}

$result = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    repository = [ordered]@{
        revision = $repoRevision; dirty = $repoDirty; package_version = $packageVersion
    }
    comparison = [ordered]@{
        name = 'massively'; version = $massivelyVersion; revision = $massivelyRevision
    }
    host = [ordered]@{
        hostname = [Environment]::MachineName; architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    }
    config = [ordered]@{
        backend = $Backend; items = $Items; workloads = $Workloads; processes = $Processes; quick = [bool]$Quick
    }
    methodology = [ordered]@{
        timing = 'per-run public API boundary; reduction returns a host scalar, other workloads end at GPU completion'
        excluded = @('host upload', 'correctness validation', 'non-reduction readback')
        allocation_difference = 'reduction includes each implementation''s scalar readback path; other wgpu-primitives workloads reuse caller-owned outputs while Massively returns owned outputs'
    }
    runs = $runs
    failures = $failures
    aggregates = $aggregates
    comparisons = $comparisons
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $OutputPath

Write-Host "`nComparison medians:"
$comparisons |
    Sort-Object items, workload |
    Format-Table workload, items, wgpu_primitives_ms, massively_ms, wgpu_primitives_speedup -AutoSize
if ($failures.Count -gt 0) {
    Write-Warning "$($failures.Count) run(s) failed; details are recorded in the result JSON."
}
Write-Host "Machine-readable results: $OutputPath"
