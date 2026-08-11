param(
    [long[]]$Items = @(1000000, 10000000, 100000000),
    [ValidateSet('bounded16', 'full_width')]
    [string[]]$Workloads = @('bounded16', 'full_width'),
    [ValidateSet('resident', 'round_trip')]
    [string[]]$Modes = @('resident', 'round_trip'),
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
$targetRoot = Join-Path $repoRoot 'target\wgpu-sort-comparison'

if (-not $OutputPath) {
    $OutputPath = Join-Path $benchmarkRoot 'results\latest.json'
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}

if ($Quick) {
    $Items = @(1000000)
    $Workloads = @('bounded16', 'full_width')
    $Modes = @('resident')
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
    param(
        [string]$Name,
        [string]$Manifest,
        [string]$TargetDirectory
    )
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
        [string]$ImplementationVersion,
        [string]$ImplementationRevision,
        [long]$ItemCount,
        [string]$Workload,
        [string]$Mode,
        [int]$Warmups,
        [long]$WarmupMs,
        [int]$Samples,
        [int]$ProcessIndex
    )
    $variables = @(
        'WGPU_BACKEND',
        'WGPU_SORT_BENCH_ITEMS',
        'WGPU_SORT_BENCH_WORKLOAD',
        'WGPU_SORT_BENCH_MODE',
        'WGPU_SORT_BENCH_WARMUPS',
        'WGPU_SORT_BENCH_WARMUP_MS',
        'WGPU_SORT_BENCH_SAMPLES',
        'WGPU_SORT_BENCH_PROCESS_INDEX',
        'WGPU_SORT_BENCH_IMPLEMENTATION_VERSION',
        'WGPU_SORT_BENCH_IMPLEMENTATION_REVISION'
    )
    $previous = @{}
    foreach ($variable in $variables) {
        $previous[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
    }

    try {
        $env:WGPU_BACKEND = $Backend
        $env:WGPU_SORT_BENCH_ITEMS = [string]$ItemCount
        $env:WGPU_SORT_BENCH_WORKLOAD = $Workload
        $env:WGPU_SORT_BENCH_MODE = $Mode
        $env:WGPU_SORT_BENCH_WARMUPS = [string]$Warmups
        $env:WGPU_SORT_BENCH_WARMUP_MS = [string]$WarmupMs
        $env:WGPU_SORT_BENCH_SAMPLES = [string]$Samples
        $env:WGPU_SORT_BENCH_PROCESS_INDEX = [string]$ProcessIndex
        $env:WGPU_SORT_BENCH_IMPLEMENTATION_VERSION = $ImplementationVersion
        $env:WGPU_SORT_BENCH_IMPLEMENTATION_REVISION = $ImplementationRevision

        Write-Host "$ImplementationVersion $Mode $Workload items=$ItemCount process=$ProcessIndex"
        $json = & $Executable
        if ($LASTEXITCODE -ne 0) {
            throw "Runner $Executable failed."
        }
        return $json | ConvertFrom-Json
    }
    finally {
        foreach ($variable in $variables) {
            [Environment]::SetEnvironmentVariable($variable, $previous[$variable], 'Process')
        }
    }
}

$primitivesManifest = Join-Path $benchmarkRoot 'lampshade-runner\Cargo.toml'
$comparisonManifest = Join-Path $benchmarkRoot 'wgpu-sort-runner\Cargo.toml'
$primitivesTarget = Join-Path $targetRoot 'lampshade'
$comparisonTarget = Join-Path $targetRoot 'wgpu-sort'

Build-Runner 'Lampshade runner' $primitivesManifest $primitivesTarget
Build-Runner 'wgpu_sort runner' $comparisonManifest $comparisonTarget

$executableSuffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$primitivesExecutable = Join-Path $primitivesTarget "release\lampshade-wgpu-sort-comparison-runner$executableSuffix"
$comparisonExecutable = Join-Path $comparisonTarget "release\wgpu-sort-comparison-runner$executableSuffix"
$repoRevision = (& git -c "safe.directory=$safeRepoRoot" -C $repoRoot rev-parse HEAD).Trim()
$repoStatus = & git -c "safe.directory=$safeRepoRoot" -C $repoRoot status --porcelain
$repoDirty = $null -ne $repoStatus -and @($repoStatus).Count -gt 0
$packageMetadata = cargo metadata --no-deps --format-version 1 --manifest-path (Join-Path $repoRoot 'Cargo.toml') | ConvertFrom-Json
$packageVersion = $packageMetadata.packages[0].version
$pinnedWgpuSortRevision = '4cb640e8cae28eba0149d470c5168cc2853466dd'

$runs = [System.Collections.Generic.List[object]]::new()
foreach ($itemCount in $Items) {
    if ($itemCount -le 0 -or $itemCount -gt [uint32]::MaxValue) {
        throw "Item count must be between 1 and $([uint32]::MaxValue), got $itemCount."
    }
    $sampling = Get-SamplingConfig $itemCount
    foreach ($workload in $Workloads) {
        foreach ($mode in $Modes) {
            for ($processIndex = 1; $processIndex -le $Processes; $processIndex++) {
                $runs.Add((Invoke-Runner `
                    $primitivesExecutable `
                    "lampshade-$packageVersion" `
                    $repoRevision `
                    $itemCount `
                    $workload `
                    $mode `
                    $sampling.Warmups `
                    $sampling.WarmupMs `
                    $sampling.Samples `
                    $processIndex))
                $runs.Add((Invoke-Runner `
                    $comparisonExecutable `
                    'wgpu_sort-git' `
                    $pinnedWgpuSortRevision `
                    $itemCount `
                    $workload `
                    $mode `
                    $sampling.Warmups `
                    $sampling.WarmupMs `
                    $sampling.Samples `
                    $processIndex))
            }
        }
    }
}

$aggregates = [System.Collections.Generic.List[object]]::new()
$groups = $runs | Group-Object -Property {
    "$($_.implementation)|$($_.config.mode)|$($_.config.workload)|$($_.config.items)"
}
foreach ($group in $groups) {
    $first = $group.Group[0]
    $processMedians = @($group.Group | ForEach-Object { [double]$_.median_ms })
    $aggregateMedian = Get-Median $processMedians
    $aggregates.Add([pscustomobject][ordered]@{
        implementation = $first.implementation
        mode = $first.config.mode
        workload = $first.config.workload
        items = [long]$first.config.items
        process_medians_ms = $processMedians
        median_of_process_medians_ms = $aggregateMedian
        throughput_pairs_per_second = [double]$first.config.items / ($aggregateMedian / 1000.0)
        memory = $first.memory
    })
}

$result = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    repository = [ordered]@{
        revision = $repoRevision
        dirty = $repoDirty
        package_version = $packageVersion
    }
    comparison = [ordered]@{
        name = 'wgpu_sort'
        revision = $pinnedWgpuSortRevision
    }
    config = [ordered]@{
        backend = $Backend
        items = $Items
        workloads = $Workloads
        modes = $Modes
        processes = $Processes
        quick = [bool]$Quick
    }
    runs = $runs
    aggregates = $aggregates
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $OutputPath

Write-Host "`nAggregate medians:"
$aggregates |
    Sort-Object items, workload, mode, implementation |
    Format-Table implementation, mode, workload, items, median_of_process_medians_ms, throughput_pairs_per_second -AutoSize
Write-Host "Machine-readable results: $OutputPath"
