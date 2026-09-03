$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'benchmark_statistics.ps1')
if ((Get-Median @(1, 2, 3)) -ne 2) { throw 'Three-value median failed.' }
if ((Get-Median @(7, 1, 5, 2, 6, 4, 3)) -ne 4) { throw 'Seven-value median failed.' }
if ((Get-Median @(1, 2, 3, 4)) -ne 2.5) { throw 'Even median failed.' }
if ((Get-Median @(1.2)) -ne 1.2) { throw 'Singleton median failed.' }
Write-Output 'Median regression tests passed (odd, even, unsorted, singleton).'
