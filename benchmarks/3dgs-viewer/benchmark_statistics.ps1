function Get-Median([double[]]$Values) {
    if ($Values.Count -eq 0) { throw 'Cannot take the median of an empty sample.' }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2.0)
    if ($sorted.Count % 2 -eq 1) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
}
