$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$matches=@();$parts=@();$checks=0
foreach($seed in 72001,72002){foreach($revised in 0,1){
 $name="local-ai-conversion-single-$seed-$revised.json";$path=Join-Path $build $name
 $result=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 if(-not $result.passed -or $result.matches.Count -ne 1){throw 'Incomplete comparison component'}
 foreach($file in $manifest.files){if(@($result.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or (Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Comparison component fingerprint mismatch'}}
 $match=$result.matches[0]
 if($match.seed -ne $seed -or [int][bool]$match.revised -ne $revised -or $match.timed -or $match.work -ne 240 -or $match.actions -ne 124 -or $match.turns.Count -ne 4 -or @($match.turns|Where-Object {$_ -ne 16}).Count){throw 'Comparison component coverage mismatch'}
 $matches+=$match;$checks+=$result.checks
 $parts+=[ordered]@{path=$name;sha256=(Get-FileHash -LiteralPath $path).Hash}
}}
[ordered]@{passed=$true;checks=$checks;matches=$matches;components=$parts;testedFiles=$manifest.files;scope='Four complete, isolated, fixed-work 240-node comparisons. Concurrent wall times are not foreground UI latency.'}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-conversion-matches.json') -Encoding UTF8
Write-Output 'Verified and merged all four fixed-work comparison games.'
