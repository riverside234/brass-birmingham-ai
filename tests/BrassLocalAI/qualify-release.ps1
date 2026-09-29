$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -eq '1.3.2'){& (Join-Path $PSScriptRoot 'qualify-gameplay-release.ps1');return}
if($candidate.version -eq '1.4.0'){& (Join-Path $PSScriptRoot 'qualify-cash-release.ps1');return}
if($candidate.version -eq '1.4.1'){& (Join-Path $PSScriptRoot 'qualify-conversion-release.ps1');return}
if($candidate.version -eq '1.4.2'){& (Join-Path $PSScriptRoot 'qualify-presentation-release.ps1');return}
if($candidate.version -eq '1.4.3'){& (Join-Path $PSScriptRoot 'qualify-budget-release.ps1');return}
$validationPath=Join-Path $build 'validation.json'
$validation=Get-Content -LiteralPath $validationPath -Raw|ConvertFrom-Json
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence does not match the final candidate'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
if(-not $validation.passed){throw 'Managed validation failed'}
Assert-Files $validation.files
$evidence=@{}
foreach($name in @('local-ai-runtime-0','local-ai-native','city-render-result','unity-isolated-result','local-ai-live')){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)}
 Assert-Files $item.testedFiles
 $evidence[$name]=$item
}
if($evidence['local-ai-native'].turns.Count -ne 36){throw 'Native strategy/configuration coverage is incomplete'}
if($evidence['local-ai-live'].matches.Count -ne 6){throw 'Full live save/reload coverage is incomplete'}
if($evidence['unity-isolated-result'].suites.Count -ne 10){throw 'Existing Unity suite coverage is incomplete'}
if($evidence['city-render-result'].renderedFiles.Count -ne 18){throw 'City rendering coverage is incomplete'}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed){throw 'H5 profile provenance failed'}

# The long-running matrices use private DLL copies. Require the exact same policy
# sources and verify all changed IL is confined to presentation/test entry points.
$snapshot=Join-Path $root 'artifacts/local-ai-runtime-snapshot'
$sources=Get-Content -LiteralPath (Join-Path $snapshot 'policy-sources.json') -Raw|ConvertFrom-Json
foreach($file in $sources){if((Get-FileHash -LiteralPath $file.Path).Hash -ne $file.Hash){throw ('Policy source changed since full-game tests: '+$file.Path)}}
$sourcePaths=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/BrassLocalAI/Host'),(Join-Path $root 'src/BrassLocalAI/Policy') -Filter '*.cs' -File|ForEach-Object FullName|Sort-Object)
if(($sourcePaths -join '|') -ne (($sources|ForEach-Object Path|Sort-Object) -join '|')){throw 'Policy source set changed'}
[void][Reflection.Assembly]::LoadFrom((Join-Path $build 'Mono.Cecil.dll'))
$currentGame=[Mono.Cecil.AssemblyDefinition]::ReadAssembly((Join-Path $build 'Managed/Assembly-CSharp.dll'))
try{
 $brewery=$currentGame.MainModule.Types|Where-Object FullName -eq 'BoardGameRules.Entities.Industries.Brewery'
 $callback=$brewery.Fields|Where-Object Name -eq '_numberOfBeersFunc'
 if(@($callback.CustomAttributes|Where-Object {$_.AttributeType.FullName -eq 'Newtonsoft.Json.JsonIgnoreAttribute'}).Count -ne 1){throw 'Brewery callback serialization repair is missing'}
}finally{$currentGame.Dispose()}
function Types($module){foreach($type in $module.Types){Walk $type}}
function Walk($type){$type;foreach($nested in $type.NestedTypes){Walk $nested}}
function Bodies($assembly){
 $map=@{}
 foreach($type in (Types $assembly.MainModule)){
  foreach($method in $type.Methods|Where-Object HasBody){
   $body=@($method.Body.Instructions|ForEach-Object {$_.OpCode.Name+' '+$_.Operand}) -join "`n"
   $body+='|locals:'+(@($method.Body.Variables|ForEach-Object {$_.VariableType.FullName}) -join ',')
   $body+='|handlers:'+(@($method.Body.ExceptionHandlers|ForEach-Object {''+$_.HandlerType+':'+$_.TryStart+':'+$_.TryEnd+':'+$_.HandlerStart+':'+$_.HandlerEnd+':'+$_.CatchType+':'+$_.FilterStart}) -join ',')
   $map[$method.FullName]=$body
  }
 }
 return $map
}
$differences=@()
foreach($dir in @($snapshot,(Join-Path $root 'artifacts/local-ai-identical-runtime'))){
 $tested=Get-Content -LiteralPath (Join-Path $dir 'build-manifest.json') -Raw|ConvertFrom-Json
 foreach($file in $tested.files){
  $oldPath=Join-Path $dir $file.name
  if((Get-FileHash -LiteralPath $oldPath).Hash -ne $file.sha256){throw 'Private test runtime fingerprint changed'}
  $newPath=Join-Path $build ('Managed/'+$file.name)
  if((Get-FileHash -LiteralPath $newPath).Hash -eq $file.sha256){continue}
  $old=[Mono.Cecil.AssemblyDefinition]::ReadAssembly($oldPath);$new=[Mono.Cecil.AssemblyDefinition]::ReadAssembly($newPath)
  try{
   $before=Bodies $old;$after=Bodies $new
   foreach($method in (@($before.Keys)+@($after.Keys)|Sort-Object -Unique)){
    if($before[$method] -ceq $after[$method]){continue}
    $allowed=switch($file.name){
     'Assembly-CSharp.dll' {$method -match ' PlayerCreateGameView::(SetPlayer|BrassPatch_Original_SetPlayer)\(' -or
       $method -match ' BrassApplication.UseCases.Game.PlayerActions.PlayerActionsUseCase::(BrassPatch_Original_)?(StartBuildAction|StartNetworkAction|StartDevelopAction|StartSellAction|StartLoanAction|StartScoutAction|StartPassAction|UndoAction|EndTurn)\(' -or
       $method -match ' BrassApplication.UseCases.Game.(Board.BoardUseCase::(BrassPatch_Original_)?RegionSelected|HandOfCards.HandOfCardsUseCase::(BrassPatch_Original_)?OnCardSelected|Replay.ReplayUseCase::(BrassPatch_Original_)?SkipReplay)\('}
     'BrassPatch.dll' {$method -match ' BrassPatch.(StartupFixes|CityLocator.CityLocatorFrame)[/+:]'}
     'BrassLegacyAI.dll' {$method -match ' BrassLegacyAI\.(LocalMenu|LocalTurnBridge|AiTurnControls|AiTurnDispatch|StrategyTooltip|TurnInputGuard)[/+:]'}
     default {$false}
    }
    if(-not $allowed){throw ('Full-game evidence needs rerunning; non-UI IL changed: '+$file.name+' '+$method)}
    $differences+=[ordered]@{runtime=(Split-Path -Leaf $dir);file=$file.name;method=$method}
   }
   if($file.name -eq 'BrassLegacyAI.dll'){
    foreach($resource in $old.MainModule.Resources){
     $other=$new.MainModule.Resources|Where-Object Name -eq $resource.Name
     if(-not $other -or [Convert]::ToBase64String($resource.GetResourceData()) -cne [Convert]::ToBase64String($other.GetResourceData())){throw 'Embedded profile changed since full-game tests'}
    }
   }
  }finally{$old.Dispose();$new.Dispose()}
 }
}
$matrices=@();$total=0
foreach($count in 2,3,4){foreach($kind in 'runtime','identical'){
 $path=Join-Path $build ("local-ai-$kind-$count.json")
 $result=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 $expected=if($kind -eq 'runtime'){$count*24}else{12}
 if(-not $result.passed -or $result.matches.Count -ne $expected){throw "Incomplete $kind matrix for $count players"}
 $total+=$result.matches.Count
 $matrices+=[ordered]@{kind=$kind;players=$count;matches=$result.matches.Count;assertions=$result.checks;workLimit=1;sha256=(Get-FileHash -LiteralPath $path).Hash;runtime=$(if($kind -eq 'runtime'){'local-ai-runtime-snapshot'}else{'local-ai-identical-runtime'})}
}}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;breweryCallbackIgnored=$true;
 fullGames=$total;liveMixedGames=$evidence['local-ai-live'].matches;liveAssertions=$evidence['local-ai-live'].checks;matrices=$matrices;scope='Complete games use deterministic one-node search with complete legal incumbents. Native turns use shipped budgets and exercise input guards. This is reliability evidence, not playing-strength qualification. Policy, simulation, and domain rule IL remain unchanged; the allowlisted UI entry guards, live thinking allowance and city rendering are tested on the final binaries.';
 h5Provenance=$h5;policySourcesVerified=$sources.Count;presentationOnlyDifferences=$differences;smokeAssertions=$evidence['local-ai-runtime-0'].checks;
 nativeTurns=$evidence['local-ai-native'].turns.Count;nativeAssertions=$evidence['local-ai-native'].checks;
 cityAssertions=$evidence['unity-isolated-result'].cityLocatorChecks;cityRenderedFiles=18;existingUnitySuites=$evidence['unity-isolated-result'].suites}
$qualification|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $qualification -Force
$validation|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $validationPath -Encoding UTF8
Write-Output "Qualified combined patch: $total completed strategy games and 36 native turns; all existing rules/UI gates passed."
