param([string]$BuildDirectory='artifacts/mixed-ai/20261102/release-1.4.51')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
[void][Reflection.Assembly]::LoadFrom((Join-Path $repo 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'))
function Types($type){$type;foreach($child in $type.NestedTypes){Types $child}}
function Bodies($module){
 $result=@{}
 foreach($root in $module.Types){foreach($type in (Types $root)){foreach($method in $type.Methods){
  if($method.HasBody){$result[$method.FullName]=@($method.Body.Instructions|ForEach-Object ToString)}
 }}}
 return $result
}
$before=[Mono.Cecil.ModuleDefinition]::ReadModule((Join-Path $repo 'artifacts/nn-il/20261102/release-1.4.50/Managed/Assembly-CSharp.dll'))
$after=[Mono.Cecil.ModuleDefinition]::ReadModule((Join-Path $repo ($BuildDirectory+'/Managed/Assembly-CSharp.dll')))
try{
 $a=Bodies $before;$b=Bodies $after
 if($a.Count -ne $b.Count){throw 'Game method inventory changed.'}
 $changed=@($a.Keys|Where-Object {($a[$_]-join "`n") -cne ($b[$_]-join "`n")})
 $target='System.Void Frameworks.Utils.UIProperties::.cctor()'
 if($changed.Count -ne 1 -or $changed[0] -ne $target){throw 'Game code changed outside animation defaults.'}
 $instructions=@($after.Types|Where-Object FullName -eq 'Frameworks.Utils.UIProperties'|ForEach-Object Methods|Where-Object Name -eq '.cctor'|ForEach-Object {$_.Body.Instructions})
 $diff=@(for($i=0;$i -lt $a[$target].Count;$i++){if($a[$target][$i] -cne $b[$target][$i]){$i}})
 if($diff.Count -ne 2){throw 'Expected exactly two animation constants.'}
 $fields=@(foreach($i in $diff){
  if($a[$target][$i] -notmatch 'ldc.r4 1$' -or $b[$target][$i] -notmatch 'ldc.r4 2$'){throw 'Unexpected animation delta.'}
  $instructions[$i+1].Operand.Name
 })
 if(($fields|Sort-Object)-join '|' -ne 'CurrentAnimationSpeed|DefaultAnimationSpeed'){throw 'Unexpected animation fields.'}
 @{passed=$true;unchangedMethods=$a.Count-1;changedMethod=$target;fields=$fields;default=2}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $repo 'artifacts/mixed-ai/20261102/game-overlay-verification.json')
 Write-Output ('Verified two animation constants; '+($a.Count-1)+' other game methods unchanged.')
}finally{$before.Dispose();$after.Dispose()}
