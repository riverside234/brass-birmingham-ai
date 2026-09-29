$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$record=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$prior=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-releases/1.1.0/Assembly-CSharp.dll.manifest.json') -Raw|ConvertFrom-Json
$woven=Get-Content -LiteralPath (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') -Raw|ConvertFrom-Json
$validation=Get-Content -LiteralPath (Join-Path $build 'validation.json') -Raw|ConvertFrom-Json
if($record.version -ne $candidate.version -or $record.onlineEnabled -or $record.status -ne 'installed'){throw 'Installation record mismatch'}
foreach($file in $candidate.files){
 $installed=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $installed).Hash -ne $file.sha256){throw 'Installed candidate fingerprint mismatch'}
 if(@($record.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Recorded fingerprint mismatch'}
 if(@($validation.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Validation fingerprint mismatch'}
}
if(-not $validation.passed -or -not $validation.unityPlayerValidation.passed){throw 'Combined regression gates have not passed'}
if(@($prior.methods|Where-Object {$_ -notin $woven.methods}).Count -gt 0){throw 'A previous patch hook was lost'}
$previousAiHooks=$null
if($candidate.version -in @('1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3')){
 $previousAiVersion=if($candidate.version -eq '1.4.3'){'1.4.2'}elseif($candidate.version -eq '1.4.2'){'1.4.1'}elseif($candidate.version -eq '1.4.1'){'1.4.0'}elseif($candidate.version -eq '1.4.0'){'1.3.2'}elseif($candidate.version -eq '1.3.2'){'1.3.1'}else{'1.3.0'}
 $aiPrior=Get-Content -LiteralPath (Join-Path $root ('artifacts/patch-releases/'+$previousAiVersion+'/Assembly-CSharp.dll.manifest.json')) -Raw|ConvertFrom-Json
 if(@($aiPrior.methods|Where-Object {$_ -notin $woven.methods}).Count -gt 0){throw ('A '+$previousAiVersion+' AI/UI hook was lost')}
 $previousAiHooks=$aiPrior.methods.Count
}
if($woven.gameOwnedBonusSites -ne $prior.gameOwnedBonusSites){throw 'Merchant bonus rewrites changed'}
if((Get-FileHash -LiteralPath (Join-Path $root 'artifacts/patch-baseline/Assembly-CSharp.dll')).Hash -ne $candidate.stockSha256){throw 'Original backup fingerprint mismatch'}
[void][Reflection.Assembly]::LoadFrom((Join-Path $build 'Mono.Cecil.dll'))
$assembly=[Mono.Cecil.AssemblyDefinition]::ReadAssembly((Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'))
$retained=$null
try{
 $menu=$assembly.MainModule.Types|Where-Object FullName -eq 'BrassApplication.UseCases.Menu.MainMenu.MainMenuUseCase'
 $method=$menu.Methods|Where-Object Name -eq 'ShowOnlineGamesView'
 if(-not($method.Body.Instructions|Where-Object {$_.Operand -and $_.Operand.ToString() -like '*BrassPatch.OfflineMode::ShowOnlineDisabled*'})){throw 'Offline menu hook missing'}
 if($candidate.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3')){
  $previous=[Mono.Cecil.AssemblyDefinition]::ReadAssembly((Join-Path $root 'artifacts/patch-releases/1.2.0/Assembly-CSharp.dll'))
  try{
   $retained=0
   foreach($type in $previous.MainModule.Types){foreach($oldMethod in $type.Methods|Where-Object HasBody){
    $calls=@($oldMethod.Body.Instructions|Where-Object {$_.Operand -is [Mono.Cecil.MethodReference] -and $_.Operand.DeclaringType.Scope.Name -eq 'BrassPatch'}|ForEach-Object {$_.Operand.FullName})
    if(-not $calls.Count){continue}
    $newType=$assembly.MainModule.Types|Where-Object FullName -eq $type.FullName
    $newMethod=$newType.Methods|Where-Object FullName -eq $oldMethod.FullName
    if(-not $newMethod){throw 'A previous method disappeared'}
    foreach($call in $calls){if(-not($newMethod.Body.Instructions|Where-Object {$_.Operand -is [Mono.Cecil.MethodReference] -and $_.Operand.FullName -eq $call})){throw ('A 1.2.0 helper call was lost: '+$oldMethod.FullName)}}
    $retained++
   }}
   if($retained -lt 88){throw 'Not all 1.2.0 hooks were accounted for'}
  }finally{$previous.Dispose()}
 }
}finally{$assembly.Dispose()}
[ordered]@{passed=$true;patchVersion=$candidate.version;verifiedUtc=[DateTime]::UtcNow.ToString('o');priorManifestHooksVerified=$prior.methods.Count;previousAiReleaseHooksVerified=$previousAiHooks;previousHelperBearingMethodsVerified=$retained;combinedHooks=$woven.methods.Count;bonusRewrites=$woven.gameOwnedBonusSites;onlineDisabled=$true;originalBackupVerified=$true;files=$candidate.files;cityLocatorChecks=$validation.unityPlayerValidation.cityLocatorChecks;unitySuites=$validation.unityPlayerValidation.suites;unityEvidenceRetainedFrom=$validation.unityPlayerValidation.retainedFrom;unityEvidenceScope=$validation.unityPlayerValidation.scope;userSavesOpened=$false}|ConvertTo-Json -Depth 8|Tee-Object -FilePath (Join-Path $build 'installation-verification.json')
