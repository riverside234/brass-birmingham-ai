param([switch]$Installed)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$bankId='bank-804c85204da65824cb9b'
$bankRoot=Join-Path $root ('training/opponents/bank/'+$bankId)
$manifestPath=Join-Path $bankRoot 'manifest.json'
$evidence=Get-Content -LiteralPath (Join-Path $root 'training/opponents/evaluations/qualification-d9db1fbda13fcfcb173d/checks.json') -Raw|ConvertFrom-Json
if((Get-FileHash -LiteralPath $manifestPath).Hash -ne $evidence.bank_manifest_sha256){throw 'H5 manifest differs from its correctness record'}
$manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
if($manifest.id -ne $bankId){throw 'Unexpected frozen bank identity'}
$sourceChecks=0
foreach($entry in $manifest.payload.source_files.PSObject.Properties){
 foreach($path in @((Join-Path $bankRoot ('source/'+$entry.Name)))){
  if((Get-FileHash -LiteralPath $path).Hash -ne $entry.Value){throw ('Frozen or active research source changed: '+$path)}
 }
 $sourceChecks++
}
$runtimeRoot=Join-Path $root ('artifacts/legacy-ai/bank/'+$bankId+'/runtime')
foreach($file in $manifest.payload.runtime_files){
 if((Get-FileHash -LiteralPath (Join-Path $runtimeRoot $file.name)).Hash -ne $file.sha256){throw ('Frozen runtime changed: '+$file.name)}
}
$pool=Get-Content -LiteralPath (Join-Path $root 'training/opponents/archive/pool-v2.json') -Raw|ConvertFrom-Json
$managed=if($Installed){Join-Path $root 'BrassBirminghamUnity_Data/Managed'}else{Join-Path $build 'Managed'}
$dll=Join-Path $managed 'BrassLegacyAI.dll'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$expected=$candidate.files|Where-Object name -eq 'BrassLegacyAI.dll'
if((Get-FileHash -LiteralPath $dll).Hash -ne $expected.sha256){throw 'AI binary differs from the qualified candidate'}
[void][Reflection.Assembly]::LoadFrom((Join-Path $build 'Mono.Cecil.dll'))
$assembly=[Mono.Cecil.AssemblyDefinition]::ReadAssembly($dll)
$profiles=@();$v3Profiles=@()
try{
 foreach($key in @('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')){
  $profileName=$key+'-v2';$policyId=$bankId+'/'+$profileName
  $purpose=if($key -eq 'pottery-flex'){'evaluation'}else{'training'}
  $admission=$pool.$purpose.PSObject.Properties[$policyId].Value
  $reportPath=Join-Path $root $admission.qualification
  if((Get-FileHash -LiteralPath $reportPath).Hash -ne $admission.sha256){throw 'H5 qualification report changed'}
  $report=Get-Content -LiteralPath $reportPath -Raw|ConvertFrom-Json
  $qualified=$report.profiles.PSObject.Properties[$profileName].Value
  if($report.bank -ne $bankId -or $qualified.policy_id -ne $policyId -or -not $qualified.niche_qualified -or -not $qualified.($purpose+'_eligible')){throw ('H5 admission mismatch: '+$profileName)}
  $resource=@($assembly.MainModule.Resources|Where-Object Name -eq ('BrassLegacyAI.Profiles.'+$profileName+'.json'))
  if($resource.Count -ne 1){throw ('Missing or duplicate embedded profile: '+$profileName)}
  $bytes=$resource[0].GetResourceData()
  $sha=[Security.Cryptography.SHA256]::Create()
  try{$hash=([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
  $pin=$manifest.payload.source_files.PSObject.Properties['configs/legacy-ai/'+$profileName+'.json'].Value
  if($hash -ne $pin){throw ('Embedded profile differs from H5: '+$profileName)}
  $profile=[Text.Encoding]::UTF8.GetString($bytes).TrimStart([char]0xFEFF)|ConvertFrom-Json
  if($profile.id -ne ('legacy-hard-'+$profileName) -or $profile.mode -ne 'S1' -or -not $profile.intent_enabled){throw 'Wrong strategy generation or search mode'}
  $profiles+=[ordered]@{strategy=$key;profile=$profileName;mode=$profile.mode;intentEnabled=$profile.intent_enabled;sha256=$hash;researchPool=$purpose}
  if($candidate.version -in @('1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){
   $v3Name=$key+'-v3'
   $resource=@($assembly.MainModule.Resources|Where-Object Name -eq ('BrassLegacyAI.Profiles.'+$v3Name+'.json'))
   if($resource.Count -ne 1){throw 'Missing version 3 resource'}
   $bytes=$resource[0].GetResourceData();$sha=[Security.Cryptography.SHA256]::Create()
   try{$v3Hash=([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
   if($v3Hash -ne (Get-FileHash -LiteralPath (Join-Path $root ('configs/legacy-ai/'+$v3Name+'.json'))).Hash.ToLowerInvariant()){throw 'V3 embedded bytes differ from source'}
   $v3=[Text.Encoding]::UTF8.GetString($bytes).TrimStart([char]0xFEFF)|ConvertFrom-Json
   if($v3.id -ne ('legacy-hard-'+$v3Name) -or $v3.cash_policy -ne 1){throw 'Wrong V3 cash policy identity'}
   $v3.PSObject.Properties.Remove('cash_policy');if($candidate.version -in @('1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){if($v3.conversion_policy -ne 1){throw 'Missing conversion policy'};$v3.PSObject.Properties.Remove('conversion_policy')};$v3.id=$profile.id
   if(($v3|ConvertTo-Json -Depth 10 -Compress) -cne ($profile|ConvertTo-Json -Depth 10 -Compress)){throw 'V3 strategy settings differ beyond the cash policy and version'}
   $v3Profiles+=[ordered]@{strategy=$key;profile=$v3Name;sha256=$v3Hash;cashPolicy=1;researchPool='See current versioned v3 bank qualification; no v2 strength inherited'}
  }
 }
}finally{$assembly.Dispose()}
function Source([string]$path){return [IO.File]::ReadAllText($path).TrimStart([char]0xFEFF).Replace("`r`n","`n")}
$frozenTactics=Source (Join-Path $bankRoot 'source/src/BrassLegacyAI/Tactics.cs')
$localTactics=Source (Join-Path $root 'src/BrassLocalAI/Policy/Tactics.cs')
if($frozenTactics.Replace('v["players"][seat]','Wire.Player(v,seat)') -cne $localTactics){throw 'Tactics differ beyond the reviewed active-seat lookup adaptation'}
# Record the deployment fork explicitly; it is not the frozen worker identity.
# Planner/Evaluator adaptations are reviewed in LOCAL_AI_1.3.1_RELEASE.md.
$sourcePins=@('Planner','Tactics','Evaluator'|ForEach-Object {
 [ordered]@{file=('src/BrassLocalAI/Policy/'+$_+'.cs');sha256=(Get-FileHash -LiteralPath (Join-Path $root ('src/BrassLocalAI/Policy/'+$_+'.cs'))).Hash}
})
$result=[ordered]@{passed=$true;checkedUtc=[DateTime]::UtcNow.ToString('o');installed=[bool]$Installed;bank=$bankId;
 bankManifestSha256=(Get-FileHash -LiteralPath $manifestPath).Hash;archivedResearchSourceFilesVerified=$sourceChecks;
 frozenRuntimeFilesVerified=@($manifest.payload.runtime_files).Count;aiDllSha256=$expected.sha256;profiles=$profiles;v3Profiles=$v3Profiles;deploymentSources=$sourcePins;
 scope='Six archived S1 v2 profiles exactly match H5. H3 tactics differ only in player-seat lookup. V3 profiles, when present, preserve those styles and enable the deployment cash repair with separate identities. Deployment behavior is not the frozen research worker or its qualified budget; new v3 requires separate admission.'}
if($candidate.version -in @('1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){
 $current=Get-Content -LiteralPath (Join-Path $root 'training/opponents/current.json') -Raw|ConvertFrom-Json
 $activeBank=$current.bank
 if($candidate.version -eq '1.4.7'){
  # Qualify the game's preserved policy against the immutable deployed bank.
  # Concurrent research development is not an input to this game package.
  $deployed=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-releases/1.4.6/local-ai-h5.json') -Raw|ConvertFrom-Json
  if(-not $deployed.passed){throw 'Qualified deployed-bank provenance is required'}
  $current=[pscustomobject]@{bank=$deployed.v3Bank}
 }
 $v3Root=Join-Path $root ('training/opponents/bank/'+$current.bank)
 $v3Manifest=Get-Content -LiteralPath (Join-Path $v3Root 'manifest.json') -Raw|ConvertFrom-Json
 if($candidate.version -eq '1.4.7' -and (Get-FileHash -LiteralPath (Join-Path $v3Root 'manifest.json')).Hash -ne $deployed.v3BankManifestSha256){throw 'Deployed immutable bank manifest changed'}
 $activeDifferences=@()
 foreach($entry in $v3Manifest.payload.source_files.PSObject.Properties){
  $archived=Join-Path $v3Root ('source/'+$entry.Name)
  if((Get-FileHash -LiteralPath $archived).Hash -ne $entry.Value){throw ('Frozen V3 bank source differs: '+$entry.Name)}
  $active=Join-Path $root $entry.Name
  if((Get-FileHash -LiteralPath $active).Hash -ne $entry.Value){
   if($candidate.version -ne '1.4.7'){throw 'V3 bank source differs'}
   $activeDifferences+=$entry.Name
  }
 }
 foreach($profile in $v3Profiles){$pin=$v3Manifest.payload.source_files.PSObject.Properties['configs/legacy-ai/'+$profile.profile+'.json'].Value;if($profile.sha256 -ne $pin){throw 'Embedded V3 profile differs from bank'}}
 foreach($name in 'Conversion','CashManagement','Evaluator'){
  $reference=if($candidate.version -eq '1.4.7'){Join-Path $v3Root ('source/src/BrassLegacyAI/'+$name+'.cs')}else{Join-Path $root ('src/BrassLegacyAI/'+$name+'.cs')}
  if((Source (Join-Path $root ('src/BrassLocalAI/Policy/'+$name+'.cs'))) -cne (Source $reference)){throw 'Shared v3 policy sources differ'}
 }
 $localPlanner=(Source (Join-Path $root 'src/BrassLocalAI/Policy/Planner.cs')).Replace('!Wire.ValidPlayerCount(p.player_count)','p.player_count!=4').Replace('(p.mode!="S0" && p.mode!="S1")','(p.mode!="S0" && p.mode!="S1" && p.mode!="S2")')
 $reference=if($candidate.version -eq '1.4.7'){Join-Path $v3Root 'source/src/BrassLegacyAI/Planner.cs'}else{Join-Path $root 'src/BrassLegacyAI/Planner.cs'}
 if($localPlanner -cne (Source $reference)){throw 'Frozen research/deployment planner differs beyond supported modes/counts'}
 $result.v3Bank=$current.bank;$result.v3BankManifestSha256=(Get-FileHash -LiteralPath (Join-Path $v3Root 'manifest.json')).Hash
 if($candidate.version -eq '1.4.7'){$result|Add-Member -NotePropertyName activeResearchDifferences -NotePropertyValue $activeDifferences;$result|Add-Member -NotePropertyName activeResearchBank -NotePropertyValue $activeBank}
}
$name=if($Installed){'local-ai-h5-installed.json'}else{'local-ai-h5.json'}
$result|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $build $name) -Encoding UTF8
Write-Output ('Profile provenance passed: 6 preserved S1 v2 profiles, '+$v3Profiles.Count+' deployment v3 profiles, '+$sourceChecks+' research sources and '+@($manifest.payload.runtime_files).Count+' frozen runtime files verified.')
