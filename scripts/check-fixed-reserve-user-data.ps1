param([ValidateSet('Capture','Verify')][string]$Action='Verify')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$study=Join-Path $root 'artifacts/fixed-reserve/20261029'
$receipt=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
$files=@()
foreach($base in $receipt.saveDataPaths){foreach($folder in 'games','GameHistoryStatistics'){
 $path=Join-Path $base $folder
 if(Test-Path -LiteralPath $path){
  $files+=@(Get-ChildItem -LiteralPath $path -File -Recurse -Force|Sort-Object FullName|ForEach-Object{@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
 }
}}
$preferences=@()
$preferencesMeasured=$true;$key=$null
try{$key=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Cublo\Brass Birmingham')}
catch [System.Security.SecurityException]{$preferencesMeasured=$false}
catch [System.UnauthorizedAccessException]{$preferencesMeasured=$false}
if($null -ne $key){try{
 foreach($name in @($key.GetValueNames()|Where-Object{$_ -like 'BrassLocalAI*'}|Sort-Object)){
  $value=$key.GetValue($name)
  $bytes=[Text.Encoding]::UTF8.GetBytes(($value|ConvertTo-Json -Compress -Depth 5))
  $algorithm=[Security.Cryptography.SHA256]::Create()
  try{$hash=[BitConverter]::ToString($algorithm.ComputeHash($bytes)).Replace('-','')}finally{$algorithm.Dispose()}
  $preferences+=@{name=$name;sha256=$hash}
 }
}finally{$key.Dispose()}}
$snapshot=@{files=$files;nnPreferences=$preferences;nnPreferencesMeasured=$preferencesMeasured}
$path=Join-Path $study 'before-user-data.json'
if($Action -eq 'Capture'){
 if(Test-Path -LiteralPath $path){throw 'Preserve the original user-data snapshot.'}
 $snapshot|ConvertTo-Json -Depth 7|Set-Content -LiteralPath $path -Encoding utf8
 Write-Output ('Captured '+$files.Count+' save/history files; registry preference measurement available: '+$preferencesMeasured);return
}
$before=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
$expected=@($before.files|ForEach-Object{$_.path+'|'+$_.sha256})
$actual=@($files|ForEach-Object{$_.path+'|'+$_.sha256})
if(($expected -join "`n") -ne ($actual -join "`n")){throw 'User save/history files differ from the pre-install snapshot.'}
$expected=@($before.nnPreferences|ForEach-Object{$_.name+'|'+$_.sha256})
$actual=@($preferences|ForEach-Object{$_.name+'|'+$_.sha256})
if($before.nnPreferencesMeasured -and $preferencesMeasured -and ($expected -join "`n") -ne ($actual -join "`n")){throw 'NN menu preferences differ from the pre-install snapshot.'}
@{passed=$true;savesAndHistoryFiles=$files.Count;nnPreferenceValues=$preferences.Count;nnPreferencesMeasured=($before.nnPreferencesMeasured -and $preferencesMeasured);beforeSha256=(Get-FileHash -LiteralPath $path).Hash}|
 ConvertTo-Json|Set-Content -LiteralPath (Join-Path $study 'user-data-verification.json') -Encoding utf8
Write-Output ('User save/history files unchanged; registry preference measurement available: '+($before.nnPreferencesMeasured -and $preferencesMeasured))
