param([string]$BaseVersion='1.4.12',[string]$Version='1.4.13')
$ErrorActionPreference='Stop'
$trackerRoot=Split-Path -Parent $PSScriptRoot
if(-not $BaseVersion){$BaseVersion=(Get-Content -LiteralPath (Join-Path $trackerRoot 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json).version}
$trackerArchive=Join-Path $trackerRoot ('artifacts/patch-releases/'+$BaseVersion)
$trackerOut=Join-Path $trackerRoot 'artifacts/card-tracker-build'
$trackerSnapshot=Join-Path $trackerOut ('baseline/'+$BaseVersion)
if(-not (Test-Path -LiteralPath (Join-Path $trackerArchive 'archive-manifest.json')) -and (Test-Path -LiteralPath (Join-Path $trackerSnapshot 'archive-manifest.json'))){$trackerArchive=$trackerSnapshot}
if(-not (Test-Path -LiteralPath (Join-Path $trackerArchive 'archive-manifest.json'))){
 # A newly installed, fully qualified release may not yet have its long-term
 # archive. Take our own verified snapshot without changing shared staging.
 $trackerLive=Get-Content -LiteralPath (Join-Path $trackerRoot 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
 $trackerShared=Join-Path $trackerRoot 'artifacts/patch-build'
 $trackerLiveManifest=Get-Content -LiteralPath (Join-Path $trackerShared 'build-manifest.json') -Raw|ConvertFrom-Json
 $trackerLiveValidation=Get-Content -LiteralPath (Join-Path $trackerShared 'validation.json') -Raw|ConvertFrom-Json
 if($trackerLive.version -ne $BaseVersion -or $trackerLiveManifest.version -ne $BaseVersion -or -not $trackerLiveValidation.passed -or -not $trackerLiveValidation.localAi.passed){throw 'Current qualified baseline is unavailable.'}
 $trackerArchive=Join-Path $trackerOut ('baseline/'+$BaseVersion)
 New-Item -ItemType Directory -Path (Join-Path $trackerArchive 'Managed') -Force|Out-Null
 foreach($file in $trackerLiveManifest.files){
  if(@($trackerLive.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or @($trackerLiveValidation.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/staged qualification fingerprints differ.'}
  $source=Join-Path $trackerRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
  if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed baseline changed.'}
  Copy-Item -LiteralPath $source -Destination (Join-Path $trackerArchive ('Managed/'+$file.name)) -Force
 }
 foreach($pin in $trackerLiveManifest.sourcePins){
  if($pin.path -notlike 'src/*' -or $pin.path.Contains('..')){throw 'Unexpected source path.'}
  $source=Join-Path $trackerRoot $pin.path
  if(-not (Test-Path -LiteralPath $source) -or (Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){
   # Our one input-composition edit is absent from the baseline. Its original
   # is retained in the prior verified archive, with the same qualified hash.
   $source=Join-Path $trackerRoot ('artifacts/patch-releases/1.4.10/'+$pin.path)
   if(-not (Test-Path -LiteralPath $source) -or (Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw ('Qualified baseline source unavailable: '+$pin.path)}
  }
  $target=Join-Path $trackerArchive $pin.path
  New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Force
 }
 foreach($name in 'build-manifest.json','validation.json','local-ai-qualification.json'){Copy-Item -LiteralPath (Join-Path $trackerShared $name) -Destination $trackerArchive -Force}
 $trackerSnapshotFiles=@(Get-ChildItem -LiteralPath $trackerArchive -Recurse -File|Where-Object Name -ne 'archive-manifest.json'|ForEach-Object {[ordered]@{path=[IO.Path]::GetRelativePath($trackerArchive,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
 @{version=$BaseVersion;files=$trackerSnapshotFiles}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $trackerArchive 'archive-manifest.json') -Encoding utf8
}
$trackerSource=Join-Path $trackerOut 'runtime'
$trackerInventory=Get-Content -LiteralPath (Join-Path $trackerArchive 'archive-manifest.json') -Raw|ConvertFrom-Json
$trackerBase=Get-Content -LiteralPath (Join-Path $trackerArchive 'build-manifest.json') -Raw|ConvertFrom-Json
$trackerQualified=Get-Content -LiteralPath (Join-Path $trackerArchive 'validation.json') -Raw|ConvertFrom-Json
if(-not $trackerQualified.passed -or $trackerBase.version -ne $BaseVersion){throw 'A qualified archived runtime is required.'}
foreach($entry in $trackerInventory.files){
 $target=[IO.Path]::GetFullPath((Join-Path $trackerArchive $entry.path))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($trackerArchive)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its root.'}
 if((Get-FileHash -LiteralPath $target).Hash -ne $entry.sha256){throw ('Archive changed: '+$entry.path)}
}
New-Item -ItemType Directory -Path $trackerOut,$trackerSource,(Join-Path $trackerOut 'Managed') -Force|Out-Null
# Build in dedicated staging. Other AI work's sources and patch-build are untouched.
foreach($entry in $trackerInventory.files|Where-Object {$_.path -like 'src/*'}){
 $target=Join-Path $trackerSource $entry.path
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $trackerArchive $entry.path) -Destination $target -Force
}
Copy-Item -LiteralPath (Join-Path $trackerRoot 'src/BrassPatch/CardTracker') -Destination (Join-Path $trackerSource 'src/BrassPatch') -Recurse -Force
$trackerEntry=Join-Path $trackerSource 'src/BrassPatch/Statistics/StatisticsEntryPoints.cs'
$trackerText=[IO.File]::ReadAllText($trackerEntry).Replace([string][char]13,'')
$trackerNeedle=@'
public static void ProcessInput(Router router)
        {
'@
if(-not $trackerText.Contains($trackerNeedle)){throw 'Archived statistics input signature changed; review before building.'}
if(-not $trackerText.Contains('if (BrassPatch.CardTracker.CardTrackerEntryPoints.ProcessInput(router)) return;')){
 $trackerText=$trackerText.Replace($trackerNeedle,$trackerNeedle+[Environment]::NewLine+'            if (BrassPatch.CardTracker.CardTrackerEntryPoints.ProcessInput(router)) return;')
 [IO.File]::WriteAllText($trackerEntry,$trackerText)
}
$trackerTools=Join-Path $trackerRoot 'artifacts/patch-tools'
$trackerCompiler=Join-Path $trackerTools 'microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$trackerCecil=Join-Path $trackerTools 'mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
$trackerManaged=Join-Path $trackerRoot 'BrassBirminghamUnity_Data/Managed'
$trackerStock=Join-Path $trackerRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $trackerStock).Hash -ne $trackerBase.stockSha256){throw 'Stock fingerprint differs.'}
$trackerHelper=Join-Path $trackerOut 'Managed/BrassPatch.dll'
$trackerReferences=@(Get-ChildItem -LiteralPath $trackerManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})
$trackerReferences+='/reference:"'+$trackerStock+'"'
$trackerSources=@(Get-ChildItem -LiteralPath (Join-Path $trackerSource 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object {'"'+$_.FullName+'"'})
$trackerRsp=Join-Path $trackerOut 'helper.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+$trackerHelper+'"'))+$trackerReferences+$trackerSources|Set-Content -LiteralPath $trackerRsp -Encoding utf8
& $trackerCompiler ('@'+$trackerRsp)
if($LASTEXITCODE -ne 0){throw 'Card Tracker helper compilation failed.'}
$trackerWeaver=Join-Path $trackerOut 'BrassPatch.Weaver.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$trackerWeaver+'"'),('/reference:"'+$trackerCecil+'"'),('/reference:"'+(Join-Path $trackerManaged 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $trackerRoot 'src/BrassPatch.Weaver/Program.cs')+'"'))|Set-Content -LiteralPath (Join-Path $trackerOut 'weaver.rsp') -Encoding utf8
& $trackerCompiler ('@'+(Join-Path $trackerOut 'weaver.rsp'))
if($LASTEXITCODE -ne 0){throw 'Weaver compilation failed.'}
Copy-Item -LiteralPath $trackerCecil -Destination (Join-Path $trackerOut 'Mono.Cecil.dll') -Force
Copy-Item -LiteralPath (Join-Path $trackerManaged 'Newtonsoft.Json.dll') -Destination (Join-Path $trackerOut 'Newtonsoft.Json.dll') -Force
Get-ChildItem -LiteralPath $trackerManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|Copy-Item -Destination (Join-Path $trackerOut 'Managed') -Force
$trackerAssembly=Join-Path $trackerOut 'Managed/Assembly-CSharp.dll'
& $trackerWeaver $trackerStock $trackerHelper (Join-Path $trackerSource 'src/BrassPatch') $trackerAssembly
if($LASTEXITCODE -ne 0){throw 'Tracker weaving failed.'}
$trackerAi=Join-Path $trackerOut 'Managed/BrassLegacyAI.dll'
Copy-Item -LiteralPath (Join-Path $trackerArchive 'Managed/BrassLegacyAI.dll') -Destination $trackerAi -Force
$trackerOverlay=Join-Path $trackerOut 'Managed/Assembly-CSharp.with-ai.dll'
& $trackerWeaver '--overlay' $trackerAssembly $trackerAi (Join-Path $trackerSource 'src/BrassLocalAI/Unity') $trackerOverlay ((Get-FileHash -LiteralPath $trackerAssembly).Hash)
if($LASTEXITCODE -ne 0){throw 'Preserved AI weaving failed.'}
Copy-Item -LiteralPath $trackerOverlay -Destination $trackerAssembly -Force
Copy-Item -LiteralPath ($trackerOverlay+'.manifest.json') -Destination ($trackerAssembly+'.manifest.json') -Force
$trackerPinnedPaths=@($trackerBase.sourcePins|ForEach-Object path)+@(Get-ChildItem -LiteralPath (Join-Path $trackerSource 'src/BrassPatch/CardTracker') -Recurse -File|Where-Object Extension -in '.cs','.json'|ForEach-Object {[IO.Path]::GetRelativePath($trackerSource,$_.FullName).Replace('\','/')})
$trackerPins=@($trackerPinnedPaths|Sort-Object -Unique|ForEach-Object {[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $trackerSource $_)).Hash}})
$trackerManifest=[ordered]@{version=$Version;stockSha256=$trackerBase.stockSha256;baseVersion=$BaseVersion;baseFiles=$trackerBase.files;sourcePins=$trackerPins;files=@(Get-Item $trackerHelper,$trackerAi,$trackerAssembly|ForEach-Object {[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}
$trackerManifest|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $trackerOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built Card Tracker '+$Version+' from qualified '+$BaseVersion+' in dedicated staging. AI DLL preserved; game installation unchanged.')
