param([Parameter(Mandatory=$true)][ValidateSet('1.4.8','1.4.9','1.4.10','1.4.11','1.4.12','1.4.13','1.4.15','1.4.16','1.4.17','1.4.18','1.4.19','1.4.20','1.4.21','1.4.22','1.4.23','1.4.24','1.4.25','1.4.26','1.4.27','1.4.28')][string]$Version,[switch]$VerifyOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$archive=(Resolve-Path -LiteralPath (Join-Path $root ('artifacts/patch-releases/'+$Version))).Path
$prefix=$archive.TrimEnd([char[]]@('\','/'))+[IO.Path]::DirectorySeparatorChar
$inventory=Read-BrassJson (Join-Path $archive 'archive-manifest.json')
if(-not $inventory -or $inventory.version -ne $Version){throw 'Archive inventory/version is missing.'}
foreach($entry in $inventory.files){
 $path=[IO.Path]::GetFullPath((Join-Path $archive $entry.path))
 if(-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive member escapes its directory.'}
 if((Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw ('Archive member changed: '+$entry.path)}
}
$manifest=Read-BrassJson (Join-Path $archive 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $archive 'validation.json')
if($manifest.version -ne $Version -or -not $validation.passed){throw 'Qualified archive/version is required.'}
$names=@($manifest.files|ForEach-Object name|Sort-Object)
if(($names -join '|') -ne 'Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'){throw 'Unexpected archive binary list.'}
$sources=@{}
foreach($file in $manifest.files){
 $relative=if(Test-Path -LiteralPath (Join-Path $archive ('Managed/'+$file.name))){'Managed/'+$file.name}else{$file.name}
 if(@($inventory.files|Where-Object {$_.path -eq $relative -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Binary is not in the qualified inventory.'}
 if(@($validation.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Validation fingerprint differs.'}
 $sources[$file.name]=Join-Path $archive $relative
}
if($VerifyOnly){Write-Output ('Verified '+$Version+' archive: '+$inventory.files.Count+' files. Live game and staging unchanged.');return}
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before changing its staging package.'}
$build=Join-Path $root 'artifacts/patch-build'
New-Item -ItemType Directory -Path (Join-Path $build 'Managed') -Force|Out-Null
foreach($file in $manifest.files){Copy-BrassAtomic $sources[$file.name] (Join-Path $build ('Managed/'+$file.name))}
foreach($name in 'build-manifest.json','validation.json','local-ai-qualification.json'){
 Copy-BrassAtomic (Join-Path $archive $name) (Join-Path $build $name)
}
Assert-BrassCandidate $manifest $validation $build
Write-Output ('Staged qualified '+$Version+'. Run scripts/install-patch.ps1 to change the live installation.')

