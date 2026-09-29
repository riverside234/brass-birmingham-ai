param([string]$ManagedPath = (Join-Path $PSScriptRoot '..\..\..\artifacts\patch-build\Managed'), [switch]$CompileOnly)
$ErrorActionPreference = 'Stop'
$ManagedPath = (Resolve-Path -LiteralPath $ManagedPath).Path
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'Assembly-CSharp.dll'))
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'BrassPatch.dll'))
$references = @((Join-Path $ManagedPath 'Assembly-CSharp.dll'), (Join-Path $ManagedPath 'BrassPatch.dll'), 'System.Core.dll')
Add-Type -Path (Join-Path $PSScriptRoot 'ScoutTests.cs') -ReferencedAssemblies $references
if ($CompileOnly) { Write-Output 'Scout tests compiled.'; return }
$checks = [BrassPatch.Tests.Scout.ScoutTests]::Run()
[ordered]@{ suite = 'Scout'; assertions = $checks; succeeded = $true; runtime = [Environment]::Version.ToString(); managedPath = $ManagedPath } | ConvertTo-Json
