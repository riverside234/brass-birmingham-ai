param([ValidateSet(2,3,4)][int]$Players=4,[switch]$BuildMatrix)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$candidate=Join-Path $root 'artifacts/patch-build/Managed'
$workspace=Join-Path $root 'artifacts/patch-audit'
$managed=Join-Path $workspace 'BrassBirminghamUnity_Data/Managed'
$scripts=Join-Path $workspace 'rules-audit'
New-Item -ItemType Directory -Force -Path $managed,$scripts|Out-Null
# Each caller uses the identical candidate. Prepare first, then run player counts sequentially or in separate workspaces.
foreach($file in Get-ChildItem -LiteralPath $candidate -Filter '*.dll'){
    $destination=Join-Path $managed $file.Name
    if(-not(Test-Path -LiteralPath $destination) -or (Get-FileHash -LiteralPath $destination).Hash -ne (Get-FileHash -LiteralPath $file.FullName).Hash){
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}
$hash=(Get-FileHash -LiteralPath (Join-Path $candidate 'Assembly-CSharp.dll')).Hash
$name=if($BuildMatrix){'extended-build-probes.ps1'}else{'extended-ai-probes.ps1'}
$source=Get-Content -LiteralPath (Join-Path $root ('rules-audit/'+$name)) -Raw
$source=$source.Replace('600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB',$hash)
$path=Join-Path $scripts $(if($BuildMatrix){$name}else{"extended-ai-$Players-probes.ps1"})
$source|Set-Content -LiteralPath $path -Encoding UTF8
if($BuildMatrix){
    & pwsh -NoProfile -File $path
    if($LASTEXITCODE -ne 0){throw 'Build matrix process failed'}
    $result=Get-Content -LiteralPath (Join-Path $scripts 'extended-build-results.json') -Raw|ConvertFrom-Json
    if($result.failures.Count -gt 0 -or @($result.aiEdges|Where-Object exception).Count -gt 0){throw 'Patched build/AI edge regression failed'}
}else{
    & pwsh -NoProfile -File $path -Players $Players
    if($LASTEXITCODE -ne 0){throw 'AI suite process failed'}
    $result=Get-Content -LiteralPath (Join-Path $scripts "extended-ai-$Players-results.json") -Raw|ConvertFrom-Json
    if(@($result.runs|Where-Object {$_.error -or -not $_.complete}).Count -gt 0){throw 'Patched AI game failed'}
}
