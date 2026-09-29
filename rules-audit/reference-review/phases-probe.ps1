$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$managed = Join-Path $workspace 'BrassBirminghamUnity_Data/Managed'
$assembly = Join-Path $managed 'Assembly-CSharp.dll'
$helper = Join-Path $managed 'BrassPatch.dll'
if ((Get-FileHash -LiteralPath $assembly).Hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A') { throw 'Unexpected installed game build' }
if ((Get-FileHash -LiteralPath $helper).Hash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D') { throw 'Unexpected installed patch build' }
[void][Reflection.Assembly]::LoadFrom($assembly)
[void][Reflection.Assembly]::LoadFrom($helper)
Add-Type -Path (Join-Path $PSScriptRoot 'phases-probe.cs') -ReferencedAssemblies @($assembly,$helper,'System.Core.dll')
$result = [ordered]@{ assemblySha256=(Get-FileHash -LiteralPath $assembly).Hash; helperSha256=(Get-FileHash -LiteralPath $helper).Hash; runtime=[Environment]::Version.ToString(); scoutCases=[ReferencePhasesProbe]::Run((Join-Path $workspace 'artifacts/reference-review/phases-scout')) }
$json = ConvertTo-Json $result -Depth 12
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'phases-results.json') -Encoding UTF8
$json
