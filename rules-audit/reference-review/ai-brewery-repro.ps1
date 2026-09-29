$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$assemblyPath=Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
if((Get-FileHash $assemblyPath).Hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A'){throw 'Unexpected patch'}
if((Get-FileHash (Join-Path $root 'BrassBirminghamUnity_Data/Managed/BrassPatch.dll')).Hash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D'){throw 'Unexpected helper patch'}
[void][Reflection.Assembly]::LoadFrom($assemblyPath)
$rows=@()
foreach($players in 2,3,4){
 foreach($control in 'empty-brewery-late-canal','one-brewery-left','rail-era','easy','medium'){
  $config=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$players);$config.Seed=20260928
  $g=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config);$g.StartGame();$p=$g.GameProgressInformation.CurrentPlayer
  $brewery=[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Brewery
  $first=$p.Mat.GetFirstBuildingOfType($brewery)
  $p.Mat.GetIndustryList($brewery).Clear()
  if($control -eq 'one-brewery-left'){$p.Mat.GetIndustryList($brewery).Add($first)}
  $era=if($control -eq 'rail-era'){[BoardGameRules.Entities.GamePhases.GameEra]::RailEra}else{[BoardGameRules.Entities.GamePhases.GameEra]::CanalEra}
  $g.GameProgressInformation.CurrentGameEra=$era;$g.GameProgressInformation.CurrentRoundNumber=7
  $g.TurnFlow.StartTurn($p,7,$era);$p.Money=0;$p.IncomeMarkerPosition=1
  $app=[BrassApplication.Game.BrassApplicationGame]::new($g,[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local))
  $ai=switch($control){'easy'{[BrassApplication.AI.AiBehaviors.EasyAiBehavior]::new('probe')};'medium'{[BrassApplication.AI.AiBehaviors.NormalAiBehavior]::new('probe')};default{[BrassApplication.AI.AiBehaviors.HardAiBehavior]::new('probe')}}
  $errorText=$null;$actionCompleted=$false
  try{$move=$ai.GetNextAIMove($app);$move.Invoke();$g.TurnFlow.ConfirmCurrentAction();$actionCompleted=$true}catch{$errorText=$_.Exception.ToString()}
  $rows+=[ordered]@{players=$players;control=$control;round=7;era=$era.ToString();actionCompleted=$actionCompleted;exception=$errorText}
 }
}
$rows|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ai-brewery-results.json') -Encoding UTF8
$rows|ForEach-Object {[pscustomobject]$_}|Select-Object players,control,actionCompleted,@{n='failed';e={$null -ne $_.exception}}|ConvertTo-Json
