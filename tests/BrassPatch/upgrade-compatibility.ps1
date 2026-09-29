param([switch]$CreateBaseline,[string]$ManagedPath)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if(-not $ManagedPath){$ManagedPath=Join-Path $root 'artifacts/patch-build/Managed'}
$artifact=Join-Path $root 'artifacts/patch-upgrade-tests'
New-Item -ItemType Directory -Force -Path $artifact | Out-Null
$expectedPath=Join-Path $artifact 'v1-baseline.json'
$gameDll=Join-Path $ManagedPath 'Assembly-CSharp.dll'
if($CreateBaseline){
 if(Test-Path -LiteralPath $expectedPath){throw 'The previous-release fixture already exists; preserve it.'}
 if((Get-FileHash -LiteralPath $gameDll).Hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A'){throw 'Baseline requires the verified 1.0.0 game'}
 if((Get-FileHash -LiteralPath (Join-Path $ManagedPath 'BrassPatch.dll')).Hash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D'){throw 'Baseline requires the verified 1.0.0 helper'}
}
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'Newtonsoft.Json.dll'))
[void][Reflection.Assembly]::LoadFrom($gameDll)
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'BrassPatch.dll'))
function New-Repository([string]$path){
 $serializer=[Frameworks.Persistence.Serialization.JsonSerializer]::new()
 [Frameworks.Persistence.Repository.BaseRepository]::new([Frameworks.Persistence.RepositoryBackends.Filesystem.FilesystemGameStorage]::new($path,$serializer),[Frameworks.Persistence.RepositoryBackends.Filesystem.FilesystemSnapshotStorage]::new($path,$serializer),[Frameworks.Persistence.GameFactory.BrassBirminghamFactory]::new())
}
function Snapshot($app){
 $g=$app.Game
 $players=@($g.PlayersTrack.Players | ForEach-Object {
  $p=$_
  [ordered]@{color=$p.Color.ToString();money=$p.Money;income=$p.IncomeMarkerPosition;vp=$p.VP;linkTiles=$p.LinkTileCount;
   hand=@($p.Hand|ForEach-Object{$_.UniqueId});discard=@($p.DiscardPile|ForEach-Object{$_.UniqueId});
   mat=@([Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])|Where-Object {$_.ToString() -ne 'Undefined'}|ForEach-Object{$p.Mat.GetIndustryList($_)}|ForEach-Object{$_.UniqueId})}
 })
 [ordered]@{revision=$g.Revision;round=$g.GameProgressInformation.CurrentRoundNumber;era=$g.GameProgressInformation.CurrentGameEra.ToString();
  currentPlayer=$g.GameProgressInformation.CurrentPlayer.Color.ToString();actions=$g.TurnFlow.GetRemainingActionsCount();players=$players;
  deck=@($g.Board.CardDeck|ForEach-Object{$_.UniqueId});wildIndustry=@($g.Board.WildIndustryCardDeck|ForEach-Object{$_.UniqueId});wildLocation=@($g.Board.WildLocationCardDeck|ForEach-Object{$_.UniqueId})} | ConvertTo-Json -Depth 8 -Compress
}
if($CreateBaseline){
 $baseline=Join-Path $artifact ('v1-saves-'+[Guid]::NewGuid().ToString('N'))
 $repository=New-Repository $baseline
 $rows=@()
 foreach($count in 2,3,4){
  $config=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$count);$config.Seed=20260929
  $metadata=[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local)
  foreach($p in $config.Players){$pm=[BrassApplication.Game.PlayerMetadata]::new();$pm.PlayerStartConfiguration=$p;$pm.PlayerNetworkId='upgrade-'+$p.Color;$pm.NickName='Upgrade fixture';$pm.AiType=[BrassApplication.AI.AIType]::Human;$metadata.JoinGame($pm)}
  $app=$repository.Create($metadata);$app.Game.StartGame();$p=$app.Game.GameProgressInformation.CurrentPlayer
  $action=$app.Game.TurnFlow.StartScoutAction();$action.DiscardCard($p.Hand[0],$p)
  $trade=[Collections.Generic.List[BoardGameRules.Entities.Cards.BaseCard]]::new();$trade.Add($p.Hand[0]);$trade.Add($p.Hand[1]);$action.Scout($trade,$p)
  $app.Game.TurnFlow.ConfirmCurrentAction();$app.Game.GameFlow.EndTurn();$repository.Save($app)
  $loaded=$repository.Load($app.Metadata.GameId)
  $rows += [ordered]@{players=$count;gameId=$app.Metadata.GameId;snapshot=(Snapshot $loaded)}
 }
 [ordered]@{version='1.0.0';path=$baseline;games=$rows;files=@(Get-ChildItem -LiteralPath $baseline -Recurse -File|ForEach-Object{[ordered]@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $expectedPath -Encoding UTF8
 'Created three artificial saves with the verified previous release.'
 return
}
$expected=Get-Content -LiteralPath $expectedPath -Raw|ConvertFrom-Json
$working=Join-Path $artifact ('candidate-saves-'+[Guid]::NewGuid().ToString('N'))
Copy-Item -LiteralPath $expected.path -Destination $working -Recurse
$repository=New-Repository $working
$checks=0
foreach($row in $expected.games){
 $app=$repository.Load($row.gameId)
 if((Snapshot $app) -cne $row.snapshot){throw ('Previous-release state changed for '+$row.players+' players')};$checks++
 $p=$app.Game.GameProgressInformation.CurrentPlayer;$before=$p.Hand.Count;$actions=$app.Game.TurnFlow.GetRemainingActionsCount()
 $app.Game.TurnFlow.StartPassAction().Pass($p.Hand[0],$p);$app.Game.TurnFlow.ConfirmCurrentAction()
 if($p.Hand.Count -ne $before-1 -or $app.Game.TurnFlow.GetRemainingActionsCount() -ne $actions-1){throw 'Continued game action accounting differs'};$checks++
 $app.Game.GameFlow.EndTurn();$repository.Save($app)
 if((Snapshot ($repository.Load($row.gameId))) -cne (Snapshot $app)){throw 'Continued upgraded game reload differs'};$checks++
}
foreach($file in $expected.files){if((Get-FileHash -LiteralPath $file.path).Hash -ne $file.sha256){throw 'Original previous-release fixture changed'};$checks++}
[ordered]@{passed=$true;assertions=$checks;previousRelease='1.0.0';games=$expected.games.Count;gameSha256=(Get-FileHash -LiteralPath $gameDll).Hash}|ConvertTo-Json|Tee-Object -FilePath (Join-Path $artifact 'upgrade-results.json')
