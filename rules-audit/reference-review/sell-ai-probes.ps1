$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$managed=Join-Path $root 'BrassBirminghamUnity_Data/Managed'
$assemblyPath=Join-Path $managed 'Assembly-CSharp.dll'
$assemblyHash='34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A'
$helperHash='4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D'
if((Get-FileHash $assemblyPath).Hash -ne $assemblyHash -or (Get-FileHash (Join-Path $managed 'BrassPatch.dll')).Hash -ne $helperHash){throw 'Unexpected installed patch'}
[void][Reflection.Assembly]::LoadFrom($assemblyPath)
$failures=[Collections.Generic.List[object]]::new()
$types=@([Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])|Where-Object {$_.ToString() -ne 'Undefined'})
$count=0
foreach($players in 2,3,4){
 foreach($era in [BoardGameRules.Entities.GamePhases.GameEra]::CanalEra,[BoardGameRules.Entities.GamePhases.GameEra]::RailEra){
  foreach($difficulty in 'Easy','Medium','Hard'){
   for($mask=0;$mask -lt 64;$mask++){
    $config=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$players)
    $config.Seed=20260928
    $game=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config);$game.StartGame()
    $player=$game.GameProgressInformation.CurrentPlayer
    $game.GameProgressInformation.CurrentGameEra=$era;$game.GameProgressInformation.CurrentRoundNumber=2
    $game.TurnFlow.StartTurn($player,2,$era)
    for($bit=0;$bit -lt 6;$bit++){if($mask -band (1 -shl $bit)){$player.Mat.GetIndustryList($types[$bit]).Clear()}}
    $metadata=[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local)
    $app=[BrassApplication.Game.BrassApplicationGame]::new($game,$metadata)
    $ai=switch($difficulty){
     'Easy'{[BrassApplication.AI.AiBehaviors.EasyAiBehavior]::new('review')}
     'Medium'{[BrassApplication.AI.AiBehaviors.NormalAiBehavior]::new('review')}
     'Hard'{[BrassApplication.AI.AiBehaviors.HardAiBehavior]::new('review')}
    }
    $count++
    try{
     $revision=$game.Revision;$hand=$player.Hand.Count
     $move=$ai.GetNextAIMove($app)
     if($game.Revision -ne $revision){throw 'Decision query appended an event'}
     $move.Invoke()
     if($player.Hand.Count -ne ($hand-1)){throw 'Chosen move did not discard exactly one card'}
     if($player.Money -lt 0){throw 'Chosen move made cash negative'}
     $game.TurnFlow.ConfirmCurrentAction()
     if($game.TurnFlow.GetRemainingActionsCount() -ne 1){throw 'Chosen move did not consume exactly one action'}
    }catch{$failures.Add([ordered]@{players=$players;era=$era.ToString();difficulty=$difficulty;emptyMatMask=$mask;error=$_.Exception.ToString()})}
   }
  }
 }
}
$report=[ordered]@{assemblySha256=$assemblyHash;helperSha256=$helperHash;fixtures=$count;failures=@($failures.ToArray());limitation='Synthetic empty-stack combinations on a sparse board; not a proof that every combination is reachable in a legal played game. Decision and chosen action executed, no persistence or user saves.'}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'sell-ai-results.json') -Encoding UTF8
[ordered]@{fixtures=$count;failureCount=$failures.Count;firstFailures=@($failures|Select-Object -First 3)}|ConvertTo-Json -Depth 8
