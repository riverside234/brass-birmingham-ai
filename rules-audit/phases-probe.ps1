$ErrorActionPreference = 'Stop'
$gameRoot = Split-Path -Parent $PSScriptRoot
$assemblyPath = Join-Path $gameRoot 'BrassBirminghamUnity_Data\Managed\Assembly-CSharp.dll'
$expectedHash = '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'
$actualHash = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) { throw 'Assembly version mismatch' }
[void][System.Reflection.Assembly]::LoadFrom($assemblyPath)
function New-ProbeGame {
    param([int]$Players = 2)
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$Players)
    $config.Seed = 20260927
    $game = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $game.StartGame()
    return $game
}
function End-ProbeGame($game) {
    $zeros = [System.Collections.Generic.Dictionary[BoardGameRules.Entities.Players.PlayerColor,int]]::new()
    foreach ($player in $game.PlayersTrack.Players) { $zeros.Add($player.Color, 0) }
    $end = $game.GameFlow.GetType().GetField('_endOfGame', [System.Reflection.BindingFlags]'NonPublic,Instance').GetValue($game.GameFlow)
    $end.EndGame($zeros, $zeros, $zeros)
}
$loans = @()
foreach ($income in @(-8,-7,-6)) {
    $game = New-ProbeGame
    $player = $game.GameProgressInformation.CurrentPlayer
    $player.IncomeMarkerPosition = $income + 10
    $beforeMoney = $player.Money
    $reasons = @($game.TurnFlow.LoanAction.GetLoanActionNotPossibleReasons($game.GameProgressInformation) | ForEach-Object { $_.ToString() })
    $errorMessage = $null
    try { $game.TurnFlow.LoanAction.Loan($player.Color) } catch { $errorMessage = $_.Exception.InnerException.Message }
    $loans += [ordered]@{ startingIncome = $income; uiReasons = $reasons; executionError = $errorMessage; endingIncome = $player.IncomeLevel; moneyGained = $player.Money - $beforeMoney }
}
$game = New-ProbeGame
$players = $game.PlayersTrack.Players
$players[0].VP = 100; $players[0].IncomeMarkerPosition = 12; $players[0].Money = 0
$players[1].VP = 100; $players[1].IncomeMarkerPosition = 11; $players[1].Money = 10
$expectedWinner = $players[1].Color.ToString()
End-ProbeGame $game
$incomeTie = [ordered]@{ expectedWinnerByIncomeLevelThenMoney = $expectedWinner; actualWinner = $game.GameProgressInformation.EndGameRanking[0].ToString(); players = @($players | ForEach-Object { [ordered]@{ color = $_.Color.ToString(); vp = $_.VP; incomeMarker = $_.IncomeMarkerPosition; incomeLevel = $_.IncomeLevel; money = $_.Money } }) }
$game = New-ProbeGame
foreach ($player in $game.PlayersTrack.Players) { $player.VP = 100; $player.IncomeMarkerPosition = 12; $player.Money = 10 }
End-ProbeGame $game
$exactTie = @($game.PlayersTrack.Players | ForEach-Object { [ordered]@{ color = $_.Color.ToString(); assignedPlace = $game.GameProgressInformation.EndGamePoints[$_.Color].Item5 } })
$game = New-ProbeGame
$progress = $game.GameProgressInformation
$progress.CurrentGameEra = [BoardGameRules.Entities.GamePhases.GameEra]::RailEra
$progress.CurrentRoundNumber = 10
$progress.CurrentTurnNumber = 2
$player = $progress.CurrentPlayer
$player.Money = 0; $player.IncomeMarkerPosition = 9; $player.VP = 10
$game.TurnFlow.DecreaseRemainingActionsCount()
$game.GameFlow.EndTurn()
$finalDebtStarted = $game.GameFlow.IsSellIndustryNecessary()
$vpBefore = $player.VP
$game.GameFlow.EndRound()
$finalRound = [ordered]@{ debtPhaseStarted = $finalDebtStarted; vpBefore = $vpBefore; vpAfter = $player.VP; moneyAfter = $player.Money; gameOver = $progress.IsGameOver }
$cardCoverage = @()
foreach ($count in @(2,3,4)) {
    foreach ($useScouts in @($false,$true)) {
        $game = New-ProbeGame $count
        $turns = 0; $actions = 0; $scouts = 0; $eraEndHands = @()
        while (-not $game.GameProgressInformation.IsGameOver) {
            if ($turns -gt 100) { throw 'Unexpected turn count' }
            $player = $game.GameProgressInformation.CurrentPlayer
            while ($game.TurnFlow.GetRemainingActionsCount() -gt 0) {
                if ($player.Hand.Count -eq 0) { throw 'No card for required action' }
                if ($useScouts -and $game.TurnFlow.ScoutAction.GetScoutActionNotPossibleReasons($game.Board, $game.GameProgressInformation).Count -eq 0) {
                    $action = $game.TurnFlow.StartScoutAction()
                    $action.DiscardCard($player.Hand[0], $player)
                    $trade = [System.Collections.Generic.List[BoardGameRules.Entities.Cards.BaseCard]]::new()
                    $trade.Add($player.Hand[0]); $trade.Add($player.Hand[1])
                    $action.Scout($trade, $player)
                    $scouts++
                } else {
                    $action = $game.TurnFlow.StartPassAction()
                    $action.Pass($player.Hand[0], $player)
                }
                $game.TurnFlow.ConfirmCurrentAction()
                $actions++
            }
            $game.GameFlow.EndTurn()
            $turns++
            if ($game.GameFlow.IsLastTurnOver()) {
                if ($game.GameProgressInformation.CurrentRoundNumber -eq $game.GameFlow.GetNumberOfRounds($game.GameStartConfiguration.NumberOfPlayers)) {
                    $eraEndHands += [ordered]@{ era = $game.GameProgressInformation.CurrentGameEra.ToString(); deck = $game.Board.CardDeck.Count; cardsInHands = @($game.PlayersTrack.Players | ForEach-Object { $_.Hand.Count }) }
                }
                $game.GameFlow.EndRound()
            }
        }
        $cardCoverage += [ordered]@{ players = $count; scoutsEnabled = $useScouts; turns = $turns; actions = $actions; scouts = $scouts; eraEnds = $eraEndHands; wildIndustrySupply = $game.Board.WildIndustryCardDeck.Count; wildLocationSupply = $game.Board.WildLocationCardDeck.Count }
    }
}
$result = [ordered]@{ description = 'Targeted state fixtures on the original installed DLL; no production files or saves modified. Loan execution calls its public method directly for gate comparison; end-round uses normal EndTurn and EndRound flow on final-round fixture. Card coverage executes complete legal pass/scout games for two, three, and four players.'; assemblySha256 = $actualHash; loans = $loans; incomeTie = $incomeTie; exactTie = $exactTie; finalRailRound = $finalRound; cardCoverage = $cardCoverage }
$json = ConvertTo-Json -InputObject $result -Depth 10
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'phases-probe-results.json') -Encoding utf8
$json
