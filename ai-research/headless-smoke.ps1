param(
    [ValidateSet(2, 3, 4)][int]$Players = 2,
    [int]$Seed = 20260927,
    [ValidateRange(1, 100)][int]$Games = 1,
    [ValidateSet('Hard', 'Medium', 'Easy')][string]$Difficulty = 'Hard',
    [string]$OutputPath,
    [string]$GameRoot = $env:BRASS_GAME_ROOT
)

# Calls the installed game's rules and original AI in a separate PowerShell process.
# No Unity launch, plugin installation, game save access, or game file modification.
# Keep games sequential: the original engine and AI contain static mutable state.
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($GameRoot)) {
    $GameRoot = Split-Path -Parent $PSScriptRoot
}
$assemblyPath = Join-Path $GameRoot 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
if (-not (Test-Path -LiteralPath $assemblyPath -PathType Leaf)) {
    throw 'Game assembly not found. Set -GameRoot or BRASS_GAME_ROOT to your local Brass Birmingham installation; game files are excluded from this repository.'
}
$expectedHash = '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'
$actualHash = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) {
    throw 'The game assembly changed. Recheck the API before using this probe.'
}
[void][System.Reflection.Assembly]::LoadFrom($assemblyPath)
$results = [System.Collections.Generic.List[object]]::new()

for ($gameIndex = 0; $gameIndex -lt $Games; $gameIndex++) {
    $gameSeed = $Seed + $gameIndex
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new(
        [BoardGameRules.Entities.Common.NumberOfPlayers]$Players)
    $config.Seed = $gameSeed
    $metadata = [BrassApplication.Game.GameMetadata]::new(
        $config, [BrassApplication.Game.GameType]::Local)
    $metadata.GameId = 'headless-review-' + $gameSeed
    foreach ($playerConfig in $config.Players) {
        $playerMetadata = [BrassApplication.Game.PlayerMetadata]::new()
        $playerMetadata.PlayerNetworkId = 'offline-' + $playerConfig.Color
        $playerMetadata.NickName = 'Probe ' + $playerConfig.Color
        $playerMetadata.PlayerStartConfiguration = $playerConfig
        $playerMetadata.AiType = [BrassApplication.AI.AIType]::$Difficulty
        $playerMetadata.AIBehaviour = [BrassApplication.AI.AiBehaviors.AiBehaviorFactory]::CreateAiBehavior($playerMetadata)
        $metadata.JoinGame($playerMetadata)
    }
    $rules = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $applicationGame = [BrassApplication.Game.BrassApplicationGame]::new($rules, $metadata)
    $rules.StartGame()
    $turns = 0
    $actions = 0
    $debtPhases = 0
    $actionCounts = @{}
    $decisionMilliseconds = [System.Collections.Generic.List[double]]::new()
    $debtSelector = [BrassApplication.AI.RuleBasedSystem.RuleSelectors.RuleSelectorFactory]::CreatePrioritySelector(
        [BrassApplication.AI.RuleBasedSystem.RuleSetFactory]::CreateSellIndustryRuleSet())

    while (-not $rules.GameProgressInformation.IsGameOver) {
        if ($turns -ge 200 -or $actions -ge 400) {
            throw "Progress guard reached for seed $gameSeed."
        }
        $current = $rules.GameProgressInformation.CurrentPlayer
        $behavior = $metadata.GetPlayerMetadataByColor($current.Color).AIBehaviour
        while ($rules.TurnFlow.GetRemainingActionsCount() -gt 0) {
            $decisionWatch = [System.Diagnostics.Stopwatch]::StartNew()
            $move = $behavior.GetNextAIMove($applicationGame)
            if ($null -eq $move) { throw 'AI returned no move.' }
            $move.Invoke()
            $decisionWatch.Stop()
            $decisionMilliseconds.Add($decisionWatch.Elapsed.TotalMilliseconds)
            $actionName = if ($null -ne $rules.TurnFlow.CurrentAction) {
                $rules.TurnFlow.CurrentAction.GetType().Name
            } else { 'NoCurrentAction' }
            if (-not $actionCounts.ContainsKey($actionName)) { $actionCounts[$actionName] = 0 }
            $actionCounts[$actionName]++
            $rules.TurnFlow.ConfirmCurrentAction()
            $actions++
        }
        $rules.GameFlow.EndTurn()
        $turns++
        if ($rules.GameFlow.IsLastTurnOver()) {
            if ($rules.GameFlow.IsSellIndustryNecessary()) {
                $debtPhases++
                $debtRule = $debtSelector.Select($applicationGame)
                $debtRule.MakeAction($applicationGame).Invoke()
            }
            $rules.GameFlow.EndRound()
        }
    }
    $watch.Stop()
    $scores = @($rules.PlayersTrack.Players | ForEach-Object {
        [ordered]@{ color = $_.Color.ToString(); vp = $_.VP; money = $_.Money; income = $_.IncomeLevel }
    })
    $sortedDecisions = @($decisionMilliseconds | Sort-Object)
    $result = [ordered]@{
        seed = $gameSeed
        players = $Players
        difficulty = $Difficulty
        gameOver = $rules.GameProgressInformation.IsGameOver
        finalEra = $rules.GameProgressInformation.CurrentGameEra.ToString()
        turns = $turns
        actions = $actions
        actionCounts = $actionCounts
        debtPhases = $debtPhases
        revision = $rules.Revision
        elapsedMilliseconds = [math]::Round($watch.Elapsed.TotalMilliseconds, 3)
        decisionMeanMilliseconds = [math]::Round(($decisionMilliseconds | Measure-Object -Average).Average, 3)
        decisionP95Milliseconds = [math]::Round($sortedDecisions[[math]::Ceiling(0.95 * $sortedDecisions.Count) - 1], 3)
        ranking = @($rules.GameProgressInformation.EndGameRanking | ForEach-Object { $_.ToString() })
        scores = $scores
    }
    $results.Add($result)
}

$report = [ordered]@{
    purpose = 'Feasibility smoke test using the original installed rules and original AI, not a replacement AI or training benchmark.'
    assemblySha256 = $actualHash
    runtime = [System.Runtime.InteropServices.RuntimeInformation]::FrameworkDescription
    powershell = $PSVersionTable.PSVersion.ToString()
    utc = [DateTime]::UtcNow.ToString('o')
    results = @($results.ToArray())
}
$json = ConvertTo-Json -InputObject $report -Depth 12
if ($OutputPath) { $json | Set-Content -LiteralPath $OutputPath -Encoding utf8 }
$json
