using System;
using System.Collections;
using System.IO;
using System.Linq;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BrassLegacyAI;
using BrassLegacyAI.Execution;
using BrassLegacyAI.Native;
using BrassPatch.Persistence;
using Frameworks.View.Game;
using InterfaceAdapters.Common.CreateGame;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    IEnumerator TurnHistoryCases()
    {
        var regression = TurnHistoryRegression.Run(false);
        Check((bool)regression["passed"], "city Cancel/Undo and completed-turn checkpoints pass on shipped Unity Mono");
        router.HideView<ICreateOfflineGameView>();
        yield return SceneManager.LoadSceneAsync("GameScene"); yield return null; router.ReloadPrefabs();
        var view = (PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn = new PlayAiTurnUseCase(new PlayAITurnPresenter(router), history);
        router.UseCases[typeof(PlayAiTurnUseCase)] = turn; view.Controller = new PlayAiTurnController(turn);
        var rows = new JArray();
        foreach (bool native in new[] { true, false })
        {
            var config = new GameStartConfiguration((NumberOfPlayers)4) { Seed = 102602, IsIntroductoryGame = false };
            var metadata = new GameMetadata(config, GameType.Local);
            foreach (var start in config.Players) metadata.JoinGame(new PlayerMetadata {
                PlayerStartConfiguration = start, AiType = AIType.Hard, NickName = "Smoke " + start.Color, PlayerNetworkId = "local-" + start.Color });
            var app = history.Repository.Create(metadata); app.Game.StartGame();
            // Reach an ordinary two-action turn; designate its actor as human.
            while (app.Game.GameProgressInformation.CurrentRoundNumber < 2) { TurnHistoryRegression.PassTurn(app); TurnHistoryRegression.End(app); }
            var human = app.Game.GameProgressInformation.CurrentPlayer;
            app.Metadata.GetPlayerMetadataByColor(human.Color).AiType = AIType.Human;
            var state = native ? NativeGameState.Create(app.Metadata, p => "easy") : LocalGameState.Create(app.Metadata, p => "balanced", p => "easy");
            state["game_id"] = app.Metadata.GameId; state["ready"] = true; SaveExtensions.Set(app.Metadata, state);
            history.Repository.Save(app); history.SetGame(app);
            var output = new HumanCheckpointOutput(router, history);
            var play = new PlayGameUseCase(output, history, null, new Analytics());
            router.UseCases[typeof(PlayGameUseCase)] = play;
            TurnHistoryRegression.PassTurn(app); play.EndTurn(); int humanCheckpoint = app.Game.Revision;
            Check(output.Saves == 1 && output.Starts == 0, "human End checkpoints before AI dispatch");
            output.Complete();
            int actor = (int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            router.ShowView<IPlayAiTurnView>(); float startedTurn = Time.realtimeSinceStartup; turn.PlayAITurn(false);
            float until = startedTurn + 15;
            while ((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup < until) yield return null;
            Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle, "live search and validation retire within short smoke");
            Check(app.Game.TurnFlow.GetRemainingActionsCount() == 0 && app.Game.Revision > humanCheckpoint, "actual AI search completes both actions");
            Check(output.Saves == 1 && history.Repository.Load(app.Metadata.GameId).Game.Revision == humanCheckpoint, "actual search and action commits perform no saves");
            play.EndTurn(); int aiCheckpoint = app.Game.Revision;
            Check(output.Saves == 2 && output.Starts == 1, "actual AI End writes exactly once before next dispatch");
            play.EndTurn(); Check(output.Saves == 2 && app.Game.Revision == aiCheckpoint, "duplicate AI End cannot repeat save or advance actor");
            var saved = history.Repository.Load(app.Metadata.GameId);
            Check(saved.Game.Revision == aiCheckpoint && saved.Game.GameProgressInformation.CurrentPlayer.Color == app.Game.GameProgressInformation.CurrentPlayer.Color,
                "filesystem checkpoint resumes after completed AI turn");
            Check(JToken.DeepEquals(LocalGameState.Read(saved.Metadata)["assignments"], state["assignments"]), "disk checkpoint preserves NN/V4 seat and difficulty assignments");
            output.Complete(); Check(output.Starts == 2, "next player dispatch follows AI save completion");
            history.SetGame(saved);
            rows.Add(new JObject { ["family"] = native ? "nn" : "v4", ["search_ms"] = 4000, ["actor"] = actor,
                ["human_checkpoint"] = humanCheckpoint, ["ai_checkpoint"] = aiCheckpoint, ["saves_after_setup"] = output.Saves,
                ["seconds"] = Time.realtimeSinceStartup - startedTurn });
        }
        Check(errors.Count == 0, "no Unity errors during cancellation and live AI checkpoint smoke: " + String.Join("\n", errors));
        File.WriteAllText(Output + ".turn-history.json", new JObject { ["passed"] = true, ["regression"] = regression, ["live_turns"] = rows }.ToString());
    }
}
