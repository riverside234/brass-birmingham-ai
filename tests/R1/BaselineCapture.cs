// Execute exactly one stock selector decision per action. This is an H0 capture,
// not an improved policy, fairness certification or a competitive tournament.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Common;
using BrassApplication.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.AI.RuleBasedSystem.Rules;

public static class BaselineCapture
{
    sealed class RecordingSelector : RuleSelector
    {
        readonly RuleSelector original;
        public string Selected;
        public RecordingSelector(RuleSelector original):base(null){this.original=original;}
        public override IRule Select(BrassApplicationGame app)
        {
            var rule=original.Select(app); // exactly once, retaining the original returned object
            Selected=rule==null?null:rule.GetType().FullName+":"+rule.Id;
            return rule;
        }
    }
    static string Hash(string path){using(var s=File.OpenRead(path))using(var h=SHA256.Create())return BitConverter.ToString(h.ComputeHash(s)).Replace("-","").ToLowerInvariant();}
    public static int Main(string[] args)
    {
        string identity=args[0],output=args[1];int first=int.Parse(args[2]),count=int.Parse(args[3]);bool instrument=args.Length<5 || args[4]!="plain";
        var reports=new List<object>();
        for(int seed=first;seed<first+count;seed++)
        {
            var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=seed};
            var metadata=new GameMetadata(config,GameType.Local);
            var recorders=new Dictionary<int,RecordingSelector>();
            foreach(var start in config.Players)
            {
                var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="h0-"+start.Color,NickName="H0 "+start.Color,AiType=AIType.Hard};p.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(p);
                var field=typeof(HardAiBehavior).GetField("_actionRuleSelector",BindingFlags.Instance|BindingFlags.NonPublic);
                var recorder=new RecordingSelector((RuleSelector)field.GetValue(p.AIBehaviour));if(instrument)field.SetValue(p.AIBehaviour,recorder);recorders.Add((int)start.Color,recorder);
                metadata.JoinGame(p);
            }
            var rules=new BoardGameRulesGame(config);var app=new BrassApplicationGame(rules,metadata);rules.StartGame();
            var initial=new{players=rules.PlayersTrack.Players.Select(p=>new{seat=(int)p.Color,hand=p.Hand.Select(c=>c.ToString()).ToArray(),hidden_setup=p.DiscardPile.Select(c=>c.ToString()).ToArray()}).ToArray(),deck=rules.Board.CardDeck.Select(c=>c.ToString()).ToArray(),merchants=rules.Board.Regions.OfType<BoardGameRules.Entities.Board.Regions.BorderRegion>().SelectMany(r=>r.MerchantSlots.Select((s,i)=>new{region=r.Name.ToString(),slot=i,types=s.MerchantTile==null?null:s.MerchantTile.BuildingTypes.Select(t=>t.ToString()).ToArray()})).ToArray()};
            int actions=0,turns=0,debts=0;string failure=null;var trace=new List<object>();var watch=Stopwatch.StartNew();
            var selector=RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());
            try
            {
                while(!rules.GameProgressInformation.IsGameOver)
                {
                    if(turns>=80 || actions>=160)throw new InvalidOperationException("progress_limit");
                    var player=rules.GameProgressInformation.CurrentPlayer;
                    var behavior=metadata.GetPlayerMetadataByColor(player.Color).AIBehaviour;
                    while(rules.TurnFlow.GetRemainingActionsCount()>0)
                    {
                        var timer=Stopwatch.StartNew();
                        var move=behavior.GetNextAIMove(app); // Never call Print/Select again.
                        if(move==null)throw new InvalidOperationException("no_move");
                        int revision=rules.Revision;move.Invoke();timer.Stop();
                        string action=rules.TurnFlow.CurrentAction==null?"none":rules.TurnFlow.CurrentAction.GetType().Name;
                        rules.TurnFlow.ConfirmCurrentAction();actions++;
                        // Deep JSON copy now: vendor event collections and objects are mutable.
                        string events=JsonConvert.SerializeObject(rules.GetUncommittedEvents().Where(e=>e.Metadata.Revision>revision).ToArray());
                        trace.Add(new{seat=(int)player.Color,era=rules.GameProgressInformation.CurrentGameEra.ToString(),round=rules.GameProgressInformation.CurrentRoundNumber,action,selected_rule=recorders[(int)player.Color].Selected,milliseconds=timer.Elapsed.TotalMilliseconds,events});
                    }
                    rules.GameFlow.EndTurn();turns++;
                    if(rules.GameFlow.IsLastTurnOver())
                    {
                        if(rules.GameFlow.IsSellIndustryNecessary()){debts++;selector.Select(app).MakeAction(app).Invoke();}
                        rules.GameFlow.EndRound();
                    }
                }
            }
            catch(Exception e){failure=e.ToString();}
            var report=new{seed,player_count=4,identity,completed=rules.GameProgressInformation.IsGameOver,failure,actions,turns,debts,elapsed_ms=watch.Elapsed.TotalMilliseconds,
                scores=rules.PlayersTrack.Players.OrderBy(p=>(int)p.Color).Select(p=>new{seat=(int)p.Color,vp=p.VP,cash=p.Money,income=p.IncomeLevel}).ToArray(),
                ranking=rules.GameProgressInformation.EndGameRanking,initial,trace};reports.Add(report);
            Console.WriteLine(JsonConvert.SerializeObject(new{seed,identity,completed=rules.GameProgressInformation.IsGameOver,actions,turns,error=failure==null?null:failure.Split('\n')[0]}));
        }
        File.WriteAllText(output,JsonConvert.SerializeObject(new{format="brass-h0-capture/0.1.0",identity,player_count=4,assembly_sha256=Hash(typeof(BoardGameRulesGame).Assembly.Location),runtime=Environment.Version.ToString(),policy="original Hard priority rules; guarded profile includes only existing patch behavior guards",limitations=new[]{"all-Hard tables, not mixed opponents or seat-balanced strength evidence","failures are reliability outcomes, never wins","private diagnostic trace: do not use as policy input","guarded stock has not been converted to the strict observation policy API"},games=reports},Formatting.Indented));
        return 0;
    }
}
