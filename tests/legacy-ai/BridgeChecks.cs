using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Common;
using BrassApplication.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
public static class BridgeChecks
{
    static int checks;
    static void Check(bool value,string name){checks++;if(!value)throw new Exception(name);}
    public static object Run(string root)
    {
        checks=0;var composition=JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/contracts/v0.2.0/card-composition.json")));
        var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/balanced-v1.json"))));profile.milliseconds=200;profile.work_limit=24;
        var h=new Host(327);var view=(JObject)Wire.Token(h.Observe(h.Actor));var before=Wire.Hash(h.Canonical());
        var ticket=Ticket.Create(h.GameId,h.Revision,h.Actor,h.Phase,profile);
        using(var c=new TurnCoordinator(ticket,view,h.Choices(),profile,composition))
        {
            Check(c.Worker.Wait(30000),"async_completion");Check(c.Ready,"valid_incumbent");
            Check(before==Wire.Hash(h.Canonical()),"background_did_not_mutate_live_engine");
            var p=c.Take(ticket,true);Check(p!=null,"play_now");Check(c.Take(ticket,true)==null,"one_shot");
            h.CommitPlan(p.actions[0]);Check(h.Actor!=ticket.actor,"one_confirmation_first_turn");
        }
        view=(JObject)Wire.Token(h.Observe(h.Actor));ticket=Ticket.Create(h.GameId,h.Revision,h.Actor,h.Phase,profile);
        using(var c=new TurnCoordinator(ticket,view,h.Choices(),profile,composition))
        {
            c.Worker.Wait(30000);var stale=JObject.FromObject(ticket).ToObject<Ticket>();stale.game_id="after-load";
            Check(c.Take(stale,true)==null,"stale_game_rejected");Check(c.Take(ticket,true)==null,"stale_response_cannot_retry");
        }
        using(var c=new TurnCoordinator(ticket,view,h.Choices(),profile,composition))
        {
            c.Dispose();c.Worker.Wait(30000);Check(c.Take(ticket,true)==null,"cancelled_menu_turn_cannot_apply");
        }
        using(var c=new TurnCoordinator(ticket,view,new System.Collections.Generic.List<Choice>(),profile,composition))
        {
            c.Worker.Wait(30000);Check(c.Failure!=null,"worker_failure_reported");Check(c.Take(ticket,true)==null,"failed_worker_no_partial_action");
        }
        // Exercise observer binding and one-shot adapter on an actual vendor game.
        var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=327};var metadata=new GameMetadata(config,GameType.Local){GameId="bridge-check"};
        foreach(var start in config.Players){var m=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="test-"+start.Color,AiType=AIType.Hard};m.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(m);metadata.JoinGame(m);}
        var game=new BoardGameRulesGame(config);game.StartGame();var app=new BrassApplicationGame(game,metadata);
        var live=Host.Capture(app);view=(JObject)Wire.Token(live.Observe(live.Actor));
        var d=new Planner(composition).Decide(view,live.Choices(),profile,true);
        int remaining=game.TurnFlow.GetRemainingActionsCount();
        var adapter=new LegacyBehavior(app,()=>live.LiveCommit(d.selected.actions[0],composition));
        var command=adapter.GetNextAIMove(app);command();Check(game.TurnFlow.GetRemainingActionsCount()==remaining-1,"adapter_confirms_once");
        bool rejected=false;try{command();}catch(InvalidOperationException){rejected=true;}Check(rejected,"adapter_duplicate_rejected");
        // The engine remains at its own end-turn boundary for the vendor presenter.
        Check(game.GameProgressInformation.CurrentTurnNumber==1,"live_adapter_does_not_end_turn");
        var events=game.GetUncommittedEvents().ToArray();
        var observed=Wire.Token(Host.Capture(app).Observe((int)game.GameProgressInformation.CurrentPlayer.Color));
        game.ClearUncommittedEvents();
        Check(JToken.DeepEquals(observed,Wire.Token(Host.Capture(app,events).Observe((int)game.GameProgressInformation.CurrentPlayer.Color))),"saved_history_observation_parity");
        foreach(var count in new[]{NumberOfPlayers.Two,NumberOfPlayers.Three})
        {
            var other=new GameStartConfiguration(count){Seed=19};var rules=new BoardGameRulesGame(other);rules.StartGame();
            rejected=false;try{Host.Capture(new BrassApplicationGame(rules,new GameMetadata(other,GameType.Local)));}catch(Exception e){rejected=e.Message=="four_players_required";}
            Check(rejected,"non_four_player_observer_rejected");
        }
        // Intent follows the saved branch and can never migrate between seats.
        var journalPath=Path.Combine(root,"artifacts/legacy-ai/intent-check-"+Guid.NewGuid().ToString("N")+".json");
        var profiles=new JArray("a","b","c","d");
        var journal=new IntentJournal(journalPath,"game-A",profiles);
        var intent=new IntentState{profile="cotton",seat=0,revision=10,focus="CottonMill",target="Worcester:0"};
        journal.Commit(12,intent);intent.target="changed-outside-journal";
        journal=new IntentJournal(journalPath,"game-A",profiles);
        Check(journal.Read(0,12,"cotton").target=="Worcester:0","intent_save_load_and_detached_state");
        Check(journal.Read(0,11,"cotton")==null && journal.Read(1,12,"cotton")==null,"intent_revision_and_seat_isolation");
        rejected=false;try{journal.Read(0,12,"supplier");}catch(Exception){rejected=true;}Check(rejected,"intent_profile_isolation");
        journal.Commit(30,new IntentState{profile="supplier",seat=1,revision=20});
        journal.Commit(15,new IntentState{profile="cotton",seat=0,revision=12,focus="Brewery"});
        journal=new IntentJournal(journalPath,"game-A",profiles);
        Check(journal.Read(1,100,"supplier")==null && journal.Read(0,15,"cotton").focus=="Brewery","intent_undo_branch_prunes_future");
        rejected=false;try{new IntentJournal(journalPath,"other-game",profiles);}catch(Exception){rejected=true;}Check(rejected,"intent_game_isolation");
        var corrupt=JObject.Parse(File.ReadAllText(journalPath));corrupt["records"][0]["revision"]=999;File.WriteAllText(journalPath,Wire.Json(corrupt));
        rejected=false;try{new IntentJournal(journalPath,"game-A",profiles);}catch(Exception){rejected=true;}Check(rejected,"intent_corruption_rejected");
        File.Delete(journalPath);
        var s2=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/balanced-s2-v2.json"))));s2.work_limit=32;s2.milliseconds=200;
        h=new Host(904);view=(JObject)Wire.Token(h.Observe(h.Actor));ticket=Ticket.Create(h.GameId,h.Revision,h.Actor,h.Phase,s2);
        using(var c=new TurnCoordinator(ticket,view,h.Choices(),s2,composition))
        {
            Check(c.Worker.Wait(30000),"s2_async_completion");var plan=c.Take(ticket,true);Check(plan!=null,"s2_play_now_complete_incumbent");h.CommitPlan(plan.actions[0]);
        }
        var fullMatch=LiveMatchChecks.Run(root);
        return new{passed=true,assertions=checks,runtime=Environment.Version.ToString(),full_match=fullMatch};
    }
    public static int Main(string[] args)
    {try{var result=Run(args[0]);File.WriteAllText(args[1],Wire.Json(result));Console.WriteLine(Wire.Json(result));return 0;}catch(Exception ex){Console.Error.WriteLine(ex);return 1;}}
}
