using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassHost;
using BrassLegacyAI.Native;
using BrassApplication.Game;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.Events;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases.Events;
using Newtonsoft.Json.Linq;

// Replays a supplied event trace entirely in memory; never opens a repository
// or writes a save. The runner keeps the user's trace in ignored artifacts.
public static class PublicHistoryBonusChecks
{
    static int checks;
    static void Check(bool okay, string message)
    {
        checks++;
        if (!okay) throw new Exception(message);
    }

    static BrassApplicationGame App(BoardGameRulesGame game)
    {
        return new BrassApplicationGame(game,
            new GameMetadata(game.GameStartConfiguration, GameType.Local) { GameId="bonus-history-regression" });
    }

    static BrassApplicationGame Replay(IList<IEvent> events, int persistedThrough)
    {
        var game=new BoardGameRulesGame(((GameStarted)events[0]).GameStartConfiguration);
        foreach (var e in events) {
            game.ApplyEvent(e);
            if (e.Metadata.Revision<=persistedThrough) game.ClearUncommittedEvents();
        }
        return App(game);
    }

    static JObject Observe(BrassApplicationGame app, IEnumerable<IEvent> persisted)
    {
        int[] debts;
        var host=Host.CapturePersonality(app,persisted,out debts);
        Check(host.Actor==(int)app.Game.GameProgressInformation.CurrentPlayer.Color,"Capture keeps the actual actor");
        return host.ObservePersonality(host.Actor);
    }

    static JArray ActionAt(JArray history, int ordinal)
    {
        var action=new JArray(); int current=0;
        foreach (var entry in history) {
            action.Add(entry["choice"].DeepClone());
            if ((string)entry["choice"]["kind"]!="finish") continue;
            if (++current==ordinal) return action;
            action.Clear();
        }
        throw new Exception("Missing confirmed action in public history");
    }

    static void OrdinaryActions()
    {
        foreach (string requested in new[]{"pass","loan","build","develop","network","scout"}) {
            var host=new Host(327); var app=App(host.Game); bool finished=false;
            for (int step=0;step<40 && !finished;step++) {
                var choices=host.Choices();
                var choice=host.Phase=="action" ? choices.First(c=>c.kind==requested) :
                    choices.FirstOrDefault(c=>c.kind=="finish")??choices[0];
                host.Apply(host.Revision,choice.id);
                if (choice.kind!="finish") continue;
                finished=true;
                var expected=(JObject)Wire.Token(host.Observe(host.Actor));
                var actual=Observe(app,null);
                Check(JToken.DeepEquals(expected["history"],actual["history"]),"Ordinary action history unchanged: "+requested);
            }
            Check(finished,"Bounded ordinary action completed: "+requested);
        }
    }

    public static int Main(string[] args)
    {
        Exception failure=null; string observedError=null; int revision=0;
        try {
            List<IEvent> events;
            using (var stream=File.OpenRead(args[0]))
                events=new Frameworks.Persistence.Serialization.JsonSerializer().Deserialize<List<IEvent>>(stream);
            // Locate the actual bonus-before-flip sequence, rather than relying
            // on fixed revision numbers or fabricated policy observations.
            int bonus=events.FindLastIndex(e=>e is MerchantBonusReceived && ((MerchantBonusReceived)e).BonusType.ToString()=="Develop");
            Check(bonus>=0,"Trace contains a merchant development bonus");
            int finish=events.FindIndex(bonus,e=>e is ActionCountDecreased);
            int developed=events.FindIndex(bonus,e=>e is IndustryDeveloped);
            int flipped=events.FindIndex(bonus,e=>e is IndustryFlipped);
            Check(bonus<developed && developed<flipped && flipped<finish,"Bonus development precedes the sale flip");
            var app=Replay(events,Int32.MaxValue); revision=app.Game.Revision;
            if (args[1]=="baseline") {
                try { Observe(app,events); }
                catch (InvalidOperationException e) { observedError=e.Message; }
                Check(observedError=="unrecorded_stock_action","Baseline reproduces the reported exception");
            } else {
                string before=Wire.Json(Host.Capture(app,events).Canonical());
                var saved=Observe(app,events);
                Check(app.Game.Revision==revision && before==Wire.Json(Host.Capture(app,events).Canonical()),"Capture does not mutate the saved board");
                int ordinal=events.Take(finish+1).OfType<ActionCountDecreased>().Count();
                var action=ActionAt((JArray)saved["history"],ordinal);
                Check((string)action[0]["kind"]=="sell","Merchant bonus belongs to Sell");
                Check(action.Count(c=>(string)c["kind"]=="bonus_develop")==1 && !action.Any(c=>(string)c["kind"]=="develop"),"Free development is recorded only as a bonus");
                Check(action.Count(c=>c["card"]!=null && c["card"].Type!=JTokenType.Null)==1,"Exactly one card pays for the whole sale");
                Check(action.Count(c=>(string)c["kind"]=="sell")==events.Skip(bonus).Take(finish-bonus).OfType<IndustryFlipped>().Count(),"Both sold industries remain in the history");
                Check(action.Count(c=>(string)c["kind"]=="finish")==1,"Bonus does not consume a second action");
                var pending=Observe(Replay(events,-1),null);
                Check(JToken.DeepEquals(saved,pending),"Saved and in-memory histories agree");
                int split=events.Take(bonus).OfType<TurnStarted>().Last().Metadata.Revision;
                var mixed=Observe(Replay(events,split),events.Where(e=>e.Metadata.Revision<=split));
                Check(JToken.DeepEquals(saved,mixed),"Persisted history plus pending human turn agrees");
                var input=NativeInput.Capture(app,events,(int)saved["seat"],new JArray());
                Check(JToken.DeepEquals(saved,input["view"]),"Native AI input preparation succeeds with the same observation");
                OrdinaryActions();
            }
        } catch (Exception error) { failure=error; }
        File.WriteAllText(args[2],new JObject { ["passed"]=failure==null,["checks"]=checks,
            ["runtime_assembly"]=typeof(Host).Assembly.Location,
            ["arm"]=args[1],["revision"]=revision,["baseline_error"]=observedError,["error"]=failure?.ToString() }.ToString());
        if (failure!=null) Console.Error.WriteLine(failure);
        return failure==null?0:1;
    }
}
