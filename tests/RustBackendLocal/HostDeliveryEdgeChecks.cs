using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using BrassHost;
using BrassLegacyAI;
using BrassApplication.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.GamePhases.Events;
using Newtonsoft.Json.Linq;

// Short structural checks for branches not reached by the fixed action corpus.
public static class HostDeliveryEdgeChecks
{
    static int checks;
    static void Check(bool okay,string message){checks++;if(!okay)throw new Exception(message);}
    static readonly MethodInfo resolver=typeof(Host).GetMethod("ResolveChoice");
    static void Membership(Host h)
    {
        var legal=h.Choices();
        foreach(var c in legal) {
            var actual=resolver==null?c:(Choice)resolver.Invoke(h,new object[]{c.id});
            Check(Wire.Json(c)==Wire.Json(actual),"bonus/debt selected membership");
        }
    }
    public static int Main(string[] args)
    {
        var results=new JArray();Exception error=null;
        try {
            var fixtures=JArray.Parse(File.ReadAllText(args[0]));
            foreach(string operation in new[]{"bonus_skip","bonus_develop","debt"}) {
                var fixture=(JObject)(operation=="debt"?fixtures.First(f=>f["input"]["view"]["slots"].Any(s=>s["tile"] is JObject)):fixtures[0]).DeepClone();
                var view=fixture["input"]["view"];var context=view["context"];
                if(operation=="debt") {
                    // Explicit synthetic debt perspective: retain every tile and
                    // card count, select an owner who actually has collateral.
                    int seat=(int)view["slots"].First(s=>s["tile"] is JObject)["tile"]["seat"];
                    Check((int)view["players"].Single(p=>(int)p["seat"]==seat)["hand_count"]==view["own_hand"].Count(),"synthetic debt perspective conserves the hand count");
                    view["seat"]=seat;
                }
                context["phase"]=operation=="debt"?"debt":"bonus";
                context["actor"]=view["seat"].DeepClone();context["turn_owner"]=view["seat"].DeepClone();
                if(operation=="debt") {
                    context["actions_remaining"]=0;
                    foreach(var p in view["players"])p["debt"]=(int)p["seat"]==(int)view["seat"]?10:0;
                }
                var h=ValidationBench.Root((JObject)fixture["input"]);Membership(h);
                if(operation!="debt") {
                    var choice=h.Choices().First(c=>c.kind==operation);h.SearchApply(choice.id);
                    Check(!h.Game.DevelopMerchantBonus.MerchantDevelopmentNeeded,"bonus completes exactly once");
                }
                results.Add(new JObject{["operation"]=operation,["choices"]=Wire.Token(h.Choices()),["state"]=Wire.Token(h.Canonical())});
            }
            List<IEvent> replay;var serializer=new Frameworks.Persistence.Serialization.JsonSerializer();
            using(var stream=File.OpenRead(args[2]))replay=serializer.Deserialize<List<IEvent>>(stream);
            var game=new BoardGameRulesGame(((GameStarted)replay[0]).GameStartConfiguration);
            foreach(var e in replay)game.ApplyEvent(e);
            var metadata=new GameMetadata(game.GameStartConfiguration,GameType.Local){GameId="topology-edge"};
            var app=new BrassApplicationGame(game,metadata);var host=Host.Capture(app,replay);
            var old=host.Links.ToArray();int turns=0,liquidations=0;
            while(game.GameProgressInformation.CurrentGameEra==GameEra.CanalEra && turns++<12) {
                while(host.Remaining>0) {
                    var pass=host.Choices().First(c=>c.kind=="pass");host.SearchApply(pass.id);host.SearchApply("finish|||||");
                }
                host.Advance();
                // The recorded late-Canal fixture can owe income. Complete its
                // actual legal liquidation instead of assuming cash suffices.
                while(host.Phase=="debt" && liquidations++<32) {
                    Membership(host);host.Apply(host.Revision,host.Choices().First().id);
                }
                Check(host.Phase!="debt","bounded liquidation reaches the next ordinary boundary");
            }
            Check(game.GameProgressInformation.CurrentGameEra==GameEra.RailEra,"actual event replay crosses Canal to Rail");
            var expected=game.Board.Regions.SelectMany(r=>r.Links).Distinct().OrderBy(l=>l.LinkId,StringComparer.Ordinal).ToArray();
            Check(host.Links.SequenceEqual(expected),"cached link references equal the new era topology");
            Membership(host);
            results.Add(new JObject{["operation"]="actual-era-transition",["turns"]=turns,["liquidations"]=liquidations,["choices"]=Wire.Token(host.Choices()),["state"]=Wire.Token(host.Canonical())});
        }catch(Exception e){error=e;}
        File.WriteAllText(args[1],new JObject{["passed"]=error==null,["checks"]=checks,["error"]=error?.ToString(),["results"]=results}.ToString());
        if(error!=null)Console.Error.WriteLine(error);return error==null?0:1;
    }
}
