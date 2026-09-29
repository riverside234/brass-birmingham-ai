using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassPatch.Persistence;

// H3 behaviors ported from tests/legacy-ai/test_advanced.py to the deployed runtime.
// This suite uses in-memory metadata only; it never loads or writes game saves.
public static class TacticalIntentTests
{
    static int checks;
    static void Check(bool result, string name) { checks++; if (!result) throw new Exception(name); }
    static JObject Fixture()
    {
        var h = new Host(1301); var v = (JObject)Wire.Token(h.Observe(h.Actor)); int seat = h.Actor;
        v["context"]["round"] = 3; v["context"]["actions_remaining"] = 2;
        var p = Wire.Player(v, seat); p["cash"] = 50;
        string[] cards = { "location:Worcester", "location:Coalbrookdale", "location:Birmingham", "location:Coventry" };
        v["deck_count"] = (int)v["deck_count"] + v["own_hand"].Count() - cards.Length;
        p["hand_count"] = cards.Length; v["own_hand"] = new JArray(cards.OrderBy(c=>c));
        foreach (string id in new[]{"CanalWorcesterGloucester","CanalCoalbrookdaleShrewsbury"}) v["links"].Single(l=>(string)l["id"]==id)["owner"] = seat;
        var borders = new HashSet<string>(v["merchants"].Select(m=>Evaluator.Region((string)m["id"])));
        foreach (string city in new[]{"Birmingham","Coventry"})
        {
            var queue = new Queue<Tuple<string,List<JToken>>>(); var seen = new HashSet<string>{city};
            queue.Enqueue(Tuple.Create(city,new List<JToken>()));
            while(queue.Count>0)
            {
                var current=queue.Dequeue();
                if(borders.Contains(current.Item1)){foreach(var l in current.Item2)l["owner"]=seat;break;}
                foreach(var l in v["links"].Where(l=>((string)l["id"]).StartsWith("Canal") && l["regions"].Any(r=>(string)r==current.Item1)))
                    foreach(var r in l["regions"].Select(x=>(string)x)) if(seen.Add(r)) queue.Enqueue(Tuple.Create(r,current.Item2.Concat(new[]{l}).ToList()));
            }
        }
        p["link_tiles"] = 14-v["links"].Count(l=>l["owner"]!=null && l["owner"].Type!=JTokenType.Null && (int)l["owner"]==seat);
        foreach(var m in v["merchants"]){m["types"]=new JArray("CottonMill","Manufacturer","Pottery");m["beer"]=1;}
        v["iron_market"]=0;return v;
    }
    static void EntryRetentionRecovery()
    {
        foreach(var pair in new[]{Tuple.Create("cotton","CottonMill"),Tuple.Create("goods-beer","Manufacturer"),Tuple.Create("supplier","IronWorks"),Tuple.Create("pottery-flex","Pottery")})
        {
            var v=Fixture();var p=ProfileCatalog.Create(pair.Item1,4);var a=Tactics.Refresh(v,p,null);
            Check(a.focus==pair.Item2,"conditional entry "+pair.Item1);string profile=p.id;
            v["context"]["revision"]=(int)v["context"]["revision"]+1;var b=Tactics.Refresh(v,p,a);
            Check(b.target==a.target && b.reason.Contains("retained") && b.age==1,"feasible retention "+pair.Item1);
            var unchanged=Tactics.Refresh(v,p,b);Check(unchanged.age==b.age,"same-decision refresh does not age "+pair.Item1);
            var stalled=b;
            for(int i=0;i<3;i++){v["context"]["revision"]=(int)v["context"]["revision"]+1;stalled=Tactics.Refresh(v,p,stalled);}
            Check(stalled.focus=="balanced" && stalled.reason.Contains("stalled"),"four-decision recovery "+pair.Item1);
            v["own_hand"]=new JArray();var c=Tactics.Refresh(v,p,b);
            Check(c.focus=="balanced" && c.reason.Contains("abandoned"),"lost-card recovery "+pair.Item1);
            Check(p.id==profile && a.profile==c.profile,"adaptive intent preserves identity "+pair.Item1);
        }
        var rail=Fixture();rail["context"]["era"]="RailEra";var rp=ProfileCatalog.Create("rail-tempo",4);var ri=Tactics.Refresh(rail,rp,null);
        Check(ri.focus=="network","rail conditional network focus");
        Wire.Player(rail,(int)rail["seat"])["cash"]=0;Wire.Player(rail,(int)rail["seat"])["income_marker"]=0;
        rail["own_hand"]=new JArray();var rr=Tactics.Refresh(rail,rp,ri);Check(rr.focus=="balanced" && rr.reason.Contains("abandoned"),"rail funding recovery");
    }
    static void BeerRecovery()
    {
        var v=Fixture();int seat=(int)v["seat"];var p=ProfileCatalog.Create("cotton",4);
        var tiles=(JArray)Wire.Player(v,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
        int index=Enumerable.Range(0,tiles.Count).First(i=>(int)tiles[i]["level"]==2);var tile=(JObject)tiles[index].DeepClone();
        for(int i=0;i<=index;i++)tiles.RemoveAt(0);tile["flipped"]=false;v["slots"].Single(s=>(string)s["id"]=="Worcester:0")["tile"]=tile;
        foreach(var m in v["merchants"])m["beer"]=0;
        var a=Tactics.Refresh(v,p,null);Check(a.focus=="Brewery" && a.milestone=="secure_own_beer" && a.required_cards.Length>0,"cotton can switch to own brewery contingency");
        Wire.Player(v,seat)["cash"]=0;var b=Tactics.Refresh(v,p,a);Check(b.focus=="balanced" && b.reason.Contains("no affordable"),"unaffordable brewery contingency falls back");
        foreach(var m in v["merchants"])m["types"]=new JArray();Check(Tactics.Refresh(v,p,null).focus=="balanced","no matching merchant falls back");
    }
    static void Bounds()
    {
        var choose=typeof(Planner).GetMethod("Choose",BindingFlags.NonPublic|BindingFlags.Static);
        foreach(string key in ProfileCatalog.Keys)
        {
            var p=ProfileCatalog.Create(key,4);var best=new Plan{score=100,preference=0,cash=new CashProjection()};
            var poor=new Plan{score=100-p.credible_window-0.01,preference=100,cash=new CashProjection()};
            var chosen=(Plan)choose.Invoke(null,new object[]{new List<Plan>{best,poor},p});
            Check(object.ReferenceEquals(chosen,best),"style never outranks credible bound "+key);
            var insolvent=new Plan{score=200,preference=100,cash=new CashProjection{shortfall=1}};
            chosen=(Plan)choose.Invoke(null,new object[]{new List<Plan>{best,insolvent},p});
            Check(object.ReferenceEquals(chosen,best),"style and speculative score cannot spend income reserve "+key);
            var winner=new Plan{score=1,preference=0,ledger=new Ledger{win_share=1,rank=1,banked=80}};
            var loser=new Plan{score=0,preference=100,ledger=new Ledger{win_share=0,rank=2,banked=79}};
            chosen=(Plan)choose.Invoke(null,new object[]{new List<Plan>{winner,loser},p});Check(object.ReferenceEquals(chosen,winner),"terminal win outranks style "+key);
        }
        var v=Fixture();var composition=Host.Composition(Host.Configuration(new Host(1301).Game.GameStartConfiguration));
        var h=new Host(1,Belief.Setup(v,composition,2));v=(JObject)Wire.Token(h.Observe(h.Actor));
        foreach(string key in ProfileCatalog.Keys)
        {
            var p=ProfileCatalog.Create(key,4);p.work_limit=140;var d=new Planner(composition).Decide(v,h.Choices(),p,true);
            Check(d.selected.actions.Count==2 && d.selected.intent.profile==p.id,"complete adaptive selected plan "+key);
            Check(d.selected.score>=d.alternatives.Where(a=>a.cash.shortfall==d.selected.cash.shortfall).Max(a=>a.score)-p.credible_window-0.000001,"complete selected plan obeys credible bound within affordable plans "+key);
            var replay=new Host(1,Belief.Setup(v,composition,2));foreach(var action in d.selected.actions)foreach(string id in action)replay.SearchApply(id);
            Check(replay.Remaining==0,"adaptive plan legal through both actions "+key);
        }
    }
    static void InMemoryHandoff()
    {
        var h=new Host(1301);var metadata=new GameMetadata(h.Game.GameStartConfiguration,GameType.Local);
        foreach(var start in metadata.GameStartConfiguration.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="audit-"+start.Color,NickName="Audit",AiType=AIType.Hard});
        var state=LocalGameState.Create(metadata,p=>"cotton");state["ready"]=true;SaveExtensions.Set(metadata,state);
        int seat=h.Actor;var profile=ProfileCatalog.Create("cotton",4);var v=Fixture();var original=Tactics.Refresh(v,profile,null);
        // Exercise the bridge's in-memory handoff only; never create a repository or save file.
        var events=h.Game.GetUncommittedEvents().ToList();int revision=h.Game.Revision;original.revision=revision;
        LocalGameState.CommitIntent(metadata,original,revision,events);string target=original.target;original.target="outside mutation";
        var read=LocalGameState.Intent(LocalGameState.Read(metadata),seat,revision,events);
        Check(read!=null && read.target==target,"in-memory intent handed to following decision as detached copy");
        read.target="outside read mutation";Check(LocalGameState.Intent(LocalGameState.Read(metadata),seat,revision,events).target==target,"read intent detached");
        Check(LocalGameState.Intent(LocalGameState.Read(metadata),(seat+1)%4,revision,events)==null,"other seat cannot inherit intent");
    }
    public static int Main(string[] args)
    {
        try{EntryRetentionRecovery();BeerRecovery();Bounds();InMemoryHandoff();File.WriteAllText(args[0],JObject.FromObject(new{passed=true,checks,scope="Candidate DLLs; adaptive tactics, complete-plan bounds and in-memory intent handoff; no save games"}).ToString());Console.WriteLine("PASS: "+checks+" adaptive-intent assertions");return 0;}
        catch(Exception e){File.WriteAllText(args[0],JObject.FromObject(new{passed=false,checks,error=e.ToString()}).ToString());Console.Error.WriteLine(e);return 1;}
    }
}

