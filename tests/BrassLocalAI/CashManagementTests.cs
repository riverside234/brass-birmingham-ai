using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class CashManagementTests
{
    static int checks;
    static readonly List<object> cases=new List<object>();
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    static Host Fixture(int cash,int marker,int actions,out JObject composition)
    {
        var source=new Host(1301);composition=Host.Composition(Host.Configuration(source.Game.GameStartConfiguration));
        var view=(JObject)Wire.Token(source.Observe(source.Actor));var player=Wire.Player(view,source.Actor);
        view["context"]["round"]=2;view["context"]["actions_remaining"]=actions;player["cash"]=cash;player["income_marker"]=marker;
        return new Host(1,Belief.Setup(view,composition,2));
    }
    static JObject After(Host source,JObject composition,Plan plan)
    {
        int seat=source.Actor;
        var result=new Host(1,Belief.Setup((JObject)Wire.Token(source.Observe(seat)),composition,2));
        foreach(var action in plan.actions)foreach(var choice in action)result.SearchApply(choice);
        Check(result.Remaining==0 && result.Game.PlayersTrack.Players.Count==4,"complete legal turn retains four players");
        return (JObject)Wire.Token(result.Observe(seat));
    }
    static void Protection()
    {
        foreach(string key in ProfileCatalog.Keys)foreach(int actions in new[]{1,2})foreach(int work in new[]{1,128})
        {
            JObject composition;var h=Fixture(10,0,actions,out composition);var view=(JObject)Wire.Token(h.Observe(h.Actor));
            string original=Wire.Hash(h.Canonical());var profile=ProfileCatalog.Create(key,4);profile.work_limit=work;
            var published=new List<Plan>();
            var decision=new Planner(composition).Decide(view,h.Choices(),profile,true,1,default(System.Threading.CancellationToken),p=>published.Add(p));
            foreach(var p in published.Concat(new[]{decision.selected}))
            {
                Check(p.actions.Count==actions,"publish complete turns only, including exhausted-budget incumbent");
                var after=After(h,composition,p);var money=Wire.Player(after,h.Actor);
                Check((int)money["cash"]+(int)money["income"]>=0,"do not spend the known income payment when a safe plan exists");
                Check(p.cash.shortfall==0,"cash ledger agrees with independently replayed legal actions");
            }
            Check(decision.nodes<=work+80 && decision.failures==0,"bounded cash fallback and no failed candidate");
            Check(Wire.Hash(h.Canonical())==original,"cash search leaves live state and hidden cards unchanged");
            cases.Add(new{key,actions,work,decision.nodes,decision.selected.signature});
        }
        JObject cards;var source=Fixture(10,0,1,out cards);var old=Profile.Read(ProfileCatalog.Snapshot("balanced",4,false,ProfileCatalog.PreviousRuntime));old.work_limit=128;
        var prior=new Planner(cards).Decide((JObject)Wire.Token(source.Observe(source.Actor)),source.Choices(),old,true);
        var previous=Wire.Player(After(source,cards,prior.selected),source.Actor);
        Check((int)previous["cash"]==3 && (int)previous["income"]==-10,"exact v2 regression spends seven pounds needed for income");
        cases.Add(new{regression="v2 brewery leaves 7 pound shortfall",prior.selected.signature});
    }
    static void FinancingAndDistress()
    {
        foreach(string key in ProfileCatalog.Keys)
        {
            JObject composition;var h=Fixture(7,7,1,out composition);var view=(JObject)Wire.Token(h.Observe(h.Actor));
            var profile=ProfileCatalog.Create(key,4);profile.work_limit=1;profile.milliseconds=0;
            var d=new Planner(composition).Decide(view,h.Choices(),profile,false);
            var after=After(h,composition,d.selected);var p=Wire.Player(after,h.Actor);
            Check((int)p["cash"]+(int)p["income"]>=0,"expired-clock decision retains payment reserve");
            // Borrowing is allowed as a useful recovery, not forbidden by v3.
            h=Fixture(0,7,1,out composition);view=(JObject)Wire.Token(h.Observe(h.Actor));profile.work_limit=128;
            d=new Planner(composition).Decide(view,h.Choices(),profile,true);
            Check(d.selected.actions[0][0].StartsWith("loan|"),"legal financing recovers cash instead of passing into debt");
            h=Fixture(0,0,1,out composition);view=(JObject)Wire.Token(h.Observe(h.Actor));profile.work_limit=1;
            d=new Planner(composition).Decide(view,h.Choices(),profile,true);
            Check(d.selected.cash.shortfall==10 && d.selected.actions[0][0].StartsWith("pass|"),"unavoidable distress produces a legal decision");
            After(h,composition,d.selected);
        }
        JObject composition2;var rich=Fixture(80,20,1,out composition2);var richView=(JObject)Wire.Token(rich.Observe(rich.Actor));
        var balanced=ProfileCatalog.Create("balanced",4);balanced.work_limit=128;
        var choice=new Planner(composition2).Decide(richView,rich.Choices(),balanced,true);
        Check(!choice.selected.actions[0][0].StartsWith("loan|") && !choice.selected.actions[0][0].StartsWith("pass|"),"well-funded AI invests instead of hoarding more loans");
        var negative=(JObject)richView.DeepClone();var me=Wire.Player(negative,rich.Actor);me["cash"]=0;me["income"]=-10;
        negative["context"]["round"]=negative["context"]["round_limit"];
        negative["context"]["era"]="RailEra";
        Check(CashManagement.Project(negative,rich.Actor).shortfall==0,"no reserve required when final Rail income is not collected");
        negative["context"]["era"]="CanalEra";negative["context"]["introductory"]=true;
        Check(CashManagement.Project(negative,rich.Actor).shortfall==10,"final introductory Canal still pays income");
    }
    static void SaleUnlocksCredit()
    {
        JObject composition;var h=Fixture(0,0,2,out composition);int seat=h.Actor;
        var view=(JObject)Wire.Token(h.Observe(seat));
        var mat=(JArray)Wire.Player(view,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
        var tiles=mat.Where(t=>(int)t["level"]>=3).Take(2).Select(t=>t.DeepClone()).ToArray();
        Check(tiles.Length==2 && tiles.Sum(t=>(int)t["income"])<10,"two sales recover credit but do not erase all negative income");
        mat.Clear();
        var slots=new[]{view["slots"].Single(s=>(string)s["id"]=="Worcester:0"),view["slots"].Single(s=>(string)s["id"]=="Kidderminster:0")};
        for(int i=0;i<2;i++){tiles[i]["flipped"]=false;slots[i]["tile"]=tiles[i];}
        foreach(var link in view["links"].Where(l=>l["regions"].All(r=>new[]{"Worcester","Kidderminster","Gloucester"}.Contains((string)r))))link["owner"]=seat;
        foreach(var merchant in view["merchants"]){merchant["types"]=new JArray("CottonMill");merchant["beer"]=1;}
        // The two Gloucester merchants provide one beer each; there is no
        // brewery income that could independently lift the actor out of debt.
        h=new Host(1,Belief.Setup(view,composition,2));view=(JObject)Wire.Token(h.Observe(seat));
        foreach(string key in ProfileCatalog.Keys)
        {
            var profile=ProfileCatalog.Create(key,4);profile.work_limit=1;
            var d=new Planner(composition).Decide(view,h.Choices(),profile,true);
            var after=After(h,composition,d.selected);var p=Wire.Player(after,seat);
            Check((int)p["cash"]+(int)p["income"]>=0,"expired-budget rescue can sell then use newly unlocked credit");
            Check(d.selected.actions.Any(a=>a[0].StartsWith("loan|")),"income recovery loan is present");
        }
    }
    public static int Main(string[] args)
    {
        try{Protection();FinancingAndDistress();SaleUnlocksCredit();File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cases}));Console.WriteLine("Cash regression PASS "+checks);return 0;}
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,cases,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
