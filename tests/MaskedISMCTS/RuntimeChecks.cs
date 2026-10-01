using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
public static class RuntimeChecks
{
    static int checks;
    static void Check(bool value,string reason){checks++;if(!value)throw new Exception(reason);}
    static JObject Config(int count)
    {return new JObject{["player_count"]=count,["introductory"]=false,["seats"]=new JArray(Enumerable.Range(0,count).Select(i=>new JObject{["color"]=i,["sitting_order"]=i}))};}
    static Host SecondRound(int count)
    {
        var h=new Host(411,new JObject{["configuration"]=Config(count)});
        while(h.Game.GameProgressInformation.CurrentRoundNumber==1)
        {h.SearchApply(h.Choices().First(c=>c.kind=="pass").id);h.SearchApply(h.Choices().Single(c=>c.kind=="finish").id);h.Advance();}
        return h;
    }
    public static int Main(string[] args)
    {
        try
        {
            foreach(int count in new[]{2,3,4})foreach(string style in count==4?ProfileCatalog.Keys:new[]{"balanced"})
            {
                var h=SecondRound(count);var view=(JObject)Wire.Token(h.Observe(h.Actor));var legal=h.Choices();
                var composition=Host.Composition(Config(count));string before=Wire.Hash(h.Canonical());
                var p=ProfileCatalog.Create(style,count);p.work_limit=1;
                var d=new Planner(composition).Decide(view,legal,p,true);
                Check(d.mode=="masked_ismcts" && d.search.opponent_nodes==0,"one masked backend");
                Check(d.selected.actions.Count==2,"both ordinary turn actions complete");
                var clone=new Host(1,Belief.Setup(view,composition,17));
                foreach(var action in d.selected.actions)foreach(string id in action){Check(clone.Choices().Any(c=>c.id==id),"fresh legal continuation");clone.SearchApply(id);}
                Check(clone.Remaining==0 && clone.ActionComplete,"settled two-action turn");
                Check(Wire.Hash(h.Canonical())==before,"search never changes live state");
                var sampled=new Host(1,Belief.Setup(view,composition,98));
                Check(Planner.InformationKey(view,legal)==Planner.InformationKey((JObject)Wire.Token(sampled.Observe(h.Actor)),sampled.Choices()),"hidden-world root invariance");
            }
            var root=SecondRound(4);var v=(JObject)Wire.Token(root.Observe(root.Actor));var cards=Host.Composition(Config(4));
            var profile=ProfileCatalog.Create("balanced",4);profile.work_limit=10000;profile.simulations=2;
            var searched=new Planner(cards).Decide(v,root.Choices(),profile,true);
            Check(searched.search.completed==2,"finished real search iterations");
            Check(searched.search.root.Values.Sum(e=>e.visits)==2,"root backup counted once per completed world");
            Check(searched.search.root.Values.All(e=>e.returns.Length==4 && e.returns.All(x=>x>=0 && x<=e.visits)),"four stable-seat bounded returns");
            var cancelled=new CancellationTokenSource();cancelled.Cancel();
            var stopped=new Planner(cards).Decide(v,root.Choices(),profile,false,1,cancelled.Token);
            Check(stopped.selected.actions.Count==2 && stopped.search.completed==0 && stopped.search.root.Values.All(e=>e.visits==0),"cancelled search publishes only a complete legal incumbent");
            Check(DifficultyCatalog.Milliseconds("easy")==2000 && DifficultyCatalog.Milliseconds("mid")==5000 && DifficultyCatalog.Milliseconds("hard")==12000 && DifficultyCatalog.ExistingMilliseconds==10000,"original difficulty clocks");
            var old=ProfileCatalog.Snapshot("balanced",4,false,ProfileCatalog.OriginalRuntime);string saved=Wire.Json(old);
            var upgraded=ProfileCatalog.ForPlay(old);Check(upgraded.mode=="masked_ismcts" && upgraded.id==(string)old["id"] && Wire.Json(old)==saved,"original source validation then detached saved-profile upgrade");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,backend="masked_ismcts",iterations=searched.search.completed,maximum_intervening_turns=searched.search.maximum_intervening_turns}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
