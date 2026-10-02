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
            string repository=args.Length>1?args[1]:Path.GetFullPath(Path.Combine(Path.GetDirectoryName(args[0]),"../.."));
            foreach(string key in ProfileCatalog.Keys)
            {
                using(var hash=System.Security.Cryptography.SHA256.Create())
                {
                    var source=File.ReadAllBytes(Path.Combine(repository,"configs/legacy-ai/"+key+"-v3-masked-economy.json"));
                    Check(ProfileCatalog.SourceHash(key)==BitConverter.ToString(hash.ComputeHash(source)).Replace("-", "").ToLowerInvariant(),"packaged profile bytes match frozen config");
                }
            }
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
            var opening=new Host(843400,new JObject{["configuration"]=Config(4)});
            var openingView=(JObject)Wire.Token(opening.Observe(opening.Actor));var openingProfile=ProfileCatalog.Create("balanced",4);
            openingProfile.work_limit=10000;openingProfile.simulations=1;
            var openingDecision=new Planner(cards).Decide(openingView,opening.Choices(),openingProfile,true);
            Check(openingDecision.alternatives.Any(plan=>plan.actions[0][0].StartsWith("build|")),"one-action Canal opening compares a completed industry incumbent");
            var profile=ProfileCatalog.Create("balanced",4);profile.work_limit=10000;profile.simulations=8;
            var searched=new Planner(cards).Decide(v,root.Choices(),profile,true);
            Check(searched.search.completed==8,"finished real search iterations");
            Check(searched.search.root.Values.Sum(e=>e.visits)==8,"root backup counted once per completed world");
            Check(searched.search.root.Values.All(e=>e.returns.Length==4 && e.returns.All(x=>x>=0 && x<=e.visits)),"four stable-seat bounded returns");
            Check(searched.search.revisited_plans>=2 && searched.search.incumbent_visits>=2,"complete turns receive their own repeated credit");
            Check(searched.search.normalization_uses>0,"normalization waits for two supported siblings");
            Check(searched.search.plans.Values.All(e=>e.visits>=e.scenarios.Count && e.returns.Length==4),"raw plan evidence and scenario identity");
            Check(searched.search.maximum_intervening_turns==0,"zero-reply lab cutoff settles current turn");
            var easy=ProfileCatalog.Create("balanced",4);ProfileCatalog.ConfigureClock(easy,2000);
            Check(easy.root_top_k==4 && easy.interior_top_k==2 && easy.opponent_turn_limit==1,"easy heuristic settings");
            ProfileCatalog.ConfigureClock(easy,12000);Check(easy.root_top_k==8 && easy.interior_top_k==4 && easy.opponent_turn_limit==2,"hard heuristic settings");
            Check(Math.Abs(SearchSelection.Transform(.26,.25,.251,.05)-.2)<1e-9,"Q regularization prevents tiny-range amplification");
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
