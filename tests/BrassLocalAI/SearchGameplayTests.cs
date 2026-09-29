using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class SearchGameplayTests
{
    static int checks;
    static void Check(bool condition,string message) { checks++;if(!condition)throw new Exception(message); }
    static JObject Configuration(int count,bool intro)
    {
        int[] colors=count==2?new[]{3,1}:count==3?new[]{3,0,2}:new[]{2,0,3,1};
        return new JObject { ["player_count"]=count,["introductory"]=intro,
            ["seats"]=new JArray(colors.Select((color,index)=>new JObject { ["color"]=color,["sitting_order"]=index })) };
    }
    static Host Fixture(int count,bool intro,bool rail,int actions,string boundary,out JObject composition)
    {
        var configuration=Configuration(count,intro);composition=Host.Composition(configuration);
        var source=new Host(1701,new JObject { ["configuration"]=configuration });
        int round=source.Game.GameFlow.GetNumberOfRounds(source.Game.GameStartConfiguration.NumberOfPlayers)-(boundary=="earlier_round"?1:0);
        int turn=count-(boundary=="earlier_seat"?1:0);string era=rail?"RailEra":"CanalEra";
        // Reach the requested scheduler boundary through legal passes. Keep its
        // real remaining hands, public discards and deck counts in every fixture.
        int setupActions=0;
        while(source.Game.GameProgressInformation.CurrentGameEra.ToString()!=era ||
            source.Game.GameProgressInformation.CurrentRoundNumber!=round || source.Game.GameProgressInformation.CurrentTurnNumber!=turn)
        {
            if(++setupActions>512 || source.Terminal)throw new Exception("fixture boundary was not reached");
            source.SearchApply(source.Choices().First(c=>c.kind=="pass").id);
            source.SearchApply(source.Choices().Single(c=>c.kind=="finish").id);source.Advance();
        }
        while(source.Remaining>actions)
        {
            source.SearchApply(source.Choices().First(c=>c.kind=="pass").id);
            source.SearchApply(source.Choices().Single(c=>c.kind=="finish").id);
        }
        int seat=source.Actor;var view=(JObject)Wire.Token(source.Observe(seat));
        var me=Wire.Player(view,seat);me["cash"]=0;me["income_marker"]=7;me["income"]=-3;me["vp"]=80;me["link_tiles"]=0;
        var cotton=me["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"].First(t=>!rail || (int)t["level"]==2).DeepClone();cotton["flipped"]=true;
        foreach(var mat in me["mat"])mat["tiles"]=new JArray();
        view["slots"].Single(s=>(string)s["id"]=="Worcester:0")["tile"]=cotton;
        foreach(var player in view["players"].Where(p=>(int)p["seat"]!=seat))player["vp"]=0;
        view["players"].First(p=>(int)p["seat"]!=seat)["vp"]=78;
        return new Host(1,Belief.Setup(view,composition,2));
    }
    static Host Apply(Host source,JObject composition,Plan plan,bool settle)
    {
        var view=(JObject)Wire.Token(source.Observe(source.Actor));
        var result=new Host(1,Belief.Setup(view,composition,71));
        foreach(var action in plan.actions)foreach(var choice in action)result.SearchApply(choice);
        if(settle)
        {
            result.Advance();
            while(result.Phase=="debt")
            {
                // Every fixture has only one liquidatable tile, so no choice model
                // is involved in the asserted official win/loss comparison.
                Check(result.Choices().Count==1,"forced final liquidation has a unique legal choice");
                result.SearchApply(result.Choices()[0].id);
            }
        }
        return result;
    }
    static void FinalOutcomes()
    {
        foreach(int count in new[]{2,3,4})foreach(bool intro in new[]{false,true})foreach(int actions in new[]{1,2})
        {
            JObject composition;var source=Fixture(count,intro,!intro,actions,"final",out composition);
            var view=(JObject)Wire.Token(source.Observe(source.Actor));string original=Wire.Hash(source.Canonical());
            foreach(string key in ProfileCatalog.Keys)
            {
                var profile=ProfileCatalog.Create(key,count,intro);profile.work_limit=1024;profile.milliseconds=10000;
                var decision=new Planner(composition).Decide(view,source.Choices(),profile,true);
                Check(decision.selected.actions.Count==actions,"candidate completes the final ordinary turn");
                Check(decision.selected.ledger.win_share.HasValue,"selected final result always uses terminal units");
                foreach(var plan in decision.alternatives)
                {
                    var result=Apply(source,composition,plan,true);Check(result.Terminal,"settled final candidate reaches game end");
                    var official=((JObject)Wire.Token(result.Result()))["seats"].Single(p=>(int)p["seat"]==source.Actor);
                    Check(plan.ledger.win_share.HasValue && Math.Abs(plan.score-(double)official["utility"])<1e-9 && plan.ledger.rank==(int)official["rank"],"candidate uses official settled win share and rank");
                    Check(decision.selected.ledger.win_share>=plan.ledger.win_share,"terminal selection never trades outcome for style or heuristic points");
                }
                if(intro)
                {
                    Check(decision.selected.ledger.win_share==1,"avoid forced-liquidation loss when a loan wins");
                    Check(decision.selected.actions.Any(a=>a[0].StartsWith("loan|")),"winning final introductory plan includes financing");
                    var pass=decision.family_best["pass"];
                    if(actions==1)Check(pass.ledger.win_share==0 && pass.ledger.banked==78,"pass is a settled 78VP loss, not an 85VP estimate");
                }
                Check(Wire.Hash(source.Canonical())==original,"final settlement search leaves live game and cards unchanged");
            }
        }
    }
    static void NonFinalBoundaries()
    {
        foreach(int count in new[]{2,3,4})foreach(bool intro in new[]{false,true})foreach(string boundary in new[]{"earlier_round","earlier_seat"})
        {
            JObject composition;var source=Fixture(count,intro,!intro,2,boundary,out composition);
            var view=(JObject)Wire.Token(source.Observe(source.Actor));string original=Wire.Hash(source.Canonical());
            var profile=ProfileCatalog.Create("balanced",count,intro);profile.work_limit=1024;
            var decision=new Planner(composition).Decide(view,source.Choices(),profile,true);
            foreach(var plan in decision.alternatives)
            {
                Check(!plan.ledger.win_share.HasValue,"nonfinal ordinary turn retains heuristic units");
                var result=Apply(source,composition,plan,false);var after=(JObject)Wire.Token(result.Observe(source.Actor));
                Check(result.Actor==source.Actor && result.Remaining==0 && !result.Terminal,"nonfinal plan stops at current turn action boundary");
                Check((int)after["deck_count"]==(int)view["deck_count"] && (int)after["context"]["round"]==(int)view["context"]["round"],"no future draws or round transition in nonfinal plans");
            }
            Check(Wire.Hash(source.Canonical())==original,"nonfinal search leaves live game unchanged");
        }
        foreach(int count in new[]{2,3,4})
        {
            JObject composition;var source=Fixture(count,false,false,2,"final",out composition);
            var profile=ProfileCatalog.Create("balanced",count);profile.work_limit=1024;
            var decision=new Planner(composition).Decide((JObject)Wire.Token(source.Observe(source.Actor)),source.Choices(),profile,true);
            Check(decision.alternatives.All(p=>!p.ledger.win_share.HasValue),"full game's final Canal turn is not terminal");
        }
    }
    static void MultipleDebtors()
    {
        foreach(int count in new[]{3,4})
        {
            JObject composition;var source=Fixture(count,true,false,1,"final",out composition);
            int actor=source.Actor;var view=(JObject)Wire.Token(source.Observe(actor));
            var prototype=new Host(1701,new JObject { ["configuration"]=Configuration(count,true) });
            var tiles=(JObject)Wire.Token(prototype.Observe(prototype.Actor));
            Action<int,string> put=(seat,type)=>{
                var tile=Wire.Player(tiles,seat)["mat"].Single(m=>(string)m["type"]==type)["tiles"][0].DeepClone();
                var mat=(JArray)Wire.Player(view,seat)["mat"].Single(m=>(string)m["type"]==type)["tiles"];
                var existing=mat.FirstOrDefault(t=>(string)t["id"]==(string)tile["id"]);if(existing!=null)mat.Remove(existing);
                tile["flipped"]=true;
                view["slots"].First(s=>(s["tile"]==null || s["tile"].Type==JTokenType.Null) && s["types"].Any(t=>(string)t==type))["tile"]=tile;
            };
            foreach(var player in view["players"]){player["cash"]=0;player["income_marker"]=0;player["income"]=-10;player["vp"]=40;}
            put(actor,"CoalMine");put(actor,"IronWorks");put(actor,"Brewery");
            int second=(int)view["players"].First(p=>(int)p["seat"]!=actor)["seat"];
            put(second,"IronWorks");put(second,"Pottery");
            int noTiles=(int)view["players"].First(p=>(int)p["seat"]!=actor && (int)p["seat"]!=second)["seat"];
            source=new Host(1,Belief.Setup(view,composition,2));view=(JObject)Wire.Token(source.Observe(actor));
            string original=Wire.Hash(source.Canonical());
            var profile=ProfileCatalog.Create("balanced",count,true);profile.work_limit=1024;
            var decision=new Planner(composition).Decide(view,source.Choices(),profile,true);
            var result=Apply(source,composition,decision.selected,false);result.Advance();
            int liquidations=0;var debtors=new HashSet<int>();int maximum=source.Slots.Count(s=>s.Value.BuildedIndustry!=null);
            while(result.Phase=="debt")
            {
                Check(++liquidations<=maximum,"each forced settlement step removes an industry");debtors.Add(result.Actor);
                var publicState=(JObject)Wire.Token(result.Observe(actor));
                var choice=result.Choices().OrderBy(c=>{
                    var tile=publicState["slots"].Single(s=>(string)s["id"]==c.target)["tile"];
                    return ((int)tile["level"]==2?2:1)*((int)tile["vp"]+(int)tile["link_vp"]);
                }).ThenBy(c=>c.target,StringComparer.Ordinal).First();
                result.SearchApply(choice.id);
            }
            Check(result.Terminal && liquidations>=3 && debtors.Count==2,"final settlement covers multiple industries and players");
            var final=(JObject)Wire.Token(result.Observe(actor));
            Check((int)Wire.Player(final,noTiles)["vp"]==20,"tileless debtor pays VP shortfall then final introductory income score");
            var ledger=Evaluator.Evaluate(final,actor);
            Check(decision.selected.ledger.win_share==ledger.win_share && decision.selected.ledger.rank==ledger.rank && decision.selected.ledger.banked==ledger.banked,"modeled multiple-debtor settlement matches completed public result");
            // The vendor's Canal-era cleanup rebuilds its unused deck even for an
            // Introductory game end. It must not deal a new hand or start a turn.
            Check((int)final["context"]["round"]==(int)view["context"]["round"] && result.Actor==-1 &&
                final["players"].All(p=>(int)p["hand_count"]==0),"forced settlement never advances to an ordinary turn or deals future cards");
            Check(Wire.Hash(source.Canonical())==original,"multiple-debtor search leaves live state unchanged");
        }
    }
    public static int Main(string[] args)
    {
        try { FinalOutcomes();NonFinalBoundaries();MultipleDebtors();File.WriteAllText(args[0],Wire.Json(new { passed=true,checks }));Console.WriteLine("PASS "+checks);return 0; }
        catch(Exception error) { Console.Error.WriteLine(error);if(args.Length>0)File.WriteAllText(args[0],Wire.Json(new { passed=false,checks,error=error.ToString() }));return 1; }
    }
}
