// Trusted setup/RNG adapter and qualification oracle. Production native lanes
// call init once per frozen job. No per-transition C# rule call is required.
using System;
using System.Linq;
using BrassHost;
using Newtonsoft.Json.Linq;
using BoardGameRules.Utils;
using BoardGameRules.Entities.Players;

public static class BrassLabBootstrap
{
    static Host host;
    static string[] tileIds;
    static JObject Export()
    {
        var full=(JObject)Wire.Token(host.Canonical());var view=(JObject)full["observation"];
        var context=(JObject)Wire.Token(host.Context());context.Remove("game_id");context["chance"]=false;
        var players=new JArray();
        foreach(JObject player in view["players"]){var row=(JObject)player.DeepClone();
            foreach(string key in new[]{"hand_count","discards","hidden_discards","known_wild"})row.Remove(key);players.Add(row);}
        var used=players.SelectMany(p=>p["mat"]).SelectMany(m=>m["tiles"]).Select(t=>(string)t["id"])
            .Concat(view["slots"].Where(s=>s["tile"] is JObject).Select(s=>(string)s["tile"]["id"]))
            .Concat(full["removed"].Select(t=>(string)t)).ToArray();
        bool debt=host.Game.GameFlow.IsSellIndustryNecessary();
        var result=new JObject{["format"]="brass-native-state/1.0.0",["rules"]=Wire.Rules,["context"]=context,
            ["players"]=players,["cards"]=full["cards"].DeepClone(),["current_player"]=(int)host.Player.Color,
            ["round_pending"]=full["round_pending"].DeepClone(),["debt_pending"]=debt,
            ["settlement_debts"]=new JArray(Enumerable.Range(0,4).Select(s=>debt?host.Game.SellIndustryFlow._moneyToPayBySelling[(PlayerColor)s]:0)),
            ["removed"]=full["removed"].DeepClone(),["retired"]=new JArray(tileIds.Except(used).OrderBy(x=>x,StringComparer.Ordinal)),
            ["chance"]=JValue.CreateNull(),["result"]=full["result"].DeepClone()};
        foreach(string key in new[]{"order","slots","links","merchants","coal_market","iron_market","pending","history"})result[key]=view[key].DeepClone();
        return result;
    }
    public static int Main()
    {
        string line;
        while((line=Console.ReadLine())!=null)try{
            var r=JObject.Parse(line);string op=(string)r["op"];object value;
            if(op=="init"){
                host=new Host((int)r["seed"]);
                tileIds=((JObject)Wire.Token(host.Observe(0)))["players"].SelectMany(p=>p["mat"]).SelectMany(m=>m["tiles"]).Select(t=>(string)t["id"]).ToArray();
                host.EnableLocalScoring();
                var state=Export();var saved=host.Game.Randomer.RandomerInitData;
                var clone=new Randomer(new RandomerData(saved.Seed,saved.FastForwardCount));
                value=new{state,rail_keys=Enumerable.Range(0,64).Select(_=>clone.Next()).ToArray(),rng=saved};
            }else if(op=="shuffle"){
                int count=(int)r["count"];if(count<0 || count>64)throw new Exception("shuffle size");
                var rng=new Random((int)r["seed"]);value=Enumerable.Range(1,Math.Max(0,count-1)).Reverse().Select(i=>rng.Next(i+1)).ToArray();
            }else if(op=="decision")value=host.Decision();
            else if(op=="step"){host.Apply(host.Revision,(string)r["choice"]);value=host.Decision();}
            else if(op=="checkpoint")value=host.Checkpoint();
            else if(op=="restore"){
                host=Host.Restore((JObject)r["checkpoint"]);host.ShuffleTrainingDeck((int?)r["shuffle_seed"]);host.EnableLocalScoring(true);value=host.Decision();
            }else if(op=="state")value=Export();
            else if(op=="scoring")value=host.LocalScoringTargets();
            else throw new Exception("unknown operation");
            Console.WriteLine(Wire.Json(new{ok=true,data=value}));
        }catch(Exception e){Console.WriteLine(Wire.Json(new{ok=false,error=e.ToString()}));}
        return 0;
    }
}
