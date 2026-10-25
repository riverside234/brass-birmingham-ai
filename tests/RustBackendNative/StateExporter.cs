// Additive trusted-only diagnostic executable. References the frozen host;
// never changes its source, the installed game, or policy-facing endpoints.
using System;
using System.Linq;
using BrassHost;
using Newtonsoft.Json.Linq;

public static class StateExporter
{
    public static JObject Export(Host host)
    {
        var full=(JObject)Wire.Token(host.Canonical());
        var view=(JObject)full["observation"];
        var context=(JObject)Wire.Token(host.SearchContext());context.Remove("game_id");
        var players=new JArray();
        foreach(JObject player in view["players"])
        {
            var row=(JObject)player.DeepClone();
            foreach(string field in new[]{"hand_count","discards","hidden_discards","known_wild"})row.Remove(field);
            players.Add(row);
        }
        var result=new JObject {
            ["format"]="brass-native-state/1.0.0", ["rules"]=Wire.Rules,
            ["context"]=context,["players"]=players,["cards"]=full["cards"].DeepClone(),
            ["current_player"]=(int)host.Player.Color,
            ["round_pending"]=full["round_pending"].DeepClone(),
            ["debt_pending"]=full["debt_pending"].DeepClone(),
            ["removed"]=full["removed"].DeepClone(),
            ["chance"]=host.ChancePending?Wire.Token(host.ChanceState()):JValue.CreateNull(),
            ["result"]=full["result"]==null?JValue.CreateNull():full["result"].DeepClone()
        };
        foreach(string key in new[]{"order","slots","links","merchants","coal_market","iron_market","pending","history"})
            result[key]=view[key].DeepClone();
        return result;
    }

    public static int Main()
    {
        string line;
        while((line=Console.ReadLine())!=null)
        {
            try
            {
                var request=JObject.Parse(line);
                Host host=request["checkpoint"]==null?new Host(890):Host.Restore((JObject)request["checkpoint"]);
                host.ExplicitChance=true;
                var snapshot=(JObject)Wire.Token(host.SearchSnapshot());
                Console.WriteLine(Wire.Json(new {ok=true,state=Export(host),snapshot}));
            }
            catch(Exception error) {Console.WriteLine(Wire.Json(new {ok=false,message=error.ToString()}));}
        }
        return 0;
    }
}
