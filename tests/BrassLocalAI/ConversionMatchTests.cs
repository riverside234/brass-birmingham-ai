using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class ConversionMatchTests
{
    static int checks;
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static object Match(int seed,string style,bool v3,int work,bool timed=false)
    {
        var h=new Host(seed);var composition=Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));var planner=new Planner(composition);
        var eraReports=new List<object>();var merchantBeerUses=new int[4];
        var intents=new Dictionary<int,IntentState>();var turns=new int[4];var zeroCash=new int[4];var creditFloor=new int[4];
        int actions=0,liquidations=0,shortfalls=0,avoidableShortfalls=0;var watch=System.Diagnostics.Stopwatch.StartNew();
        while(!h.Terminal)
        {
            Check(turns.Sum()<65 && liquidations<300,"four-player game makes bounded progress");
            int seat=h.Actor;bool debt=h.Phase=="debt";
            string key=style=="mixed"?ProfileCatalog.Keys[(seat+seed)%6]:style;
            var profile=v3?ProfileCatalog.Create(key,4):Profile.Read(ProfileCatalog.Snapshot(key,4,false,ProfileCatalog.CashRuntime));
            profile.work_limit=work;
            profile.milliseconds=10000;
            var view=(JObject)Wire.Token(h.Observe(seat));var before=Wire.Player(view,seat);var legal=h.Choices();
            int income=(int)before["income"],cash=(int)before["cash"];
            bool safeAvailable=cash+income>=0 || (legal.Any(c=>c.kind=="loan") && cash+30+income-3>=0);
            IntentState prior;intents.TryGetValue(seat,out prior);
            var decision=planner.Decide(view,legal,profile,!timed,1,default(System.Threading.CancellationToken),null,prior);
            Check(decision.failures==0,"no failed candidate in complete all-AI game");
            if(debt)liquidations++;else{turns[seat]++;Check(decision.selected.actions.Count==h.Remaining,"all ordinary actions planned");}
            foreach(var action in decision.selected.actions){foreach(var id in action){if(id.StartsWith("beer|merchant:"))merchantBeerUses[seat]++;h.SearchApply(id);}if(!debt)actions++;}
            if(!debt)
            {
                intents[seat]=decision.selected.intent;var after=(JObject)Wire.Token(h.Observe(seat));var money=Wire.Player(after,seat);
                var forecast=CashManagement.Project(after,seat);
                if((int)money["cash"]==0)zeroCash[seat]++;
                if((int)money["income"]<-7)creditFloor[seat]++;
                if(forecast.shortfall>0){shortfalls++;if(safeAvailable)avoidableShortfalls++;}
                if(v3)Check(!safeAvailable || forecast.shortfall==0,"v3 retains an available payment-safe complete turn");
            }
            if(!debt && (int)view["context"]["round"]==8 && (int)view["context"]["turn"]==4)
            {
                var end=(JObject)Wire.Token(h.Observe(seat));
                eraReports.Add(new{era=(string)end["context"]["era"],tiles=end["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null).Select(s=>s["tile"].DeepClone()).ToArray()});
            }
            h.Advance();
            Check(h.Seats.Count()==4 && h.Seats.Select(p=>(int)p.Color).Distinct().Count()==4,"every round retains all four players");
        }
        Check(turns.All(n=>n==16) && actions==124,"all four AI players complete both eras and all 124 actions");
        return new{seed,style,generation=3,revised=v3,package=v3?"1.4.1":"1.4.0",eraReports,merchantBeerUses,work,timed,milliseconds=10000,turns,actions,liquidations,shortfalls,avoidableShortfalls,zeroCash,creditFloor,
            elapsed_ms=watch.ElapsedMilliseconds,result=h.Result()};
    }
    public static int Main(string[] args)
    {
        var matches=new List<object>();
        try
        {
            if(args.Length>3 && args[1]=="single")
            {
                int seed=int.Parse(args[2]);bool revised=bool.Parse(args[3]);
                Check(seed==72001 || seed==72002,"declared comparison seed");
                matches.Add(Match(seed,"mixed",revised,240));Write(args[0],matches,true);return 0;
            }
            if(args.Length>1 && args[1]=="live-budget")
            {
                matches.Add(Match(72003,"mixed",true,1024,true));Write(args[0],matches,true);return 0;
            }
            foreach(int seed in new[]{72001,72002})foreach(bool revised in new[]{false,true})
            {matches.Add(Match(seed,"mixed",revised,240));Write(args[0],matches,false);Console.WriteLine("Complete conversion comparison "+seed+" revised="+revised);}
            Write(args[0],matches,true);return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,matches,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
    static void Write(string path,List<object> matches,bool passed)
    {File.WriteAllText(path,Wire.Json(new{passed,checks,matches,scope="Four-player full games at the per-row work/time settings. Cash/reliability evidence, not a strength tournament."}));}
}
