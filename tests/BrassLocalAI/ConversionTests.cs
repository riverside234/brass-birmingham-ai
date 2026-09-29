using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class ConversionTests
{
    static int checks;static string root;static readonly List<object> cases=new List<object>();
    static readonly string[] keys={"balanced","rail-tempo","cotton","goods-beer","supplier","pottery-flex"};
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static JObject Composition(){return JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/contracts/v0.2.0/card-composition.json")));}
    static Profile Profile(string key){return BrassLegacyAI.Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/"+key+"-v3.json"))));}
    static readonly Dictionary<string,JObject> positions=new Dictionary<string,JObject>();
    static JObject Position(bool rail,int turn=1,int actions=1,int round=8)
    {
        string key=rail+"/"+turn+"/"+actions+"/"+round;JObject cached;
        if(positions.TryGetValue(key,out cached))return (JObject)cached.DeepClone();
        var h=new Host(2701);
        for(int i=0;i<150;i++)
        {
            var v=(JObject)Wire.Token(h.Observe(h.Actor));
            if((string)v["context"]["era"]==(rail?"RailEra":"CanalEra") && (int)v["context"]["round"]==round && (int)v["context"]["turn"]==turn && h.Remaining==actions)
            {
                Conversion.Player(v,h.Actor)["cash"]=50;Conversion.Player(v,h.Actor)["income_marker"]=20;
                positions[key]=(JObject)v.DeepClone();return v;
            }
            h.SearchApply(h.Choices().First(c=>c.kind=="pass").id);h.SearchApply(h.Choices().Single(c=>c.kind=="finish").id);h.Advance();
        }
        throw new Exception("fixture boundary");
    }
    static JToken Put(JObject v,string id,string type,int level,int resource=-1)
    {
        var mat=(JArray)Conversion.Player(v,(int)v["seat"])["mat"].Single(m=>(string)m["type"]==type)["tiles"];
        var tile=mat.First(t=>(int)t["level"]==level);int index=mat.IndexOf(tile);var copy=tile.DeepClone();
        for(int i=0;i<=index;i++)mat.RemoveAt(0);
        copy["flipped"]=false;if(resource>=0)copy["resource"]=resource;
        var slot=v["slots"].Single(s=>(string)s["id"]==id);slot["tile"]=copy;return slot;
    }
    static void Routes(JObject v,string[] regions,string merchantRegion,string type)
    {
        foreach(var link in v["links"].Where(l=>((string)l["id"]).StartsWith(((string)v["context"]["era"]).Replace("Era","")) && l["regions"].All(r=>regions.Contains((string)r))))link["owner"]=(int)v["seat"];
        foreach(var m in v["merchants"]){m["types"]=new JArray(type);m["beer"]=Evaluator.Region((string)m["id"])==merchantRegion?1:0;}
    }
    static JObject Replay(Host h,Plan plan)
    {
        var copy=new Host(1,Belief.Setup((JObject)Wire.Token(h.Observe(h.Actor)),Composition(),7));
        foreach(var a in plan.actions)foreach(string id in a){Check(copy.Choices().Any(c=>c.id==id),"every continuation legal");copy.SearchApply(id);}
        Check(copy.Remaining==0 && copy.ActionComplete,"complete current turn only");return (JObject)Wire.Token(copy.Observe(h.Actor));
    }
    static void DeadlineSales()
    {
        foreach(bool rail in new[]{false,true})foreach(string key in keys)
        {
            var v=Position(rail);Put(v,"Worcester:0","CottonMill",rail?2:1);
            Routes(v,new[]{"Worcester","Gloucester"},"Gloucester","CottonMill");
            var h=new Host(1,Belief.Setup(v,Composition(),7));v=(JObject)Wire.Token(h.Observe(h.Actor));
            string original=Wire.Hash(h.Canonical());var p=Profile(key);p.work_limit=240;
            var d=new Planner(Composition()).Decide(v,h.Choices(),p,true);var after=Replay(h,d.selected);
            Check((bool)after["slots"].Single(s=>(string)s["id"]=="Worcester:0")["tile"]["flipped"],"last own turn sells valuable cotton: "+key+" "+rail+" "+d.selected.signature);
            Check(d.failures==0 && original==Wire.Hash(h.Canonical()),"planning isolated and successful");
            var unsold=(JObject)v.DeepClone();unsold["context"]["actions_remaining"]=0;
            var slot=unsold["slots"].Single(s=>(string)s["id"]=="Worcester:0");
            Check(Conversion.Completion(unsold,slot,h.Actor)==0,"possible sale earns zero after own last action, even before other seats finish");
            cases.Add(new{scenario="deadline-sale",rail,key,d.selected.signature});
        }
    }
    static void ResourceDeadlines()
    {
        foreach(bool rail in new[]{false,true})
        {
            var v=Position(rail,4);int seat=(int)v["seat"];
            var brewery=Put(v,"BurtonOnTrent:0","Brewery",rail?2:1,rail?2:1);
            var iron=Put(v,"Birmingham:0","IronWorks",rail?2:1,3);
            v["context"]["actions_remaining"]=0;
            Check(Conversion.Completion(v,brewery,seat)==0 && Conversion.Completion(v,iron,seat)==0,"no imaginary resource consumption after every player finished");
            Check(Conversion.IndustryValue(v,brewery,seat)==0,"unflipped expiring resources have no industry VP");
            brewery["tile"]["flipped"]=true;
            Check(Conversion.IndustryValue(v,brewery,seat)==(int)brewery["tile"]["vp"],"flipped expiring resource scores exactly once");
        }
        var canal=Position(false,1);int owner=(int)canal["seat"];
        var carry=Put(canal,"Worcester:0","CottonMill",2);canal["context"]["actions_remaining"]=0;
        Check(Conversion.Completion(canal,carry,owner)==0 && Conversion.IndustryValue(canal,carry,owner)>0,"level two carry-over has only discounted next-era value");
        Check(Conversion.Actions(canal,(int)canal["order"][1])==2,"later seat retains its actual final turn");
        Check(Conversion.Actions(canal,owner)==0,"finished owner has no final action");
    }
    static void BeerChoices()
    {
        foreach(string key in keys)foreach(string bonusRegion in new[]{"Gloucester","Warrington","Shrewsbury","Nottingham","Oxford"})
        {
            var v=Position(true,1,1,6);int seat=(int)v["seat"];
            Conversion.Player(v,seat)["cash"]=0;Conversion.Player(v,seat)["income_marker"]=0;
            Put(v,"Worcester:0","CottonMill",2);
            // One sale needs one beer; a full Rail brewery has two. Merchant
            // beer gives the bonus and preserves beer without forfeiting a flip.
            Put(v,"BurtonOnTrent:0","Brewery",2,2);
            if(bonusRegion=="Gloucester")
            {
                // Level one pottery was used in Canal and removed at the era
                // boundary. Free development can now remove its weak level two
                // and unlock level three. This makes the bonus clearly useful,
                // rather than requiring merchant beer for a marginal upgrade.
                var mat=(JArray)Conversion.Player(v,seat)["mat"].Single(m=>(string)m["type"]=="Pottery")["tiles"];
                mat.RemoveAt(0);
            }
            int n=0;foreach(var link in v["links"].Where(l=>((string)l["id"]).StartsWith("Rail")))link["owner"]=n++%4;
            if(bonusRegion=="Gloucester")foreach(var link in v["links"].Where(l=>((string)l["id"]).StartsWith("Rail") && l["regions"].Any(r=>(string)r=="BurtonOnTrent")))
                link["owner"]=(seat+1)%4; // No own-link windfall from draining this brewery.
            foreach(var merchant in v["merchants"]){merchant["types"]=new JArray("CottonMill");merchant["beer"]=Evaluator.Region((string)merchant["id"])==bonusRegion?1:0;}
            var h=new Host(1,Belief.Setup(v,Composition(),7));v=(JObject)Wire.Token(h.Observe(seat));
            var p=Profile(key);p.work_limit=240;var d=new Planner(Composition()).Decide(v,h.Choices(),p,true);
            var after=Replay(h,d.selected);string[] path=d.selected.actions.SelectMany(a=>a).ToArray();
            Check(path.Any(id=>id.StartsWith("beer|merchant:"+bonusRegion+":")),"merchant beer and bonus are considered: "+key+" "+bonusRegion+" "+d.selected.signature);
            Check((int)after["slots"].Single(s=>(string)s["id"]=="BurtonOnTrent:0")["tile"]["resource"]==2,"use merchant beer without wasting own stock");
            int beforeVp=(int)Conversion.Player(v,seat)["vp"],afterVp=(int)Conversion.Player(after,seat)["vp"];
            if(bonusRegion=="Shrewsbury" || bonusRegion=="Nottingham")Check(afterVp-beforeVp==(bonusRegion=="Shrewsbury"?4:3),"official merchant VP awarded exactly once");
            Check(Evaluator.Seat(after,seat,true).banked==afterVp,"bonus counter is not double counted");
            if(bonusRegion=="Gloucester")Check((int)Conversion.Player(after,seat)["mat"].Single(m=>(string)m["type"]=="Pottery")["tiles"][0]["level"]==3,"free merchant development unlocks the useful pottery upgrade");
            cases.Add(new{scenario="merchant-beer",key,bonusRegion,d.selected.signature});Console.WriteLine("Merchant check "+key+" "+bonusRegion);
        }
        foreach(string key in keys)
        {
            var v=Position(false);int seat=(int)v["seat"];
            Put(v,"Worcester:0","CottonMill",1);Put(v,"Kidderminster:0","CottonMill",1);Put(v,"BurtonOnTrent:0","Brewery",1,1);
            Routes(v,new[]{"Worcester","Kidderminster","SouthWest","Gloucester"},"Gloucester","CottonMill");
            // Only one merchant barrel: batching both sales must also drain the
            // last barrel of our level-one brewery, flipping all three tiles.
            var merchants=v["merchants"].Where(m=>Evaluator.Region((string)m["id"])=="Gloucester").ToArray();
            foreach(var m in merchants.Skip(1))m["beer"]=0;
            var h=new Host(1,Belief.Setup(v,Composition(),7));var p=Profile(key);p.work_limit=240;
            var d=new Planner(Composition()).Decide((JObject)Wire.Token(h.Observe(seat)),h.Choices(),p,true);var after=Replay(h,d.selected);
            foreach(string id in new[]{"Worcester:0","Kidderminster:0","BurtonOnTrent:0"})Check((bool)after["slots"].Single(s=>(string)s["id"]==id)["tile"]["flipped"],"batch sale flips cotton and consumes expiring brewery: "+id+" "+d.selected.signature);
            cases.Add(new{scenario="batch-sale-and-brewery",key,d.selected.signature});
        }
        foreach(string key in keys)
        {
            var v=Position(true,4);int seat=(int)v["seat"];
            Put(v,"Worcester:0","CottonMill",2);Put(v,"BurtonOnTrent:0","Brewery",2,1);
            int n=0;foreach(var link in v["links"].Where(l=>((string)l["id"]).StartsWith("Rail")))link["owner"]=n++%4;
            foreach(var m in v["merchants"]){m["types"]=new JArray("CottonMill");m["beer"]=Evaluator.Region((string)m["id"])=="Shrewsbury"?1:0;}
            var h=new Host(1,Belief.Setup(v,Composition(),7));var p=Profile(key);p.work_limit=240;
            var d=new Planner(Composition()).Decide((JObject)Wire.Token(h.Observe(seat)),h.Choices(),p,true);var after=Replay(h,d.selected);
            Check((bool)after["slots"].Single(s=>(string)s["id"]=="BurtonOnTrent:0")["tile"]["flipped"],"last brewery barrel can outweigh merchant VP bonus");
            Check(d.selected.ledger.win_share.HasValue,"tradeoff uses official final outcome");
            cases.Add(new{scenario="brewery-over-merchant",key,d.selected.signature});
        }
    }
    public static int Main(string[] args)
    {
        root=args[1];try{DeadlineSales();ResourceDeadlines();BeerChoices();File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cases}));Console.WriteLine("Conversion PASS "+checks);return 0;}
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,cases,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
