using System;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using BrassHost;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.NumericalResources.Events;

public static class LocalScoringChecks
{
    static object Field(object value,string name) {return value.GetType().GetField(name,BindingFlags.NonPublic|BindingFlags.Instance).GetValue(value);}
    static BaseIndustry Place(Host h,string slot,int seat,BuildingType type,int level,bool flipped)
    {
        var p=h.Seats.ElementAt(seat);var list=p.Mat.GetIndustryList(type);var tile=list.First(t=>t.Level==level);
        list.Remove(tile);tile.IsFlipped=flipped;h.Slots.Single(s=>s.Key==slot).Value.BuildedIndustry=tile;return tile;
    }
    static void Need(bool value,string message) {if(!value)throw new Exception(message);}
    public static int Main()
    {
        try
        {
            var h=new Host(153);h.EnableLocalScoring();
            var removed=Place(h,"Birmingham:0",0,BuildingType.CoalMine,1,true);
            var survivor=Place(h,"Birmingham:1",1,BuildingType.IronWorks,2,true);
            var zero=Place(h,"Birmingham:2",2,BuildingType.CoalMine,2,true);
            // Authored zero-VP edge case, still scored by the real native rule.
            typeof(BaseIndustry).GetField("<VPForBuilding>k__BackingField",BindingFlags.Instance|BindingFlags.NonPublic).SetValue(zero,0);
            Place(h,"Birmingham:3",3,BuildingType.CottonMill,2,false);
            Place(h,"Derby:0",0,BuildingType.CoalMine,2,true);
            var replacement=Place(h,"Derby:0",3,BuildingType.IronWorks,2,true);
            var eliminated=Place(h,"Derby:1",2,BuildingType.CoalMine,1,true);
            var debtor=h.Seats.ElementAt(2);
            h.Game.ApplyEvent(new MoneySpent(debtor.Color,debtor.Money-9));
            h.Game.ApplyEvent(new IncomeMarkerPositionChanged(debtor.Color,-debtor.IncomeMarkerPosition,"fixture"));
            h.Game.SellIndustryFlow.StartIndustrySelling();
            Need(h.Game.SellIndustryFlow.RemainingDebtToPay(debtor.Color)>0,"Liquidation fixture has no debt");
            h.Game.SellIndustryFlow.SellIndustry(debtor.Color,h.Slots.Single(s=>s.Key=="Derby:1").Value);
            Need(h.Slots.Single(s=>s.Key=="Derby:1").Value.BuildedIndustry==null,"Native liquidation did not remove tile");
            var canal=h.Links.Where(l=>l.GetType().Name=="Canal").ToArray();
            for(int j=0;j<canal.Length;j++)canal[j].SetOwner(h.Seats.ElementAt(j%4));
            ((CanalEra)Field(h.Game.GameFlow,"_canalEra")).EndCanalEra(false);
            Need(h.Slots.Single(s=>s.Key=="Birmingham:0").Value.BuildedIndustry==null,"Canal obsolete industry survived");
            Need(h.Slots.Single(s=>s.Key=="Birmingham:1").Value.BuildedIndustry==survivor,"Level-two industry removed");
            Need(canal.All(l=>!l.HasOwner),"Canal links survived cleanup");
            h.Game.GameProgressInformation.CurrentGameEra=GameEra.RailEra;
            var rail=h.Links.Where(l=>l.GetType().Name=="Rail").ToArray();
            for(int j=0;j<rail.Length;j++)rail[j].SetOwner(h.Seats.ElementAt(j%4));
            var cash=h.Seats.Select(p=>p.Money).ToArray();
            ((RailEra)Field(h.Game.GameFlow,"_railEra")).EndRailEra();
            Need(cash.SequenceEqual(h.Seats.Select(p=>p.Money)),"Final Rail scoring performed income settlement");
            // Direct authored scoring transactions do not drive the round-flow
            // terminal flag. Inspect the private receipt only in this fixture;
            // the production RPC still rejects incomplete games.
            var recorder=Field(h,"localRecorder");
            var result=(JObject)Wire.Token(recorder.GetType().GetMethod("Result").Invoke(recorder,null));var first=result["eras"][0];var last=result["eras"][1];
            Func<JToken,string,JToken> slot=(era,id)=>era["slots"].Single(s=>(string)s["id"]==id);
            Need((int)slot(first,"Birmingham:2")["owner"]==3 && (int)slot(first,"Birmingham:2")["vp"]==0,"Zero VP lost owner");
            Need((int)slot(first,"Birmingham:3")["owner"]==0,"Unflipped industry got owner");
            Need((int)slot(first,"Derby:0")["owner"]==4,"Overbuild target used old owner");
            Need((int)slot(first,"Derby:1")["owner"]==0,"Removed industry scored");
            Need((int)slot(first,"Birmingham:0")["owner"]==1 && (int)slot(last,"Birmingham:0")["owner"]==0,"Canal cleanup contaminated receipt");
            Need((int)slot(last,"Birmingham:1")["owner"]==2,"Surviving industry lost Rail label");
            Need(rail.Any(l=>l.Region3!=null),"Three-endpoint fixture missing");
            Console.WriteLine(Wire.Json(result));return 0;
        }
        catch(Exception e){Console.Error.WriteLine(e);return 1;}
    }
}
