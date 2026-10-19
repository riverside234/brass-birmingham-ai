// Original project adapter. Requires the user's locally installed, fingerprinted game.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Industries;

namespace BrassHost
{
    public sealed class Choice
    {
        public string id, kind, card, target, industry, merchant;
        public string[] cards;
        public Choice(string kind, string target = null, string industry = null, string card = null, string merchant = null, string[] cards = null)
        {
            this.kind=kind; this.target=target; this.industry=industry; this.card=card; this.merchant=merchant; this.cards=cards;
            // IDs contain semantic, public/own-hand data only; never vendor GUIDs or seed.
            id=string.Join("|", new[]{kind,target??"",industry??"",card??"",merchant??"",cards==null?"":string.Join(",",cards)});
        }
    }
    public static class Wire
    {
        public const string Version="0.2.0";
        public const string Rules="steam-brass-patch-1.1.0";
        public static readonly JsonSerializerSettings Settings=new JsonSerializerSettings { NullValueHandling=NullValueHandling.Ignore, TypeNameHandling=TypeNameHandling.None };
        public static string Json(object value) { return JsonConvert.SerializeObject(value, Formatting.None, Settings); }
        public static JToken Token(object value) { return JToken.Parse(Json(value)); }
        public static string Hash(object value) { using(var h=SHA256.Create()) return BitConverter.ToString(h.ComputeHash(Encoding.UTF8.GetBytes(Json(value)))).Replace("-","").ToLowerInvariant(); }
        public static string Card(BaseCard c)
        {
            if(c is LocationCard) return "location:"+((LocationCard)c).Region;
            if(c is IndustryCard) return "industry:"+string.Join("+",((IndustryCard)c).BuildingTypes.Select(x=>x.ToString()).OrderBy(x=>x,StringComparer.Ordinal));
            if(c is WildIndustryCard) return "wild:industry";
            if(c is WildLocationCard) return "wild:location";
            throw new InvalidOperationException("Unsupported card type");
        }
        public static BuildingType Type(string s) { return (BuildingType)Enum.Parse(typeof(BuildingType),s); }
        public static readonly BuildingType[] Types=Enum.GetValues(typeof(BuildingType)).Cast<BuildingType>().Where(x=>x!=BuildingType.Undefined).OrderBy(x=>x.ToString(),StringComparer.Ordinal).ToArray();
        public static object Tile(BaseIndustry t)
        {
            if(t==null) return null;
            return new {seat=(int)t.Color,type=t.BuildingType.ToString(),level=t.Level,flipped=t.IsFlipped,resource=t.GetResource(),vp=t.VPForBuilding,link_vp=t.VPPerLink,income=t.IncomeIncrease,cost=t.MoneyCost,coal=t.CoalCost,iron=t.IronCost,beer=t.GetBeerNeeded(),developable=t.CanBeDeveloped,canal_only=t.IsForCanalEra,rail_only=t.IsForRailEra};
        }
    }
}
