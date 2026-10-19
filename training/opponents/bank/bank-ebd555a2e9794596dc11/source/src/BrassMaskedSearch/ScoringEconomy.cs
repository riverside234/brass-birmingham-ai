using System;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Shared by every v5 style. These are proposal priorities, never earned VP.
    public static class ScoringEconomy
    {
        public const int IncomeThreshold=8;
        public static bool Established(JObject v,int seat)
        {return (int)Conversion.Player(v,seat)["income"]>=IncomeThreshold;}
        public static double IncomeWeight(JObject v,int seat)
        {return Established(v,seat)?.15:.40;}
        public static int PreferredLevel(JObject v,string type)
        {
            switch(type)
            {
                case "CottonMill":return 4;
                case "Manufacturer":return EraEconomy.Rail(v) || Established(v,(int)v["seat"])?8:6;
                case "Pottery":return EraEconomy.Rail(v)?5:3;
                case "CoalMine":return 2;
                case "IronWorks":return 3;
                default:return 0;
            }
        }
        public static double TierPriority(JObject v,JToken tile)
        {
            if(!Conversion.Sellable(tile) || (int)tile["vp"]<5)return 1;
            int preferred=PreferredLevel(v,(string)tile["type"]),level=(int)tile["level"];
            // Pottery II/IV are low-VP stepping stones, not scoring targets.
            // A high-tier preference is only used AFTER a full funding quote.
            double progress=Math.Min(1,Math.Max(0,level-1)/(double)Math.Max(1,preferred-1));
            return 1+.50*progress+(level>=preferred?.15:0);
        }
        public static double CoalPenalty(JObject v,JToken tile,int marketSold)
        {
            int seat=(int)v["seat"];bool full=marketSold==(int)tile["resource"];
            // Lower coal preference even in a shortage. Immediate flips and
            // distressed-income recovery keep a smaller, not zero, discount.
            return full?((int)Conversion.Player(v,seat)["income"]<2?1.0:2.0):Established(v,seat)?5.0:2.5;
        }
    }
}
