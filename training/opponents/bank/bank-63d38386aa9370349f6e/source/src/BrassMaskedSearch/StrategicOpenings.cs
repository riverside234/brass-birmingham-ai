using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Internal keys remain stable for saved assignments. Conditional books and
    // public strategy names are attributed in ai-research/TOURNAMENT_STYLE_V3.md.
    public static class StrategicOpenings
    {
        public static string Label(string key)
        {
            switch(key){case "cotton":return "Three Cotton";case "goods-beer":return "Goods & Beer";
                case "rail-tempo":return "Rail & Pottery";case "supplier":return "Standard Iron & Beer";
                case "pottery-flex":return "Pottery Variants";default:return "Adaptive Opening Book";}
        }
        public static IntentState Intent(JObject v,Profile p){return TournamentOpenings.Intent(v,p);}
    }
}
