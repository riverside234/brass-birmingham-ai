using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Internal keys remain stable for saved assignments. Conditional books and
    // public strategy names are attributed in ai-research/TOURNAMENT_STYLE_V3.md.
    public static class StrategicOpenings
    {
        public static string Label(string key)
        {
            switch(key){case "cotton":return "External-Beer Three Cotton";case "goods-beer":return "Worcester Iron-Cotton";
                case "rail-tempo":return "Coventry Goods Opening";case "supplier":return "Zero-Loan Industry";
                case "pottery-flex":return "Advanced Pottery Variants";default:return "Adaptive Advanced Book";}
        }
        public static IntentState Intent(JObject v,Profile p){return TournamentOpenings.Intent(v,p);}
    }
}
