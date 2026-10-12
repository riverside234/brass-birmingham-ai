using System;
using System.Linq;
using BoardGameRules.Entities.Players;
using Newtonsoft.Json.Linq;

namespace BrassHost
{
    public sealed partial class Host
    {
        public void ValidatePersonalityAction(string[] path,JObject composition) {
            if(Phase!="debt"){ValidateLiveAction(path,composition);return;}
            if(path==null || path.Length!=1 || !Choices().Any(c=>c.kind=="liquidate" && c.id==path[0]))
                throw new InvalidOperationException("Illegal neural liquidation");
        }
        public void CommitPersonalityAction(string[] path,JObject composition) {
            ValidatePersonalityAction(path,composition);
            if(Phase!="debt"){LiveCommit(path,composition);return;}
            // Preserve the game's round/save UI flow; Host.Apply would also
            // advance the detached simulator, which is not a live UI action.
            var c=Choices().Single(x=>x.id==path[0]);Game.SellIndustryFlow.SellIndustry((PlayerColor)Actor,Slot(c.target));
        }
    }
}
