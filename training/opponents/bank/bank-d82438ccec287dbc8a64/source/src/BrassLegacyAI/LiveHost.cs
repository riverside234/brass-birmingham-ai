using System;
using System.Linq;
using System.Collections.Generic;
using BoardGameRules.Entities.EventSourcing.Events;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.GamePhases;
using BrassApplication.Game;
namespace BrassHost
{
    public sealed partial class Host
    {
        // Game-thread observer, also usable after vendor save/load. Seed is used ONLY
        // to bind vendor tile identities to public semantic IDs, never exported.
        public static Host Capture(BrassApplicationGame app,IEnumerable<IEvent> persisted=null)
        {
            if((int)app.Metadata.GameStartConfiguration.NumberOfPlayers!=4)throw new Exception("four_players_required");
            var h=new Host(app.Metadata.GameStartConfiguration.Seed);
            h.Game=app.Game;h.Revision=app.Game.Revision;h.GameId=app.Metadata.GameId??"local";
            h.turnOwner=(int)h.Player.Color;
            var external=h.Seats.SelectMany(p=>Wire.Types.SelectMany(t=>p.Mat.GetIndustryList(t))).Concat(h.Slots.Where(s=>s.Value.BuildedIndustry!=null).Select(s=>s.Value.BuildedIndustry)).Concat(h.Game.Board.RemovedIndustries).GroupBy(t=>t.UniqueId).ToDictionary(g=>g.Key,g=>g.First());
            foreach(var key in h.tileCatalog.Keys.ToArray())
            {
                var id=h.tileCatalog[key].UniqueId;
                // Developed tiles may no longer be retained by the vendor. Unused
                // prototype entries stay private; every visible tile must still bind.
                if(external.ContainsKey(id))h.tileCatalog[key]=external[id];
            }
            if(external.Keys.Any(id=>!h.tileIds.ContainsKey(id)))throw new Exception("live_tile_identity_mismatch");
            string era=GameEra.CanalEra.ToString();int[] discards=new int[4];
            var events=(persisted??Enumerable.Empty<IEvent>()).Concat(h.Game.GetUncommittedEvents()).Where(e=>e.Metadata.Revision<=h.Game.Revision).GroupBy(e=>e.Metadata.Revision).Select(g=>g.Last()).OrderBy(e=>e.Metadata.Revision);
            foreach(var e in events)
            {
                if(e is RailEraStarted){era=GameEra.RailEra.ToString();Array.Clear(discards,0,4);}
                var card=e as CardDiscarded;
                if(card!=null)
                {
                    string name=Wire.Card(card.CardToDiscard);
                    if(!name.StartsWith("wild:"))discards[(int)card.Player]++;
                    h.publicHistory.Add(new{seat=(int)card.Player,era,choice=new Choice("pass",card:name)});
                }
                var scout=e as WildCardsTaken;
                if(scout!=null){discards[(int)scout.Player]+=2;h.publicHistory.Add(new{seat=(int)scout.Player,era,choice=new Choice("scout")});}
            }
            if(h.Game.GameProgressInformation.CurrentGameEra==GameEra.RailEra)
                h.discardStarts=h.Seats.Select(p=>p.DiscardPile.Count-discards[(int)p.Color]).ToArray();
            if(h.Game.GameFlow.IsSellIndustryNecessary() && !h.Game.SellIndustryFlow.CanIndustrySellingBeCompleted())h.Phase="debt";
            return h;
        }
        public void LiveCommit(string[] path,JObject composition)
        {
            if(Phase!="action" || Game.TurnFlow.CurrentAction!=null)throw new Exception("live_action_boundary");
            var view=(JObject)Wire.Token(Observe(Actor));
            var trial=new Host(1,BrassLegacyAI.Belief.Setup(view,composition,1));
            for(int i=0;i<path.Length;i++){trial.SearchApply(path[i]);if(trial.ActionComplete && i<path.Length-1)throw new Exception("multiple_actions");}
            if(!trial.ActionComplete)throw new Exception("incomplete_live_action");
            // Keep game-thread mutations here. On any unexpected engine failure the
            // bridge stops visibly; it never pretends CancelCurrentAction is rollback.
            foreach(var choice in path)SearchApply(choice);
        }
    }
}
