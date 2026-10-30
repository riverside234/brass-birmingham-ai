using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.Events;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Industries;
using BrassApplication.Game;
using Newtonsoft.Json.Linq;

namespace BrassHost
{
    public sealed partial class Host
    {
        // Explicit local-game/0.3.0 -> public-state/0.2.0 adapter. The local
        // extension adds configurable player counts/introductory rounds. P6
        // accepts only the standard four-player/eight-round contract; the
        // unchanged Python boundary validates every remaining field.
        public JObject ObservePersonality(int seat) {
            var view=ObserveToken(seat);var c=(JObject)view["context"];
            if((string)view["protocol"]!="local-game/0.3.0" || (string)view["rules"]!="steam-brass-patch-1.3.0" ||
                (int)c["player_count"]!=4 || (bool)c["introductory"] || (int)c["round_limit"]!=8)
                throw new InvalidOperationException("Unsupported local-game neural observation contract");
            view.Remove("configuration");c.Remove("player_count");c.Remove("introductory");c.Remove("round_limit");
            view["protocol"]=c["protocol"]="0.2.0";view["rules"]=c["rules"]="steam-brass-patch-1.1.0";
            return view;
        }
        // Rebuild from confirmed public events on the trusted game thread. Raw
        // vendor events/private initial deal never cross the Python boundary.
        public static Host CapturePersonality(BrassApplicationGame app,IEnumerable<IEvent> persisted,out int[] insolvencies)
        {
            var events=BrassLegacyAI.LocalGameState.History(app,persisted??Enumerable.Empty<IEvent>());
            var live=Capture(app,events);var projection=new Host(app.Metadata.GameStartConfiguration.Seed,
                new JObject { ["configuration"]=Configuration(app.Metadata.GameStartConfiguration) });
            projection.Game=new BoardGameRulesGame(app.Metadata.GameStartConfiguration);
            var cards=new List<string>();string kind=null;int start=0;insolvencies=new int[4];
            var rounds=new HashSet<string>();
            foreach(var e in events) {
                string name=e.GetType().Name;
                var discard=e as CardDiscarded;if(discard!=null)cards.Insert(0,Wire.Card(discard.CardToDiscard));
                var scout=e as WildCardsTaken;if(scout!=null){kind="scout";cards.AddRange(scout.CardsToTrade.Select(Wire.Card));}
                if(e is IndustryBuilt)kind="build";
                if(e is LinkPlaced)kind="network";
                if(name=="LoanTaken" || name=="MoneyLoaned")kind="loan";
                if(name=="ActionPassed")kind="pass";
                // Stock resolves merchant bonuses before flipping the sold
                // industry. Identify Sell now so its free development cannot
                // be mistaken for a separate Develop action.
                if(e is MerchantBonusReceived && kind==null)kind="sell";
                if(e is IndustryDeveloped && kind==null)kind="develop";
                var flipped=e as IndustryFlipped;
                if(flipped!=null && kind==null && projection.Slots.Any(s=>s.Value.BuildedIndustry is BaseSellableIndustry &&
                    s.Value.BuildedIndustry.UniqueId==flipped.IndustryUniqueId))kind="sell";
                if(name=="IndustrySold") {
                    var player=(int)e.GetType().GetProperty("PlayerColor").GetValue(e);
                    var slot=(string)e.GetType().GetProperty("SlotUniqueId").GetValue(e);
                    projection.publicHistory.Add(new {seat=player,era=projection.Game.GameProgressInformation.CurrentGameEra.ToString(),
                        choice=new Choice("liquidate",projection.Slots.Single(s=>s.Value.UniqueId==slot).Key)});
                }
                projection.Game.ApplyEvent(e);
                if(name=="IndustrySellingStarted") {
                    string round=projection.Game.GameProgressInformation.CurrentGameEra+":"+projection.Game.GameProgressInformation.CurrentRoundNumber;
                    if(rounds.Add(round))foreach(var debt in projection.Game.SellIndustryFlow._moneyToPayBySelling)
                        if(debt.Value>0)insolvencies[(int)debt.Key]++;
                }
                if(name=="ActionCountDecreased") {
                    if(kind==null)throw new InvalidOperationException("Unrecognized confirmed public action");
                    projection.P6RecordAction(kind,cards.ToArray(),start);cards.Clear();kind=null;start=e.Metadata.Revision;
                }
                if(e is TurnStarted) {cards.Clear();kind=null;start=e.Metadata.Revision;}
            }
            // Liquidation focus changes CurrentPlayer, not the owner of the
            // round's last ordinary turn used by the model's public context.
            if(live.Phase=="debt")live.turnOwner=(int)live.Game.PlayersTrack.Players[live.Game.GameProgressInformation.CurrentTurnNumber-1].Color;
            live.publicHistory.Clear();live.publicHistory.AddRange(projection.publicHistory);
            return live;
        }
    }
}
