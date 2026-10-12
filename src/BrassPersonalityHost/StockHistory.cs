// Translate only observable stock events into the existing public-choice schema.
// No hidden cards, selector state, RNG, or arbitrary event payload is exported.
using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Industries;

namespace BrassHost
{
    public sealed partial class Host
    {
        void P5RecordStock(string kind, string[] cards, int revision)
        {
            int seat = (int)Player.Color;
            string era = Game.GameProgressInformation.CurrentGameEra.ToString();
            var choices = new List<Choice>(); bool usedCard = false;
            Func<string> takeCard = () => { if (usedCard) return null; usedCard = true; return cards.FirstOrDefault(); };
            Func<string, string> slotFor = id => Slots.Single(s => s.Value.BuildedIndustry != null && s.Value.BuildedIndustry.UniqueId == id).Key;
            if (kind == "pass" || kind == "loan") choices.Add(new Choice(kind, card: takeCard()));
            if (kind == "scout") choices.Add(new Choice(kind, card: takeCard(), cards: cards.Skip(1).ToArray()));
            int saleStart = 0; string saleMerchant = null; bool bonusDevelop = false, bonusUsed = false;
            foreach (var e in Game.GetUncommittedEvents().Where(e => e.Metadata.Revision > revision))
            {
                var build = e as IndustryBuilt;
                if (build != null) choices.Add(new Choice("build", Slots.Single(s => s.Value.UniqueId == build.SlotUniqueId).Key,
                    industry: build.Type.ToString(), card: takeCard()));
                var link = e as LinkPlaced;
                if (link != null) choices.Add(new Choice("network", link.LinkId, card: takeCard()));
                var develop = e as IndustryDeveloped;
                if (develop != null) {
                    choices.Add(new Choice(bonusDevelop ? "bonus_develop" : "develop", industry: develop.Type.ToString(),
                        card: bonusDevelop ? null : takeCard())); bonusUsed |= bonusDevelop;
                }
                var coal = e as CoalSupplied;
                if (coal != null) for (int i = 0; i < coal.Amount; i++) choices.Add(new Choice("coal",
                    coal.SourceUniqueId == Game.Board.CoalMarket.UniqueId ? "market" : slotFor(coal.SourceUniqueId)));
                var iron = e as IronSupplied;
                if (iron != null) for (int i = 0; i < iron.Amount; i++) choices.Add(new Choice("iron",
                    iron.SourceId == Game.Board.IronMarket.UniqueId ? "market" : slotFor(iron.SourceId)));
                var beer = e as BeerSupplied;
                if (beer != null) {
                    var m = Merchants.FirstOrDefault(x => x.Value.UniqueId == beer.BeerSourceId);
                    string source = m.Value == null ? slotFor(beer.BeerSourceId) : "merchant:" + m.Key;
                    if (m.Value != null) saleMerchant = m.Key;
                    for (int i = 0; i < beer.Amount; i++) choices.Add(new Choice("beer", source));
                }
                var flip = e as IndustryFlipped;
                if (kind == "sell" && flip != null) {
                    var s = Slots.Single(x => x.Value.BuildedIndustry != null && x.Value.BuildedIndustry.UniqueId == flip.IndustryUniqueId);
                    if (s.Value.BuildedIndustry is BaseSellableIndustry) {
                        // Stock selling with brewery beer does not bind a merchant.
                        // Use a deterministic compatible reachable witness in that
                        // case; this is not a claim that stock selected that shop.
                        string merchantWitness = saleMerchant;
                        if (merchantWitness == null) {
                            var region = Game.Board.GetRegionContainingGivenSlot(s.Value);
                            merchantWitness = Merchants.First(m => m.Value.MerchantTile != null &&
                                m.Value.MerchantTile.BuildingTypes.Contains(s.Value.BuildedIndustry.BuildingType) &&
                                (region == Game.Board.GetRegionOfGivenMerchantTile(m.Value) ||
                                 Game.Board.DoesRouteExists(region, Game.Board.GetRegionOfGivenMerchantTile(m.Value)))).Key;
                        }
                        choices.Insert(saleStart, new Choice("sell", s.Key, card: takeCard(), merchant: merchantWitness));
                        saleStart = choices.Count; saleMerchant = null;
                    }
                }
                if (e.GetType().Name == "MerchantBonusReceived") {
                    var property = e.GetType().GetProperty("BonusType");
                    bonusDevelop = property.GetValue(e).ToString() == "Develop"; bonusUsed = false;
                }
                if (e.GetType().Name == "MerchantBonusDevelopFinished") {
                    if (bonusDevelop && !bonusUsed) choices.Add(new Choice("bonus_skip"));
                    bonusDevelop = false;
                }
            }
            if (!usedCard || !choices.Any(c => c.kind == kind)) throw new InvalidOperationException("unrecorded_stock_action");
            choices.Add(new Choice("finish"));
            foreach (var choice in choices) publicHistory.Add(new { seat, era, choice });
        }
    }
}
