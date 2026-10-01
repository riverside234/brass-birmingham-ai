using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;

namespace BrassPatch.Statistics
{
    internal sealed class StatisticsProjection
    {
        readonly CompletedMatchStatistics report;
        readonly List<IEvent> events;
        readonly Dictionary<string, StatisticsAsset> active = new Dictionary<string, StatisticsAsset>();
        readonly Dictionary<string, string> saleMerchants = new Dictionary<string, string>();
        readonly Dictionary<string, StatisticsEntry> wildEpisodes = new Dictionary<string, StatisticsEntry>();
        BoardGameRulesGame game;
        StatisticsAction frame;
        StatisticsRound round;
        int era = 1, roundNumber, turn, index, revision;
        PlayerColor actor;
        string pendingLink, pendingBuild;
        bool merchantDevelopment;
        sealed class Balance { public int Cash, Income, Marker, VP; }
        public StatisticsProjection(CompletedMatchStatistics report, List<IEvent> events) { this.report = report; this.events = events; }
        public void Run()
        {
            game = new BoardGameRulesGame(((GameStarted)events[0]).GameStartConfiguration);
            for (index = 0; index < events.Count; index++) {
                var change = events[index]; revision = change.Metadata.Revision;
                var before = game.PlayersTrack == null ? new Dictionary<PlayerColor, Balance>() : game.PlayersTrack.Players.ToDictionary(p => p.Color,
                    p => new Balance { Cash = p.Money, Income = p.IncomeLevel, Marker = p.IncomeMarkerPosition, VP = p.VP });
                string name = change.GetType().Name;
                var startedRound = change as RoundStarted;
                if (startedRound != null) { era = startedRound.GameEra.ToString() == "RailEra" ? 2 : 1; roundNumber = startedRound.RoundNumber; }
                var startedTurn = change as TurnStarted;
                if (startedTurn != null) { actor = startedTurn.PlayerColor; turn++; frame = null; pendingBuild = pendingLink = null; }
                var player = Color(change);
                string resource = name.StartsWith("Coal") ? "Coal" : name.StartsWith("Iron") ? "Iron" : name.StartsWith("Beer") ? "Beer" : null;
                int quote = game.Board != null && (resource == "Coal" || resource == "Iron") ? Quote(resource) : 0;
                if (name == "ScoreLinksPhaseStarted") { ScoreLinks(); Cities("Scoring", era); }
                if (name == "ScoreFlippedIndustryTilesPhaseStarted") ScoreIndustries(false);
                if (name == "ScoreAdditionalIndustryPhaseStarted") ScoreIndustries(true);
                var build = change as IndustryBuilt; if (build != null) Build(build);
                var developed = change as IndustryDeveloped; if (developed != null) Develop(developed);
                var placed = change as LinkPlaced; if (placed != null) Place(placed);
                var flipped = change as IndustryFlipped; if (flipped != null) Flip(flipped);
                if (name == "IndustrySold") Liquidate(change);
                if (name == "ObsoleteIndustriesRemoved") RemoveObsolete(player);
                if (name.EndsWith("Supplied")) Supply(change, resource);
                if (name == "CoalSold" || name == "IronSold") MarketSale(change, resource, quote);
                if (name == "CoalBought" || name == "IronBought") MarketBuy(change, resource, quote);
                if (name == "MerchantBonusReceived") {
                    merchantDevelopment = Text(change, "BonusType") == "Develop";
                    var bonus = Entry(player, "Merchant bonus", Text(change, "BonusType"), Text(change, "RegionName") + " · " + Text(change, "BonusType"));
                    bonus.Destination = Text(change, "RegionName"); bonus.Quantity = 1;
                }
                if (name == "MerchantBonusDevelopFinished") merchantDevelopment = false;
                var discarded = change as CardDiscarded; if (discarded != null) Card(discarded);
                var scouted = change as WildCardsTaken;
                if (scouted != null) {
                    EnsureFrame().Kind = "Scout";
                    foreach (var card in scouted.CardsToTrade) {
                        var payment = Entry(scouted.Player, "Card", "Scout trade", CardType(card)); payment.Quantity = 1;
                        EnsureFrame().notes.Add("Scout trade: " + CardType(card) + " · " + card);
                    }
                }
                if (name == "ActionPassed") EnsureFrame().Kind = "Pass";
                if (name == "MoneyLoaned") { EnsureFrame().Kind = "Loan"; EnsureFrame().Borrowed += Number(change, "Amount"); }
                if (name == "TurnEnded" && frame != null) throw new InvalidDataException("Unconfirmed action before turn end.");
                game.ApplyEvent(change);
                if (name == "GameStarted") {
                    foreach (var p in report.players) { var initial = game.PlayersTrack.GetPlayerByColor(p.Color); p.OpeningCash = initial.Money; p.OpeningVP = initial.VP; }
                }
                if (startedRound != null) {
                    round = new StatisticsRound { Era = era, Round = roundNumber };
                    round.order.AddRange(game.PlayersTrack.Players.Select(p => p.Color)); report.rounds.Add(round);
                }
                if (build != null) {
                    var asset = active[game.Board.GetBuildingSlotByUniqueId(build.SlotUniqueId).BuildedIndustry.UniqueId];
                    asset.Produced = asset.Stock = game.Board.GetBuildingSlotByUniqueId(build.SlotUniqueId).BuildedIndustry.GetResource();
                }
                if (scouted != null) {
                    foreach (var card in game.PlayersTrack.GetPlayerByColor(scouted.Player).Hand.Where(c => c is WildIndustryCard || c is WildLocationCard)) {
                        var episode = Entry(scouted.Player, "Wildcard acquired", CardType(card), card.UniqueId); episode.Quantity = 1;
                        wildEpisodes[card.UniqueId] = episode;
                    }
                }
                if (name == "CoalBought" || name == "IronBought" || name == "CoalSold" || name == "IronSold")
                    report.entries.Last(e => e.Revision == revision && e.Resource == resource).AfterQuote = Quote(resource);
                foreach (var p in game.PlayersTrack.Players) {
                    Balance old;
                    if (!before.TryGetValue(p.Color, out old)) { Sample(p, name); continue; }
                    int cash = p.Money - old.Cash, income = p.IncomeLevel - old.Income, marker = p.IncomeMarkerPosition - old.Marker, vp = p.VP - old.VP;
                    if (cash != 0 || income != 0 || marker != 0 || vp != 0) {
                        var entry = Entry(p.Color, "Balance", Category(name, vp), EventLabel(name, change));
                        entry.Cash = cash; entry.Income = income; entry.Marker = marker; entry.VP = vp;
                        entry.Detail = "Cash " + old.Cash + " → " + p.Money + "; income " + old.Income + " → " + p.IncomeLevel + "; VP " + old.VP + " → " + p.VP;
                        if (frame != null && p.Color == actor && name == "MoneySpent") {
                            frame.Spending -= cash;
                            if (round != null) { int existing; round.spending.TryGetValue(actor, out existing); round.spending[actor] = existing - cash; }
                        }
                        var bonus = report.entries.LastOrDefault(e => e.Revision == revision && e.Kind == "Merchant bonus" && e.Player == p.Color);
                        if (bonus != null) { bonus.Cash = cash; bonus.Income = income; bonus.Marker = marker; bonus.VP = vp; }
                        string sourceId = Text(change, "SourceUniqueId");
                        StatisticsAsset incomeAsset;
                        if (marker > 0 && active.TryGetValue(sourceId, out incomeAsset)) { incomeAsset.IncomeSpaces += marker; entry.AssetKey = incomeAsset.Key; }
                        Sample(p, entry.Label);
                    } else if (name == "TurnEnded" || name == "RoundEnded" || name == "GameEnded" || startedTurn != null) Sample(p, name);
                }
                if (name == "IndustryBuilt" || name.EndsWith("Supplied") || name == "CoalSold" || name == "IronSold") RefreshStock();
                if (name == "CanalEraEnded") {
                    Cities("After cleanup", 1);
                    foreach (var asset in active.Values) asset.Carried = true;
                }
                if (name == "ActionCountDecreased") Confirm();
            }
            if (!game.GameProgressInformation.IsGameOver) throw new InvalidDataException("Event history does not reconstruct a completed match.");
            foreach (var expected in report.players) {
                var actual = game.PlayersTrack.GetPlayerByColor(expected.Color);
                if (actual.VP != expected.FinalVP || actual.Money != expected.FinalCash || actual.IncomeLevel != expected.FinalIncome)
                    throw new InvalidDataException("Reconstructed result differs for " + expected.Name + ".");
                if (expected.OpeningCash + report.entries.Where(e => e.Player == expected.Color && e.Kind == "Balance").Sum(e => e.Cash) != expected.FinalCash ||
                    expected.OpeningVP + report.entries.Where(e => e.Player == expected.Color && e.Kind == "Balance").Sum(e => e.VP) != expected.FinalVP)
                    throw new InvalidDataException("Financial/score ledger failed to reconcile.");
                int expectedCount = (report.Introductory ? 2 : 4) * (12 - report.players.Count) - 1;
                int count = report.actions.Count(a => a.Player == expected.Color);
                if (count != expectedCount) report.warnings.Add(expected.Name + ": " + count + " recorded actions; standard complete-match expectation is " + expectedCount + ".");
            }
            if (report.links.Where(l => l.Scored).Sum(l => l.VP) != report.entries.Where(e => e.Kind == "Balance" && e.Category == "Links").Sum(e => e.VP))
                throw new InvalidDataException("Per-link scores failed to reconcile.");
            if (report.assets.Sum(a => a.CanalVP + a.RailVP) != report.entries.Where(e => e.Kind == "Balance" && e.Category == "Industries").Sum(e => e.VP))
                throw new InvalidDataException("Per-industry scores failed to reconcile.");
            foreach(var asset in report.assets.Where(a=>a.Produced>0)) {
                int marketSold=report.entries.Where(e=>e.Kind=="Market sale"&&e.AssetKey==asset.Key).Sum(e=>e.Quantity);
                if(asset.Produced!=asset.Supplied+marketSold+asset.Stock)
                    throw new InvalidDataException("Resource stock failed to reconcile for "+asset.Key+".");
            }
            int unknown = report.entries.Count(e => e.Kind == "Goods sale" && e.Destination == "Unattributed");
            if (unknown > 0) report.warnings.Add(unknown + " sales have no unique recorded merchant destination; these remain unattributed.");
        }
        StatisticsAction EnsureFrame()
        {
            if (frame == null) frame = new StatisticsAction { Key = "action-" + revision, Revision = revision, Era = era, Round = roundNumber, Turn = turn, Player = actor, Kind = "Unclassified" };
            return frame;
        }
        StatisticsEntry Entry(PlayerColor player, string kind, string category, string label)
        {
            var entry = new StatisticsEntry { Revision = revision, Era = era, Round = roundNumber, Turn = turn, Player = player,
                Kind = kind, Category = category, Label = label, ActionKey = frame == null ? null : frame.Key };
            report.entries.Add(entry); return entry;
        }
        void Sample(Player player, string annotation)
        {
            report.samples.Add(new StatisticsSample { Revision = revision, Era = era, Round = roundNumber, Turn = turn, Player = player.Color,
                Cash = player.Money, Income = player.IncomeLevel, VP = player.VP, Annotation = annotation });
        }
        void Build(IndustryBuilt change)
        {
            EnsureFrame().Kind = "Build";
            var slot = game.Board.GetBuildingSlotByUniqueId(change.SlotUniqueId);
            var region = game.Board.GetRegionContainingGivenSlot(slot);
            bool overbuild = slot.BuildedIndustry != null, opponent = overbuild && slot.BuildedIndustry.Color != change.Player;
            if (overbuild) Removed(slot.BuildedIndustry.UniqueId, "Overbuilt");
            var tile = game.PlayersTrack.GetPlayerByColor(change.Player).Mat.GetFirstBuildingOfType(change.Type);
            var asset = new StatisticsAsset { Key = tile.UniqueId + "@" + revision, IndustryId = tile.UniqueId, Slot = change.SlotUniqueId,
                City = region.Name.ToString(), Type = change.Type.ToString(), Level = tile.Level, Player = change.Player,
                Revision = revision, Era = era, Round = roundNumber, Turn = turn, PrintedVP = tile.VPForBuilding,
                LinkIcons = tile.VPPerLink, BaseCost = tile.MoneyCost, CoalUsed = tile.CoalCost, IronUsed = tile.IronCost,
                OpponentOverbuild = opponent, LastSlot = !overbuild && region.BuildingSlots.Count(s => s.BuildedIndustry == null && s.BuildingTypes.Contains(change.Type)) == 1 };
            report.assets.Add(asset); active.Add(tile.UniqueId, asset); pendingBuild = tile.UniqueId;
            EnsureFrame().notes.Add("Build " + asset.Type + " " + asset.Level + " in " + asset.City + (asset.LastSlot ? " · last vacant matching slot" : ""));
            var entry = Entry(change.Player, "Build", asset.Type, asset.City + " · " + asset.Type + " " + asset.Level); entry.AssetKey = asset.Key; entry.Quantity = 1;
            var payment = report.entries.LastOrDefault(e => e.ActionKey == frame.Key && e.Kind == "Card" && e.Category == "Payment");
            if (payment != null && payment.Label.StartsWith("Wild")) {
                var remaining = game.PlayersTrack.GetPlayerByColor(change.Player).Hand;
                bool inNetwork = region.BuildingSlots.Any(s => s.BuildedIndustry != null && s.BuildedIndustry.Color == change.Player) || region.Links.Any(l => l.Owner != null && l.Owner.Color == change.Player);
                bool ordinary = remaining.Any(c => c is LocationCard && ((LocationCard)c).Region == region.Name || c is IndustryCard && inNetwork && ((IndustryCard)c).BuildingTypes.Contains(change.Type));
                if (!ordinary) payment.Detail += "\nNo ordinary card remaining in hand authorized this same placement.";
            }
        }
        void Develop(IndustryDeveloped change)
        {
            var tile = game.PlayersTrack.GetPlayerByColor(change.Player).Mat.GetFirstBuildingOfType(change.Type);
            if (!merchantDevelopment) { EnsureFrame().Kind = "Develop"; EnsureFrame().Developed++; }
            var entry = Entry(change.Player, "Development", merchantDevelopment ? "Merchant" : "Develop action", change.Type + " " + tile.Level);
            entry.Quantity = tile.Level; entry.Resource = change.Type.ToString(); entry.Detail = merchantDevelopment ? "Merchant development · no additional action/iron" : "Removed from player mat";
            if (frame != null) frame.notes.Add(entry.Detail + ": " + entry.Label);
        }
        void Place(LinkPlaced change)
        {
            EnsureFrame().Kind = "Network"; EnsureFrame().Links++;
            var link = game.Board.GetLinkById(change.LinkId);
            var record = new StatisticsLink { Key = change.LinkId + "@" + revision, LinkId = change.LinkId, Player = change.Player,
                Revision = revision, Era = era, Round = roundNumber, Turn = turn, Route = String.Join(" — ", Ends(link).Select(r => r.Name.ToString())) };
            report.links.Add(record); pendingLink = change.LinkId;
            EnsureFrame().notes.Add("Place " + record.Route);
        }
        void Flip(IndustryFlipped change)
        {
            StatisticsAsset asset;
            if (!active.TryGetValue(change.IndustryUniqueId, out asset)) throw new InvalidDataException("Flip references an untracked industry.");
            asset.FlipTurn = turn; asset.FlipRevision = revision; asset.FlipEra = era;
            var tile = game.Board.GetIndustryByUniqueId(change.IndustryUniqueId);
            if (!(tile is BaseSellableIndustry)) return;
            EnsureFrame().Kind = "Sell"; EnsureFrame().Goods++;
            string merchant; bool derived = false;
            if (!saleMerchants.TryGetValue(tile.UniqueId, out merchant)) {
                var region = game.Board.GetRegionContainingGivenIndustry(tile);
                var eligible = game.Board.Regions.OfType<BorderRegion>().Where(r => r.MerchantSlots.Any(s => s.MerchantTile != null && s.MerchantTile.BuildingTypes.Contains(tile.BuildingType)) && Connected(region, r, null)).ToList();
                merchant = eligible.Count == 1 ? eligible[0].Name.ToString() : "Unattributed"; derived = eligible.Count == 1;
            }
            var entry = Entry(asset.Player, "Goods sale", asset.Type, asset.City + " · " + asset.Type + " " + asset.Level);
            entry.AssetKey = asset.Key; entry.Quantity = 1; entry.Destination = merchant; entry.DestinationDerived = derived;
            entry.Detail = merchant == "Unattributed" ? "No unique merchant destination was recorded." : derived ? "Only eligible merchant location in the recorded state." : "Merchant beer identifies this destination.";
            EnsureFrame().notes.Add("Sell " + entry.Label + " → " + merchant);
            if (merchant != "Unattributed") {
                var destination = game.Board.Regions.First(r => r.Name.ToString() == merchant);
                Connection(game.Board.GetRegionContainingGivenIndustry(tile), destination, "Merchant sale");
            }
        }
        void Liquidate(IEvent change)
        {
            var slot = game.Board.GetBuildingSlotByUniqueId(Text(change, "SlotUniqueId"));
            var asset = active[slot.BuildedIndustry.UniqueId];
            var entry = Entry(Color(change), "Liquidation", asset.Type, asset.City + " · " + asset.Type + " " + asset.Level);
            entry.AssetKey = asset.Key; entry.Quantity = 1; Removed(slot.BuildedIndustry.UniqueId, "Liquidated");
        }
        void Removed(string id, string reason)
        {
            StatisticsAsset asset;
            if (!active.TryGetValue(id, out asset)) throw new InvalidDataException("Removal references an untracked industry.");
            asset.Stock = game.Board.GetIndustryByUniqueId(id).GetResource(); asset.RemovedRevision = revision; asset.RemovedEra = era; asset.Removal = reason; active.Remove(id);
        }
        void RemoveObsolete(PlayerColor player)
        {
            foreach (var tile in game.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry).Where(t => t != null && t.Color == player && t.Level == 1).ToList()) Removed(tile.UniqueId, "Canal cleanup");
        }
        void RefreshStock()
        {
            foreach (var item in active) item.Value.Stock = game.Board.GetIndustryByUniqueId(item.Key).GetResource();
        }
        void Supply(IEvent change, string resource)
        {
            EnsureFrame();
            string id = Text(change, resource == "Coal" ? "SourceUniqueId" : resource == "Iron" ? "SourceId" : "BeerSourceId");
            var tile = game.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry).FirstOrDefault(t => t != null && t.UniqueId == id);
            var merchant = game.Board.Regions.OfType<BorderRegion>().FirstOrDefault(r => r.MerchantSlots.Any(s => s.UniqueId == id));
            var entry = Entry(actor, "Resource use", resource, resource + " from " + (tile != null ? report.Name(tile.Color) : merchant != null ? merchant.Name.ToString() : "Market"));
            entry.Quantity = Number(change, "Amount"); entry.Resource = resource; entry.Source = tile != null ? "Industry" : merchant != null ? "Merchant" : "Market"; entry.Supplier = tile == null ? (PlayerColor?)null : tile.Color;
            if (tile != null) { var asset = active[tile.UniqueId]; asset.Supplied += entry.Quantity; entry.AssetKey = asset.Key; }
            Region target = Target(resource); if (target != null) entry.Destination = target.Name.ToString();
            if (merchant != null) {
                var sale = FutureSale(); if (sale != null) saleMerchants[sale.UniqueId] = merchant.Name.ToString();
                entry.Detail = "Merchant beer · " + merchant.Name;
            }
            if (target != null && (resource == "Coal" || resource == "Beer" && tile != null && tile.Color != actor)) {
                var start = tile != null ? game.Board.GetRegionContainingGivenIndustry(tile) : merchant;
                if (start != null) Connection(start, target, resource + " delivery");
            }
            EnsureFrame().notes.Add(entry.Label + " × " + entry.Quantity + (entry.Destination == null ? "" : " → " + entry.Destination));
        }
        BaseIndustry FutureSale()
        {
            for (int i = index + 1; i < events.Count; i++) {
                if (events[i] is ActionCountDecreased || events[i] is TurnStarted || events[i] is TurnEnded) break;
                var flip = events[i] as IndustryFlipped;
                if (flip != null) { var tile = game.Board.GetIndustryByUniqueId(flip.IndustryUniqueId); if (tile is BaseSellableIndustry && tile.Color == actor) return tile; }
            }
            return null;
        }
        Region Target(string resource)
        {
            var sale = resource == "Beer" ? FutureSale() : null;
            if (sale != null) return game.Board.GetRegionContainingGivenIndustry(sale);
            if (pendingLink != null && frame.Kind == "Network") return game.Board.GetLinkById(pendingLink).Region1;
            if (pendingBuild != null && frame.Kind == "Build") return game.Board.GetRegionContainingGivenIndustry(game.Board.GetIndustryByUniqueId(pendingBuild));
            for (int i = index + 1; i < events.Count; i++) {
                if (events[i] is ActionCountDecreased || events[i] is TurnStarted || events[i] is TurnEnded) break;
                var build = events[i] as IndustryBuilt;
                if (build != null) return game.Board.GetRegionContainingGivenSlot(game.Board.GetBuildingSlotByUniqueId(build.SlotUniqueId));
            }
            return null;
        }
        void MarketSale(IEvent change, string resource, int quote)
        {
            var tile = game.Board.GetBuildingSlotByUniqueId(Text(change, "SlotUniqueId")).BuildedIndustry;
            var asset = active[tile.UniqueId]; asset.MarketReceipts += Number(change, "MoneyReceived");
            var entry = Entry(Color(change), "Market sale", resource, asset.City + " · " + resource);
            entry.Quantity = Number(change, "Amount"); entry.Resource = resource; entry.AssetKey = asset.Key; entry.BeforeQuote = quote;
            entry.Detail = "Receipt " + StatisticsFormat.Money(Number(change, "MoneyReceived")) + "; producer " + report.Name(tile.Color);
        }
        void MarketBuy(IEvent change, string resource, int quote)
        {
            EnsureFrame().MarketCost += Number(change, "Price");
            var entry = Entry(Color(change), "Market purchase", resource, resource + " purchase");
            entry.Quantity = Number(change, "Amount"); entry.Resource = resource; entry.BeforeQuote = quote; entry.Source = "Market";
            entry.Detail = "Price " + StatisticsFormat.Money(Number(change, "Price"));
        }
        int Quote(string resource) { return resource == "Coal" ? game.Board.CoalMarket.GetBuyingPrice(1) : game.Board.IronMarket.GetBuyingPrice(1); }
        void Card(CardDiscarded change)
        {
            EnsureFrame().cards.Add(CardType(change.CardToDiscard));
            var entry = Entry(change.Player, "Card", "Payment", CardType(change.CardToDiscard)); entry.Quantity = 1; entry.Detail = change.CardToDiscard.ToString();
            if (change.CardToDiscard is WildIndustryCard || change.CardToDiscard is WildLocationCard) {
                StatisticsEntry episode;
                if (wildEpisodes.TryGetValue(change.CardToDiscard.UniqueId, out episode)) { entry.AssetKey = "wild@" + episode.Revision; wildEpisodes.Remove(change.CardToDiscard.UniqueId); }
            }
        }
        void Confirm()
        {
            if (frame == null) throw new InvalidDataException("Action confirmation has no operation frame.");
            frame.EndRevision = revision;
            frame.IncomeDrop = -report.entries.Where(e => e.ActionKey == frame.Key && e.Kind == "Balance" && e.Label == "Loan income reduction").Sum(e => e.Income);
            if (pendingBuild != null && active.ContainsKey(pendingBuild)) active[pendingBuild].MarketCost += frame.MarketCost;
            foreach (var entry in report.entries.Where(e => e.ActionKey == frame.Key)) {
                if (entry.Kind == "Resource use") entry.Category = frame.Kind;
                if (entry.Kind == "Card" && entry.Category == "Payment") entry.Category = frame.Kind == "Build" ? "Build" : "Non-Build payment";
            }
            report.actions.Add(frame); frame = null; pendingBuild = pendingLink = null;
        }
        void ScoreIndustries(bool intro)
        {
            foreach (var tile in game.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry).Where(t => t != null)) {
                var asset = active[tile.UniqueId];
                if (intro) { if (tile.Level >= 2) asset.IntroVP = tile.VPForBuilding; }
                else if (era == 1) { asset.CanalPrinted = tile.VPForBuilding; asset.CanalVP = tile.IsFlipped ? tile.VPForBuilding : 0; }
                else { asset.RailPrinted = tile.VPForBuilding; asset.RailVP = tile.IsFlipped ? tile.VPForBuilding : 0; }
            }
        }
        void ScoreLinks()
        {
            foreach (var link in game.Board.Regions.SelectMany(r => r.Links).Distinct().Where(l => l.Owner != null)) {
                var record = report.links.LastOrDefault(l => l.LinkId == link.LinkId && l.Era == era);
                if (record == null) throw new InvalidDataException("Score references an untracked link.");
                var notes = new List<string>();
                foreach (var region in Ends(link)) {
                    if (region is BorderRegion) { record.MerchantVP += 2; notes.Add(region.Name + ": merchant icons 2"); }
                    else foreach (var tile in region.BuildingSlots.Select(s => s.BuildedIndustry).Where(t => t != null && t.IsFlipped)) {
                        if (tile.Color == record.Player) record.OwnVP += tile.VPPerLink; else record.OthersVP += tile.VPPerLink;
                        notes.Add(region.Name + ": " + report.Name(tile.Color) + " " + tile.BuildingType + " " + tile.Level + " · " + tile.VPPerLink);
                    }
                }
                record.VP = record.OwnVP + record.OthersVP + record.MerchantVP; record.Scored = true; record.Contributions = String.Join("\n", notes);
            }
        }
        void Cities(string boundary, int snapshotEra)
        {
            foreach (var region in game.Board.Regions.Where(r => r.BuildingSlots.Count > 0))
            foreach (var player in report.players) report.cities.Add(new StatisticsCity { Boundary = boundary, Era = snapshotEra, City = region.Name.ToString(), Player = player.Color,
                Slots = region.BuildingSlots.Count, Occupied = region.BuildingSlots.Count(s => s.BuildedIndustry != null), Owned = region.BuildingSlots.Count(s => s.BuildedIndustry != null && s.BuildedIndustry.Color == player.Color) });
        }
        void Connection(Region from, Region to, string purpose)
        {
            if (!Connected(from, to, null)) return;
            report.connections.Add(new StatisticsConnection { Revision = revision, Era = era, Round = roundNumber, Turn = turn, Player = actor,
                ActionKey = frame == null ? null : frame.Key, From = from.Name.ToString(), To = to.Name.ToString(), Purpose = purpose, OpponentRequired = !Connected(from, to, actor) });
        }
        bool Connected(Region from, Region to, PlayerColor? only)
        {
            var seen = new HashSet<Region> { from }; var queue = new Queue<Region>(); queue.Enqueue(from);
            while (queue.Count > 0) {
                var region = queue.Dequeue(); if (region == to) return true;
                foreach (var link in region.Links.Where(l => l.Owner != null && (!only.HasValue || l.Owner.Color == only.Value)))
                foreach (var next in Ends(link)) if (seen.Add(next)) queue.Enqueue(next);
            }
            return false;
        }
        static IEnumerable<Region> Ends(Link link) { return new[] { link.Region1, link.Region2, link.Region3 }.Where(r => r != null); }
        PlayerColor Color(object change)
        {
            foreach (string field in new[] { "Player", "PlayerColor" }) {
                object value = Value(change, field); if (value is PlayerColor) return (PlayerColor)value;
            }
            return actor;
        }
        internal static object Value(object value, string property) { var info = value.GetType().GetProperty(property); return info == null ? null : info.GetValue(value, null); }
        internal static string Text(object value, string property) { var item = Value(value, property); return item == null ? "" : item.ToString(); }
        internal static int Number(object value, string property) { var item = Value(value, property); return item == null ? 0 : Convert.ToInt32(item); }
        static string CardType(BaseCard card) { return card is WildIndustryCard ? "Wild Industry" : card is WildLocationCard ? "Wild Location" : card is LocationCard ? "Location" : "Industry"; }
        static string Category(string name, int vp)
        {
            if (name == "LinksVictoryPointsChanged") return "Links";
            if (name == "FlippedIndustriesVictoryPointsChanged") return "Industries";
            if (name.StartsWith("IntroductoryGame")) return "Introductory";
            if (name == "MerchantBonusReceived") return "Merchant";
            if (vp < 0) return "Penalty";
            if (name == "IncomeMoneyPaid") return "Income payment";
            if (name == "MoneyLoaned") return "Loan";
            if (name == "CoalSold" || name == "IronSold") return "Market receipt";
            if (name == "IndustrySold") return "Liquidation receipt";
            if (name == "MoneySpent") return "Action spending";
            return "State change";
        }
        static string EventLabel(string name, object change)
        {
            if (name == "IncomeMarkerPositionChanged") return "Industry/bonus income increase";
            if (name == "IncomeLevelChanged") return "Loan income reduction";
            if (name == "MerchantBonusReceived") return Text(change, "RegionName") + " merchant bonus";
            return name.Replace("VictoryPointsChanged", " VP").Replace("IntroductoryGameAdditional", "Introductory ");
        }
    }
}
