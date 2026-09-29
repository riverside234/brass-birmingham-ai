using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Consumables;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Markets;
using BoardGameRules.Entities.NumericalResources;
using BoardGameRules.Entities.Players;
using BrassApplication.AI.AIHelpers;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions.NetworkAction;
using BrassApplication.UseCases.Game.Actions.SellAction;

namespace BrassPatch
{
    /// <summary>Corrections for B02, B03, B04, B08 and B10. Events retain their shipped schema.</summary>
    public static class ResourcesFixes
    {
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        private static FieldInfo Field(object obj, string name)
        {
            for (Type type = obj.GetType(); type != null; type = type.BaseType)
            {
                FieldInfo field = type.GetField(name, Flags);
                if (field != null) return field;
            }
            throw new MissingFieldException(obj.GetType().FullName, name);
        }
        private static T Get<T>(object obj, string name) { return (T)Field(obj, name).GetValue(obj); }
        private static void Set(object obj, string name, object value) { Field(obj, name).SetValue(obj, value); }
        private static T Property<T>(object obj, string name)
        {
            for (Type type = obj.GetType(); type != null; type = type.BaseType)
            {
                PropertyInfo property = type.GetProperty(name, Flags);
                if (property != null) return (T)property.GetValue(obj, null);
            }
            throw new MissingMemberException(obj.GetType().FullName, name);
        }

        // Whole-action affordability counts every connected cube, even if a nearer mine
        // must first be depleted. Actual delivery below still enforces nearest-first.
        public static int CostOfCoalForGivenIndustry(Region region, Board board, int coalNeeded)
        {
            int available = board.Regions.Where(r => board.DoesRouteExists(r, region))
                .SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry).OfType<CoalMine>()
                .Where(m => m.HasCoal).Sum(m => m.AvailableCoal);
            int shortfall = Math.Max(0, coalNeeded - available);
            if (shortfall == 0) return 0;
            return ConsumingCoal.CanCoalBeSuppliedToRegionFromSource(board.CoalMarket, region, board, true)
                ? board.CoalMarket.CoalPrice(shortfall) : -1;
        }

        public static bool IsGivenSlotTheBestOneInGivenRegionForGivenTypeStatic(BuildingSlot slot, BuildingType type, Board board)
        {
            return IsSingleSlotOrDoubleSlotButLastPossibleStatic(slot, type, board.GetRegionContainingGivenSlot(slot));
        }
        public static bool IsSingleSlotOrDoubleSlotButLastPossibleStatic(BuildingSlot slot, BuildingType type, Region region)
        {
            if (slot.BuildedIndustry != null || slot.BuildingTypes.Count == 1) return true;
            return !region.BuildingSlots.Any(s => s != slot && s.BuildedIndustry == null &&
                s.BuildingTypes.Count == 1 && s.BuildingTypes.Contains(type));
        }

        public static List<CoalMine> ClosestCoalMines(Board board, IEnumerable<Region> regions)
        {
            List<Region> targets = regions.Where(r => r != null).Distinct().ToList();
            int minimum = int.MaxValue;
            List<CoalMine> result = new List<CoalMine>();
            foreach (Region origin in board.Regions)
            {
                List<CoalMine> mines = origin.BuildingSlots.Select(s => s.BuildedIndustry).OfType<CoalMine>()
                    .Where(m => m.HasCoal).ToList();
                if (mines.Count == 0) continue;
                int distance = int.MaxValue;
                foreach (Region target in targets)
                    if (board.DoesRouteExists(origin, target)) distance = Math.Min(distance, board.GetRouteLength(origin, target));
                if (distance == int.MaxValue || distance > minimum) continue;
                if (distance < minimum) { result.Clear(); minimum = distance; }
                result.AddRange(mines);
            }
            return result;
        }

        public static void SupplyCoal(ConsumingCoal self, ICoalSource source, int amount, int neededCoal, Player player, List<Region> regions)
        {
            if (neededCoal <= 0 || amount <= 0 || amount > neededCoal || player == null || source == null)
                throw new InvalidOperationException("Invalid coal delivery.");
            if (!(source is CoalMarket) && source.AvailableCoal < amount)
                throw new InvalidOperationException("Given source does not have enough coal.");
            Board board = Property<Board>(self, "Board");
            List<CoalMine> nearest = ClosestCoalMines(board, regions);
            bool permitted = source is CoalMarket
                ? nearest.Count == 0 && regions.Any(r => r != null && ConsumingCoal.CanCoalBeSuppliedToRegionFromSource(source, r, board, true))
                : nearest.Any(m => ReferenceEquals(m, source));
            if (!permitted) throw new InvalidOperationException("Coal must come from a closest connected mine, or an accessible market after board supply is exhausted.");
            int cost = source.CoalPrice(amount);
            if (player.Money < cost) throw new InvalidOperationException("Not enough money for coal.");
            IAggregateRoot aggregate = Property<IAggregateRoot>(self, "AggregateRoot");
            if (cost > 0) aggregate.ApplyEvent(new CoalBought(amount, board.CoalMarket.GetBuyingPrice(amount), player.Color));
            aggregate.ApplyEvent(new CoalSupplied(source.UniqueId, amount));
            BaseIndustry industry = source as BaseIndustry;
            if (industry != null && !source.HasCoal)
            {
                aggregate.ApplyEvent(new IndustryFlipped(industry.UniqueId, board.GetRegionContainingGivenIndustry(industry).Name));
                Property<PlayersTrack>(self, "PlayersTrack").GetPlayerByColor(industry.Color)
                    .ChangeIncomeMarkerPositionBy(industry.IncomeIncrease, industry.UniqueId);
            }
        }

        public static void PlaceLink(NetworkAction action, BaseCard card, Link link, Board board, Player player, bool is2nd)
        {
            if (!is2nd) action.DiscardCard(card, player);
            action.PlaceLink(link);
            List<Region> targets = new List<Region> { link.Region1, link.Region2, link.Region3 };
            while (action.IsCoalNeeded())
            {
                ICoalSource mine = ClosestCoalMines(board, targets).FirstOrDefault();
                action.SupplyCoal(mine ?? board.CoalMarket, 1);
            }
            while (action.IsBeerNeeded())
            {
                Brewery brewery = board.GetUnflippedBreweries().First(b => targets.Any(r => r != null &&
                    action.GeneralAction.ConsumingBeer.CanBeerBeSuppliedToSlotFromSource(b, r, player)));
                action.SupplyBeer(brewery, 1);
            }
            if (!is2nd)
            {
                Link second = BoardHelper.Find2ndLink();
                if (second != null) PlaceLink(action, null, second, board, player, true);
            }
        }

        public static void StartCoalDelivery(NetworkActionUseCase self)
        {
            Set(self, "_networkActionUseCaseState", NetworkActionUseCaseState.CoalDelivery);
            Link link = Get<Link>(self, "_link");
            NetworkAction action = Get<NetworkAction>(self, "_networkAction");
            Board board = Get<IGameHistory>(self, "_gameHistory").CurrentGame.Game.Board;
            Get<INetworkActionOutput>(self, "_presenter").StartCoalDeliveryUseCase(action.NeededCoal,
                ClosestCoalMines(board, new[] { link.Region1, link.Region2, link.Region3 }), self);
        }

        private sealed class SaleContext { public MerchantSlot UsedMerchant; }
        private static readonly ConditionalWeakTable<SellAction, SaleContext> Sales = new ConditionalWeakTable<SellAction, SaleContext>();
        private static SaleContext Context(SellAction action) { return Sales.GetOrCreateValue(action); }
        private static List<MerchantSlot> Merchants(Board board, BaseIndustry industry)
        {
            Region region = board.GetRegionContainingGivenIndustry(industry);
            if (region == null) return new List<MerchantSlot>();
            return board.GetBorderRegions(r => board.DoesRouteExists(r, region)).SelectMany(r => r.MerchantSlots)
                .Where(m => m.MerchantTile != null && m.MerchantTile.BuildingTypes.Contains(industry.BuildingType)).ToList();
        }
        private static List<Brewery> Breweries(SellAction action, Board board, BaseIndustry industry, Player player)
        {
            Region region = board.GetRegionContainingGivenIndustry(industry);
            return board.GetUnflippedBreweries().Where(b => b.HasBeer &&
                action.GeneralAction.ConsumingBeer.CanBeerBeSuppliedToSlotFromSource(b, region, player)).ToList();
        }
        public static bool CanIndustryBeSold(SellAction self, BaseIndustry industry, Player player)
        {
            BaseSellableIndustry sale = industry as BaseSellableIndustry;
            if (sale == null || player == null || industry.Color != player.Color || industry.IsFlipped) return false;
            Board board = Property<Board>(self, "Board");
            List<MerchantSlot> merchants = Merchants(board, industry);
            return merchants.Count > 0 && Breweries(self, board, industry, player).Sum(b => b.AvailableBeer)
                + (merchants.Any(m => m.HasBeer) ? 1 : 0) >= sale.CostOfBeerBarrelsToSellIndustry;
        }
        public static List<CantUseSellActionReason> GetSellActionNotPossibleReasons(SellAction self, Board board, GameProgressInformation progress)
        {
            Player player = progress.CurrentPlayer;
            List<BaseIndustry> industries = board.GetIndustriesOnBoardOfGivenPlayer(player.Color)
                .Where(i => i is BaseSellableIndustry && !i.IsFlipped).ToList();
            if (industries.Count == 0) return new List<CantUseSellActionReason> { CantUseSellActionReason.NO_INDUSTRY_TO_SELL };
            if (!industries.Any(i => Merchants(board, i).Count > 0)) return new List<CantUseSellActionReason> { CantUseSellActionReason.NO_ROUTE_TO_MARKET };
            return industries.Any(i => CanIndustryBeSold(self, i, player)) ? new List<CantUseSellActionReason>()
                : new List<CantUseSellActionReason> { CantUseSellActionReason.NO_BEER };
        }
        public static void StartSellProcedure(SellAction self, BaseIndustry industry)
        {
            if (Get<bool>(self, "inProgress"))
            {
                if (!ReferenceEquals(Get<BaseIndustry>(self, "_industry"), industry))
                    throw new InvalidOperationException("Finish the current industry sale first.");
                return;
            }
            if (!CanIndustryBeSold(self, industry, Get<Player>(self, "_player")))
                throw new InvalidOperationException("Chosen industry cannot be sold.");
            Context(self).UsedMerchant = null;
            Set(self, "_industry", industry);
            Set(self, "<NeededBeer>k__BackingField", ((BaseSellableIndustry)industry).CostOfBeerBarrelsToSellIndustry);
            Set(self, "inProgress", true);
            if (!self.IsBeerNeeded()) self.SellIndustry();
        }
        public static IEnumerable<IBeerSource> SaleBeerSources(SellAction action, Board board, Player player, BaseIndustry industry)
        {
            IEnumerable<IBeerSource> sources = Breweries(action, board, industry, player).Cast<IBeerSource>();
            if (Context(action).UsedMerchant == null)
                sources = Merchants(board, industry).Where(m => m.HasBeer).Cast<IBeerSource>().Concat(sources);
            return sources.ToList();
        }
        public static void SupplyBeer(SellAction self, IBeerSource source, int amount)
        {
            if (!Get<bool>(self, "inProgress") || amount <= 0 || amount > self.NeededBeer)
                throw new InvalidOperationException("Invalid beer delivery for this sale.");
            Board board = Property<Board>(self, "Board");
            BaseIndustry industry = Get<BaseIndustry>(self, "_industry");
            Player player = Get<Player>(self, "_player");
            if (!SaleBeerSources(self, board, player, industry).Any(s => ReferenceEquals(s, source)))
                throw new InvalidOperationException("Beer must come from a valid brewery or the single merchant used for this industry.");
            MerchantSlot merchant = source as MerchantSlot;
            if (merchant != null && amount != 1) throw new InvalidOperationException("A merchant supplies one barrel per sale.");
            self.GeneralAction.ConsumingBeer.SupplyBeer(source, amount, self.NeededBeer, player,
                new List<Region> { board.GetRegionContainingGivenIndustry(industry) });
            if (merchant != null) Context(self).UsedMerchant = merchant;
            Set(self, "<NeededBeer>k__BackingField", self.NeededBeer - amount);
        }
        public static IBeerSource FindBeerSource(SellAction action, Board board, Player player, BuildingSlot slot)
        {
            return SaleBeerSources(action, board, player, slot.BuildedIndustry).FirstOrDefault();
        }
        public static BuildingSlot SlotToSell(SellAction action, BaseCard card, Board board, Player player, List<BuildingSlot> candidates)
        {
            List<BuildingSlot> valid = candidates.Where(s => CanIndustryBeSold(action, s.BuildedIndustry, player)).ToList();
            return valid.LastOrDefault(s => s.BuildedIndustry.IsForCanalEra && !s.BuildedIndustry.IsForRailEra)
                ?? valid.FirstOrDefault(s => Merchants(board, s.BuildedIndustry).Any(m => m.HasBeer)) ?? valid.FirstOrDefault();
        }
        public static bool CanIndustryBeSoldButLackingBeer(BrassApplicationGame game, BaseIndustry industry, Player player, Board board)
        {
            return industry is BaseSellableIndustry && !industry.IsFlipped && Merchants(board, industry).Count > 0 &&
                !CanIndustryBeSold(game.Game.TurnFlow.SellAction, industry, player);
        }
        public static void StartBeerDelivery(SellActionUseCase self)
        {
            SellAction action = Get<SellAction>(self, "_sellAction");
            Board board = Get<IGameHistory>(self, "_gameHistory").CurrentGame.Game.Board;
            Set(self, "_sellActionUseCaseState", SellActionUseCaseState.SelectingBeer);
            Get<ISellActionOutput>(self, "_presenter").StartBeerDeliveryUseCase(action.NeededBeer,
                SaleBeerSources(action, board, Get<Player>(self, "_currentPlayer"), Get<BaseIndustry>(self, "_baseIndustry")), self);
        }
        public static void ReplayCompleted(SellActionUseCase self)
        {
            BoardGameRulesGame game = Get<IGameHistory>(self, "_gameHistory").CurrentGame.Game;
            SellAction action = Get<SellAction>(self, "_sellAction");
            if (Get<SellActionUseCaseState>(self, "_sellActionUseCaseState") == SellActionUseCaseState.SelectingBeer &&
                !action.IsBeerNeeded() && game.DevelopMerchantBonus.MerchantDevelopmentNeeded &&
                game.DevelopMerchantBonus.GetDevelopMerchantBonusNotPossibleReasons(game.GameProgressInformation).Count > 0)
                game.DevelopMerchantBonus.FinishDevelop();
            // The patcher preserves the existing state machine (and B12 scopes its bonus reads).
            try { typeof(SellActionUseCase).GetMethod("BrassPatch_Original_ReplayCompleted", Flags).Invoke(self, null); }
            catch (TargetInvocationException error) { throw error.InnerException ?? error; }
        }
    }
}
