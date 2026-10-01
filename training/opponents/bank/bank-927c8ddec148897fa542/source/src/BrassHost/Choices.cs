using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;

namespace BrassHost
{
    public sealed partial class Host
    {
        public List<Choice> Choices()
        {
            var result=EnumerateChoices().OrderBy(x=>x.id,StringComparer.Ordinal).ToList();
            if(result.Select(x=>x.id).Distinct().Count()!=result.Count) throw new InvalidOperationException("duplicate_semantic_choice");
            if(!Terminal && result.Count==0) throw new InvalidOperationException("decision_dead_end:"+Phase+":"+action);
            return result;
        }
        IEnumerable<Choice> EnumerateChoices()
        {
            if(Terminal) yield break;
            if(Phase=="debt")
            {
                foreach(var s in Slots.Where(s=>s.Value.BuildedIndustry!=null && (int)s.Value.BuildedIndustry.Color==Actor)) yield return new Choice("liquidate",s.Key);
                yield break;
            }
            if(Phase=="bonus")
            {
                yield return new Choice("bonus_skip");
                foreach(var t in Developable()) yield return new Choice("bonus_develop",industry:t.ToString());
                yield break;
            }
            if(Phase=="payment")
            {
                int coal=action=="build"?Game.TurnFlow.BuildAction.NeededCoal:action=="network"?Game.TurnFlow.NetworkAction.NeededCoal:0;
                int iron=action=="build"?Game.TurnFlow.BuildAction.NeededIron:action=="develop"?Game.TurnFlow.DevelopAction.NeededIron:0;
                int beer=action=="sell"?Game.TurnFlow.SellAction.NeededBeer:action=="network"?Game.TurnFlow.NetworkAction.NeededBeer:0;
                if(coal>0) foreach(var s in CoalSources(TargetRegions())) yield return new Choice("coal",s);
                if(iron>0) foreach(var s in IronSources()) yield return new Choice("iron",s);
                if(beer>0)
                {
                    foreach(var s in BeerSources(TargetRegions())) yield return new Choice("beer",s);
                    if(action=="sell" && Merchant(merchant).HasBeer) yield return new Choice("beer","merchant:"+merchant);
                }
                yield break;
            }
            if(Phase=="continuation")
            {
                yield return new Choice("finish");
                if(action=="develop" && completedParts<2 && IronSources().Any()) foreach(var t in Developable()) yield return new Choice("develop",industry:t.ToString());
                if(action=="network" && completedParts<2 && Game.GameProgressInformation.CurrentGameEra==GameEra.RailEra)
                    foreach(var l in Game.TurnFlow.NetworkAction.GetValidConnectionSlots(Game.Board,Game.GameProgressInformation)) yield return new Choice("network",l.LinkId);
                if(action=="sell") foreach(var c in Sales(null)) yield return c;
                yield break;
            }
            foreach(var card in Cards)
            {
                yield return new Choice("pass",card:card);
                if(Player.IncomeLevel>=-7) yield return new Choice("loan",card:card);
                foreach(var s in Slots)
                foreach(var t in s.Value.BuildingTypes.Distinct())
                {
                    var tile=Player.Mat.GetFirstBuildingOfType(t);
                    if(tile!=null && BuildAction.CanUseBuildActionForGivenSlot(s.Value,tile,Player,Game.Board,Game.GameProgressInformation)
                        && BuildAction.IsCardValidForBuildingInSlot(Card(card),s.Value,t,Player,Game.Board,Game.GameProgressInformation))
                        yield return new Choice("build",s.Key,t.ToString(),card);
                }
                if(IronSources().Any()) foreach(var t in Developable()) yield return new Choice("develop",industry:t.ToString(),card:card);
                foreach(var l in Game.TurnFlow.NetworkAction.GetValidConnectionSlots(Game.Board,Game.GameProgressInformation)) yield return new Choice("network",l.LinkId,card:card);
                foreach(var c in Sales(card)) yield return c;
            }
            if(Player.Hand.Count>=3 && !Player.Hand.Any(c=>c is WildIndustryCard || c is WildLocationCard) && Game.Board.WildIndustryCardDeck.Count>0 && Game.Board.WildLocationCardDeck.Count>0)
            {
                // Physical duplicates are interchangeable; enumerate each multiset exactly once.
                var hand=Player.Hand.Select(Wire.Card).OrderBy(x=>x,StringComparer.Ordinal).ToArray();
                var seen=new HashSet<string>();
                for(int a=0;a<hand.Length;a++) for(int b=a+1;b<hand.Length;b++) for(int c=b+1;c<hand.Length;c++)
                {
                    var choice=new Choice("scout",card:hand[a],cards:new[]{hand[b],hand[c]});
                    if(seen.Add(choice.id)) yield return choice;
                }
            }
        }
        IEnumerable<BuildingType> Developable() {return Wire.Types.Where(t=>Player.Mat.GetFirstBuildingOfType(t)!=null && Player.Mat.GetFirstBuildingOfType(t).CanBeDeveloped);}
        IEnumerable<Region> TargetRegions()
        {
            if(action=="network") {var l=Link(target); return new[]{l.Region1,l.Region2,l.Region3}.Where(r=>r!=null);}
            return new[]{RegionOf(Slot(target))};
        }
        IEnumerable<string> CoalSources(IEnumerable<Region> targets)
        {
            var regions=targets.ToList();
            var nearest=BrassPatch.ResourcesFixes.ClosestCoalMines(Game.Board,regions);
            if(nearest.Count>0) return Slots.Where(s=>nearest.Contains(s.Value.BuildedIndustry as CoalMine)).Select(s=>s.Key).ToList();
            bool connected=Game.Board.GetBorderRegions(r=>true).Any(m=>regions.Any(r=>r==m || Game.Board.DoesRouteExists(r,m)));
            return connected && Player.Money>=Game.Board.CoalMarket.CoalPrice(1)?new[]{"market"}:new string[0];
        }
        IEnumerable<string> IronSources()
        {
            var board=Slots.Where(s=>s.Value.BuildedIndustry is IronWorks && ((IronWorks)s.Value.BuildedIndustry).HasIron).Select(s=>s.Key).ToArray();
            return board.Length>0?board:Player.Money>=Game.Board.IronMarket.IronPrice(1)?new[]{"market"}:new string[0];
        }
        IEnumerable<string> BeerSources(IEnumerable<Region> targets)
        {
            var regions=targets.ToList();
            return Slots.Where(s=>s.Value.BuildedIndustry is Brewery && ((Brewery)s.Value.BuildedIndustry).HasBeer && (s.Value.BuildedIndustry.Color==Player.Color || regions.Any(r=>r==RegionOf(s.Value) || Game.Board.DoesRouteExists(r,RegionOf(s.Value))))).Select(s=>s.Key).ToArray();
        }
        IEnumerable<Choice> Sales(string card)
        {
            foreach(var s in Slots.Where(s=>s.Value.BuildedIndustry is BaseSellableIndustry && s.Value.BuildedIndustry.Color==Player.Color && !s.Value.BuildedIndustry.IsFlipped))
            {
                var tile=(BaseSellableIndustry)s.Value.BuildedIndustry;
                var r=RegionOf(s.Value);
                int ownBeer=BeerSources(new[]{r}).Sum(k=>((Brewery)Slot(k).BuildedIndustry).AvailableBeer);
                foreach(var m in Merchants.Where(m=>m.Value.MerchantTile!=null && m.Value.MerchantTile.BuildingTypes.Contains(tile.BuildingType)))
                {
                    var mr=Game.Board.GetRegionOfGivenMerchantTile(m.Value);
                    if((r==mr || Game.Board.DoesRouteExists(r,mr)) && ownBeer+m.Value.AvailableBeer>=tile.CostOfBeerBarrelsToSellIndustry)
                        yield return new Choice("sell",s.Key,card:card,merchant:m.Key);
                }
            }
        }
    }
}
