using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Players;

namespace BrassHost
{
    public sealed partial class Host
    {
        object Tile(BoardGameRules.Entities.Industries.BaseIndustry tile)
        {
            if(tile==null)return null;
            var data=(JObject)Wire.ObservationToken(Wire.Tile(tile));data["id"]=tileIds[tile.UniqueId];return data;
        }
        string[] KnownWild(int seat)
        {
            var known=new List<string>();
            string era=Game.GameProgressInformation.CurrentGameEra.ToString();
            foreach(var entry in publicHistory)
            {
                var item=Wire.ObservationToken(entry);
                if((int)item["seat"]!=seat || (string)item["era"]!=era)continue;
                string card=(string)item["choice"]["card"];
                if(card!=null && card.StartsWith("wild:"))known.Remove(card);
                if((string)item["choice"]["kind"]=="scout"){known.Add("wild:industry");known.Add("wild:location");}
            }
            return known.OrderBy(x=>x,StringComparer.Ordinal).ToArray();
        }
        // Explicit allowlist. Do not serialize Game, Player, BaseCard, events or Reflection.
        public JObject ObserveToken(int seat) {return (JObject)Wire.ObservationToken(Observe(seat));}
        public object Observe(int seat)
        {
            if(seat<0 || seat>3) throw new InvalidOperationException("invalid_seat");
            var me=Seats.Single(p=>(int)p.Color==seat);
            bool canal=Game.GameProgressInformation.CurrentGameEra==GameEra.CanalEra;
            return new {protocol=Wire.Version,rules=Wire.Rules,seat,context=PublicContext(),configuration=Configuration(Game.GameStartConfiguration),
                own_hand=me.Hand.Select(Wire.Card).OrderBy(x=>x,StringComparer.Ordinal).ToArray(),
                players=Seats.Select(p=>new{seat=(int)p.Color,cash=p.Money,income_marker=p.IncomeMarkerPosition,income=p.IncomeLevel,vp=p.VP,bonus_vp=p.VPFromBonusesInCurrentEra,link_tiles=p.LinkTileCount,hand_count=p.Hand.Count,
                    spent=Game.PlayersTrack.SpendMoneyInCurrentRound[p.Color],
                    discards=p.DiscardPile.Skip(canal?1:discardStarts[(int)p.Color]).Select(Wire.Card).ToArray(),hidden_discards=canal?1:0,
                    known_wild=KnownWild((int)p.Color),
                    mat=Wire.Types.Select(t=>new{type=t.ToString(),tiles=p.Mat.GetIndustryList(t).Select(Tile).ToArray()}).ToArray(),
                    debt=Phase=="debt"?Game.SellIndustryFlow.RemainingDebtToPay(p.Color):0}).ToArray(),
                order=Game.PlayersTrack.Players.Select(p=>(int)p.Color).ToArray(),
                slots=Slots.Select(s=>new{id=s.Key,types=s.Value.BuildingTypes.Select(t=>t.ToString()).ToArray(),tile=Tile(s.Value.BuildedIndustry)}).ToArray(),
                links=Links.Select(l=>new{id=l.LinkId,owner=l.HasOwner?(int?)l.Owner.Color:null,regions=new[]{l.Region1,l.Region2,l.Region3}.Where(r=>r!=null).Select(r=>r.Name.ToString()).ToArray(),era=l.GetType().Name}).ToArray(),
                merchants=Merchants.Select(m=>new{id=m.Key,types=m.Value.MerchantTile==null?null:m.Value.MerchantTile.BuildingTypes.Select(t=>t.ToString()).ToArray(),beer=m.Value.AvailableBeer,bonus=Game.Board.GetRegionOfGivenMerchantTile(m.Value).MerchantBeerBonusType.ToString()}).ToArray(),
                coal_market=Game.Board.CoalMarket.AvailableCoal,iron_market=Game.Board.IronMarket.AvailableIron,
                deck_count=Game.Board.CardDeck.Count,wild_industry_count=Game.Board.WildIndustryCardDeck.Count,wild_location_count=Game.Board.WildLocationCardDeck.Count,
                pending=new{kind=action,target,merchant,parts=completedParts,
                    coal=action=="build"?Game.TurnFlow.BuildAction.NeededCoal:action=="network"?Game.TurnFlow.NetworkAction.NeededCoal:0,
                    iron=action=="build"?Game.TurnFlow.BuildAction.NeededIron:action=="develop"?Game.TurnFlow.DevelopAction.NeededIron:0,
                    beer=action=="sell"?Game.TurnFlow.SellAction.NeededBeer:action=="network"?Game.TurnFlow.NetworkAction.NeededBeer:0},
                history=publicHistory.ToArray()};
        }
        public object Result()
        {
            if(!Terminal) return null;
            var ranked=Seats.OrderByDescending(p=>p.VP).ThenByDescending(p=>p.IncomeLevel).ThenByDescending(p=>p.Money).ToArray();
            Func<Player,Player,bool> tie=(a,b)=>a.VP==b.VP && a.IncomeLevel==b.IncomeLevel && a.Money==b.Money;
            int winners=ranked.Count(p=>tie(p,ranked[0]));
            return new{objective="win-share/1.0.0",seats=Seats.Select(p=>new{seat=(int)p.Color,vp=p.VP,income=p.IncomeLevel,cash=p.Money,rank=Array.FindIndex(ranked,x=>tie(p,x))+1,utility=tie(p,ranked[0])?1.0/winners:0.0}).ToArray()};
        }
        public object Canonical()
        {
            return new{observation=Observe((int)Seats.First().Color),cards=ExportCards(),result=Result(),
                rng=Game.Randomer.RandomerInitData,engine_revision=Game.Revision,
                round_pending=Game.GameFlow.IsLastTurnOver(),debt_pending=Game.GameFlow.IsSellIndustryNecessary(),
                removed=Game.Board.RemovedIndustries.Select(t=>tileIds[t.UniqueId]).OrderBy(x=>x,StringComparer.Ordinal).ToArray()};
        }
        internal JObject ExportCards()
        {
            return JObject.FromObject(new{hands=Enumerable.Range(0,4).Select(s=>Seats.Where(p=>(int)p.Color==s).SelectMany(p=>p.Hand).Select(Wire.Card).ToArray()).ToArray(),discards=Enumerable.Range(0,4).Select(s=>Seats.Where(p=>(int)p.Color==s).SelectMany(p=>p.DiscardPile).Select(Wire.Card).ToArray()).ToArray(),discard_starts=discardStarts,deck=Game.Board.CardDeck.Select(Wire.Card).ToArray(),wild_industry=Game.Board.WildIndustryCardDeck.Select(Wire.Card).ToArray(),wild_location=Game.Board.WildLocationCardDeck.Select(Wire.Card).ToArray()});
        }
        internal void ImportCards(JObject cards)
        {
            // The Canal engine retains old discard references during the Rail deal. Use unique
            // physical objects; reconstruct lists separately (the aliases are vendor semantics).
            var all=Seats.SelectMany(p=>p.Hand.Concat(p.DiscardPile)).Concat(Game.Board.CardDeck).Concat(Game.Board.WildIndustryCardDeck).Concat(Game.Board.WildLocationCardDeck).GroupBy(c=>c.UniqueId).Select(g=>g.First()).ToList();
            var available=all.GroupBy(Wire.Card).ToDictionary(g=>g.Key,g=>g.ToList());
            var cursor=available.Keys.ToDictionary(k=>k,k=>0);
            Func<JToken,List<BaseCard>> take=arr=>arr.Select(n=>{
                string key=(string)n; if(!available.ContainsKey(key)) throw new InvalidOperationException("setup_card_unknown:"+key);
                int index=cursor[key]++;
                if(index>=available[key].Count) throw new InvalidOperationException("setup_card_conservation:"+key);
                return available[key][index];}).ToList();
            var ps=Seats.ToArray();
            var hands=((JArray)cards["hands"]).Select(take).ToArray();
            var deck=take(cards["deck"]); var wi=take(cards["wild_industry"]); var wl=take(cards["wild_location"]);
            discardStarts=cards["discard_starts"].Select(x=>(int)x).ToArray();
            var discards=((JArray)cards["discards"]).Select((list,i)=>{
                var old=list.Take(discardStarts[i]).Select(n=>available[(string)n][0]).ToList();
                old.AddRange(take(new JArray(list.Skip(discardStarts[i])))); return old;
            }).ToArray();
            foreach(var p in ps){int i=(int)p.Color;p.Hand.Clear();p.Hand.AddRange(hands[i]);p.DiscardPile.Clear();p.DiscardPile.AddRange(discards[i]);}
            Game.Board.CardDeck.Clear();Game.Board.CardDeck.AddRange(deck);
            Game.Board.WildIndustryCardDeck.Clear();Game.Board.WildIndustryCardDeck.AddRange(wi);
            Game.Board.WildLocationCardDeck.Clear();Game.Board.WildLocationCardDeck.AddRange(wl);
        }
        JObject ExportSetup()
        {
            return new JObject{["configuration"]=Configuration(Game.GameStartConfiguration),["cards"]=ExportCards(),["order"]=new JArray(Game.PlayersTrack.Players.Select(p=>(int)p.Color)),["merchants"]=new JArray(Merchants.Select(m=>new JObject{["id"]=m.Key,["types"]=m.Value.MerchantTile==null?null:new JArray(m.Value.MerchantTile.BuildingTypes.Select(t=>t.ToString()))}))};
        }
        void ImportSetup(JObject setup)
        {
            ImportCards((JObject)setup["cards"]);
            var players=Seats.ToDictionary(p=>(int)p.Color); Game.PlayersTrack.Players.Clear();
            foreach(var s in (JArray)setup["order"]) Game.PlayersTrack.Players.Add(players[(int)s]);
            Game.GameProgressInformation.CurrentPlayer=Game.PlayersTrack.Players[0];
            Game.TurnFlow.StartTurn(Player,1,GameEra.CanalEra);
            foreach(JObject m in (JArray)setup["merchants"])
            {
                var slot=Merchant((string)m["id"]);
                slot.SetMerchant(m["types"]==null || m["types"].Type==JTokenType.Null?null:new MerchantTile(Game.GameStartConfiguration.NumberOfPlayers,m["types"].Select(t=>Wire.Type((string)t)).ToList()));
            }
        }
    }
}
