// Original extension of the project-authored R1 host. No vendor source is copied.
using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.GamePhases;

namespace BrassHost
{
    public sealed partial class Host
    {
        public bool ExplicitChance;
        string chanceKind="";
        int chanceNeeded;
        readonly List<string> chancePool=new List<string>();
        readonly List<string> chancePicked=new List<string>();
        public bool ChancePending {get{return chanceKind!="";}}
        public object ChanceState() {return new {kind=chanceKind,needed=chanceNeeded,pool=chancePool.ToArray(),picked=chancePicked.ToArray()};}
        public string FullHash() {return ExplicitChance?Wire.Hash(new {state=Canonical(),chance=ChanceState()}):Wire.Hash(Canonical());}
        void AdoptChance(Host other)
        {
            ExplicitChance=other.ExplicitChance;chanceKind=other.chanceKind;chanceNeeded=other.chanceNeeded;
            chancePool.Clear();chancePool.AddRange(other.chancePool);
            chancePicked.Clear();chancePicked.AddRange(other.chancePicked);
        }
        bool TryPauseDraw()
        {
            int n=8-Player.Hand.Count;
            if(!ExplicitChance || n<=0 || Game.Board.CardDeck.Count<n) return false;
            chanceKind="draw";chanceNeeded=n;chancePicked.Clear();chancePool.Clear();
            chancePool.AddRange(Game.Board.CardDeck.Select(Wire.Card).OrderBy(x=>x,StringComparer.Ordinal));
            return true;
        }
        bool TryPauseRail()
        {
            if(!ExplicitChance || Game.GameProgressInformation.CurrentGameEra!=GameEra.CanalEra || Game.GameProgressInformation.CurrentRoundNumber!=8) return false;
            Phase="action"; // All liquidation decisions are settled before this pause.
            chanceKind="rail-shuffle";chancePicked.Clear();chancePool.Clear();
            chancePool.AddRange(Game.Board.CardDeck.Concat(Seats.SelectMany(p=>p.DiscardPile))
                .Select(Wire.Card).Where(x=>!x.StartsWith("wild:")).OrderBy(x=>x,StringComparer.Ordinal));
            if(chancePool.Count!=64) throw new InvalidOperationException("rail_card_conservation");
            chanceNeeded=chancePool.Count;return true;
        }
        public object ChanceOutcomes()
        {
            if(!ChancePending) return null;
            return new {kind=chanceKind,remaining=chanceNeeded-chancePicked.Count,
                outcomes=chancePool.GroupBy(x=>x).OrderBy(g=>g.Key,StringComparer.Ordinal)
                    .Select(g=>new {card=g.Key,probability=(double)g.Count()/chancePool.Count}).ToArray()};
        }
        public void ResolveChance(string card)
        {
            if(!ChancePending || !chancePool.Contains(card))throw new InvalidOperationException("illegal_chance");
            var before=Checkpoint();
            try
            {
                chancePool.Remove(card);chancePicked.Add(card);
                if(chancePicked.Count==chanceNeeded)
                {
                    string kind=chanceKind;var picked=chancePicked.ToArray();var rest=chancePool.ToArray();
                    chanceKind="";chancePool.Clear();chancePicked.Clear();chanceNeeded=0;
                    if(kind=="draw")
                    {
                        var cards=ExportCards();cards["deck"]=new JArray(picked.Concat(rest));ImportCards(cards);
                        Game.GameFlow.EndTurn();Advance();
                    }
                    else
                    {
                        // The rules settle scoring/order/income. Replace the hidden temporary
                        // shuffle/deal before any observation is released to a player.
                        Game.GameFlow.EndRound();
                        discardStarts=Seats.Select(p=>p.DiscardPile.Count).ToArray();
                        var cards=ExportCards();var hands=new JArray(Enumerable.Range(0,4).Select(i=>new JArray()));
                        var order=Game.PlayersTrack.Players.Select(p=>(int)p.Color).ToArray();
                        for(int i=0;i<4;i++)hands[order[i]]=new JArray(picked.Skip(i*8).Take(8));
                        cards["hands"]=hands;cards["deck"]=new JArray(picked.Skip(32));ImportCards(cards);
                        Advance();
                    }
                }
                Revision++;commands.Add(new JObject{["chance"]=card,["cards_after"]=ExportCards()});
            }
            catch {Adopt(Restore(before));throw;}
        }
        public object SearchContext()
        {
            var c=(JObject)Wire.Token(Context());c["chance"]=ChancePending;
            if(ChancePending)c["actor"]=-1;
            return c;
        }
    }
}
