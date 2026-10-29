using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;

namespace BrassHost
{
    // One active host per worker. No policy receives this object, a checkpoint or an event.
    public sealed partial class Host
    {
        public BoardGameRulesGame Game {get; private set;}
        public int Revision {get; private set;}
        public string GameId {get; private set;}=Guid.NewGuid().ToString("N");
        public void InvalidateRequestsAfterRestore() {GameId=Guid.NewGuid().ToString("N");}
        int seed;
        public string Phase {get; private set;}="action";
        string action="", target="", merchant="";
        int completedParts;
        JObject initial;
        readonly List<JObject> commands=new List<JObject>();
        readonly List<object> publicHistory=new List<object>();
        int[] discardStarts=new int[4];
        int turnOwner;
        readonly Dictionary<string,BaseIndustry> tileCatalog=new Dictionary<string,BaseIndustry>();
        readonly Dictionary<string,string> tileIds=new Dictionary<string,string>();
        BoardGameRulesGame indexedGame;
        GameEra indexedEra;
        KeyValuePair<string,BuildingSlot>[] indexedSlots;
        KeyValuePair<string,MerchantSlot>[] indexedMerchants;
        Link[] indexedLinks;
        public Player Player {get{return Game.GameProgressInformation.CurrentPlayer;}}
        public bool Terminal {get{return Game.GameProgressInformation.IsGameOver;}}
        public IEnumerable<Player> Seats {get{return Game.PlayersTrack.Players.OrderBy(p=>(int)p.Color);}}
        // Topology is stable within an era. Cache object indexes, never mutable
        // legality/results; rebuild when capture/load/adopt replaces the game
        // or Canal -> Rail replaces the link graph. Tile/owner fields stay live.
        void IndexBoard()
        {
            var era=Game.GameProgressInformation.CurrentGameEra;
            if(ReferenceEquals(indexedGame,Game) && indexedEra==era)return;
            indexedSlots=Game.Board.Regions.OrderBy(r=>r.Name.ToString(),StringComparer.Ordinal).SelectMany(r=>r.BuildingSlots.Select((s,i)=>new KeyValuePair<string,BuildingSlot>(r.Name+":"+i,s))).ToArray();
            indexedMerchants=Game.Board.Regions.OfType<BorderRegion>().OrderBy(r=>r.Name.ToString(),StringComparer.Ordinal).SelectMany(r=>r.MerchantSlots.Select((s,i)=>new KeyValuePair<string,MerchantSlot>(r.Name+":"+i,s))).ToArray();
            indexedLinks=Game.Board.Regions.SelectMany(r=>r.Links).Distinct().OrderBy(l=>l.LinkId,StringComparer.Ordinal).ToArray();
            indexedGame=Game;indexedEra=era;
        }
        public IEnumerable<KeyValuePair<string,BuildingSlot>> Slots { get {IndexBoard();return indexedSlots;}}
        public IEnumerable<KeyValuePair<string,MerchantSlot>> Merchants {get{IndexBoard();return indexedMerchants;}}
        public IEnumerable<Link> Links {get{IndexBoard();return indexedLinks;}}
        BuildingSlot Slot(string key) {return Slots.Single(s=>s.Key==key).Value;}
        MerchantSlot Merchant(string key) {return Merchants.Single(s=>s.Key==key).Value;}
        Link Link(string key) {return Links.Single(l=>l.LinkId==key);}
        Region RegionOf(BuildingSlot slot) {return Game.Board.GetRegionContainingGivenSlot(slot);}
        BaseCard Card(string key) {return Player.Hand.First(c=>Wire.Card(c)==key);}
        IEnumerable<string> Cards {get{return Player.Hand.Select(Wire.Card).Distinct().OrderBy(x=>x,StringComparer.Ordinal);}}
        public Host(int seed, JObject setup=null)
        {
            this.seed=seed;
            Game=new BoardGameRulesGame(ReadConfiguration(seed, setup));
            Game.StartGame();
            foreach(var p in Seats) foreach(var t in Wire.Types)
            {
                var list=p.Mat.GetIndustryList(t);
                for(int i=0;i<list.Count;i++){string id=(int)p.Color+":"+t+":"+i;tileCatalog.Add(id,list[i]);tileIds.Add(list[i].UniqueId,id);}
            }
            if(setup?["cards"] != null) ImportSetup(setup);
            turnOwner=(int)Player.Color;
            if(setup!=null && setup["position"] is JObject) ImportPosition((JObject)setup["position"]);
            initial=ExportSetup();
            if(setup!=null && setup["position"] is JObject) initial["position"]=setup["position"].DeepClone();
        }
        public int Actor
        {
            get
            {
                if(Terminal) return -1;
                if(Phase=="debt") return Seats.Where(p=>!Game.SellIndustryFlow.CanIndustrySellingBeCompleted(p.Color)).Select(p=>(int)p.Color).First();
                return (int)Player.Color;
            }
        }
        public object PublicContext() {return new{protocol=Wire.Version,rules=Wire.Rules,revision=Revision,actor=Actor,turn_owner=turnOwner,phase=Phase,actions_remaining=Game.TurnFlow.GetRemainingActionsCount(),era=Game.GameProgressInformation.CurrentGameEra.ToString(),round=Game.GameProgressInformation.CurrentRoundNumber,turn=Game.GameProgressInformation.CurrentTurnNumber,terminal=Terminal,round_limit=Game.GameFlow.GetNumberOfRounds(Game.GameStartConfiguration.NumberOfPlayers),introductory=Game.GameStartConfiguration.IsIntroductoryGame,player_count=(int)Game.GameStartConfiguration.NumberOfPlayers};}
        public object Context() {var c=(JObject)Wire.Token(PublicContext());c["game_id"]=GameId;return c;}
        public void Apply(int revision, string id)
        {
            if(revision!=Revision) throw new InvalidOperationException("stale_revision");
            Choice c=Choices().SingleOrDefault(x=>x.id==id);
            if(c==null) throw new InvalidOperationException("illegal_choice");
            var before=Checkpoint();
            try {Execute(c); Revision++; commands.Add(new JObject{["choice"]=id,["cards_after"]=ExportCards()});}
            catch {Adopt(Restore(before)); throw;}
        }
        public void ApplyAction(int revision, string[] ids)
        {
            if(revision!=Revision || Phase!="action" || ids==null || ids.Length==0) throw new InvalidOperationException("invalid_action_boundary");
            var trial=Restore(Checkpoint());
            for(int i=0;i<ids.Length;i++)
            {
                trial.Apply(trial.Revision,ids[i]);
                if(trial.action=="" && i!=ids.Length-1) throw new InvalidOperationException("more_than_one_action");
            }
            if(trial.action!="") throw new InvalidOperationException("incomplete_action");
            Adopt(trial);
        }
        void Adopt(Host other)
        {
            Game=other.Game; GameId=other.GameId; Revision=other.Revision; seed=other.seed; Phase=other.Phase; action=other.action; target=other.target; merchant=other.merchant; completedParts=other.completedParts; initial=(JObject)other.initial.DeepClone();
            commands.Clear(); commands.AddRange(other.commands.Select(x=>(JObject)x.DeepClone()));
            publicHistory.Clear(); publicHistory.AddRange(other.publicHistory.Select(Wire.Token));
            discardStarts=(int[])other.discardStarts.Clone();
            turnOwner=other.turnOwner;
            tileCatalog.Clear();foreach(var pair in other.tileCatalog)tileCatalog.Add(pair.Key,pair.Value);
            tileIds.Clear();foreach(var pair in other.tileIds)tileIds.Add(pair.Key,pair.Value);
        }
        void Execute(Choice c)
        {
            int actor=Actor;
            string era=Game.GameProgressInformation.CurrentGameEra.ToString();
            if(Phase=="debt") {Game.SellIndustryFlow.SellIndustry((PlayerColor)actor,Slot(c.target)); Advance();}
            else if(Phase=="bonus")
            {
                if(c.kind=="bonus_skip") Game.DevelopMerchantBonus.FinishDevelop();
                else Game.DevelopMerchantBonus.Develop(Player,Wire.Type(c.industry));
                SetPhase();
            }
            else if(c.kind=="finish") {Game.TurnFlow.ConfirmCurrentAction(); action="";target="";merchant="";completedParts=0;Phase="action";Advance();}
            else if(c.kind=="coal" || c.kind=="iron" || c.kind=="beer")
            {
                var a=(BaseAction)Game.TurnFlow.CurrentAction;
                if(c.kind=="coal") a.SupplyCoal(c.target=="market"?Game.Board.CoalMarket:(ICoalSource)Slot(c.target).BuildedIndustry,1);
                if(c.kind=="iron") a.SupplyIron(c.target=="market"?Game.Board.IronMarket:(IIronSource)Slot(c.target).BuildedIndustry,1);
                if(c.kind=="beer") a.SupplyBeer(c.target.StartsWith("merchant:")?(IBeerSource)Merchant(c.target.Substring(9)):(IBeerSource)Slot(c.target).BuildedIndustry,1);
                if(action=="sell" && Game.TurnFlow.SellAction.NeededBeer==0) Game.TurnFlow.SellAction.SellIndustry();
                SetPhase();
            }
            else
            {
                if(Phase=="action")
                {
                    action=c.kind; completedParts=0;
                    BaseAction a=null;
                    switch(c.kind)
                    {
                        case "build": a=Game.TurnFlow.StartBuildAction(); break;
                        case "network": a=Game.TurnFlow.StartNetworkAction(); break;
                        case "sell": a=Game.TurnFlow.StartSellAction(); break;
                        case "develop": a=Game.TurnFlow.StartDevelopAction(); break;
                        case "loan": a=Game.TurnFlow.StartLoanAction(); break;
                        case "pass": a=Game.TurnFlow.StartPassAction(); break;
                        case "scout": a=Game.TurnFlow.StartScoutAction(); break;
                        default: throw new InvalidOperationException("unknown_action");
                    }
                    if(c.kind=="pass") Game.TurnFlow.PassAction.Pass(Card(c.card),Player);
                    else a.DiscardCard(Card(c.card),Player);
                }
                target=c.target??target; merchant=c.merchant??merchant;
                switch(c.kind)
                {
                    case "build": Game.TurnFlow.BuildAction.Build(Slot(target),Wire.Type(c.industry)); break;
                    case "network": Game.TurnFlow.NetworkAction.PlaceLink(Link(target)); break;
                    case "sell": Game.TurnFlow.SellAction.StartSellProcedure(Slot(target).BuildedIndustry); break;
                    case "develop": Game.TurnFlow.DevelopAction.Develop(Wire.Type(c.industry)); break;
                    case "loan": Game.TurnFlow.LoanAction.Loan(Player.Color); break;
                    case "scout":
                        var trade=new List<BaseCard>();
                        foreach(var name in c.cards) trade.Add(Player.Hand.First(x=>Wire.Card(x)==name && !trade.Contains(x)));
                        Game.TurnFlow.ScoutAction.Scout(trade,Player); break;
                }
                completedParts++;
                SetPhase();
            }
            // Only validated, observable choices. Never include engine event payloads.
            publicHistory.Add(new {seat=actor,era,choice=c});
        }
        void SetPhase()
        {
            if(Game.DevelopMerchantBonus.MerchantDevelopmentNeeded){Phase="bonus";return;}
            bool paying=action=="build"?(Game.TurnFlow.BuildAction.NeededCoal+Game.TurnFlow.BuildAction.NeededIron>0):action=="network"?(Game.TurnFlow.NetworkAction.NeededCoal+Game.TurnFlow.NetworkAction.NeededBeer>0):action=="sell"?Game.TurnFlow.SellAction.NeededBeer>0:action=="develop"?Game.TurnFlow.DevelopAction.NeededIron>0:false;
            Phase=paying?"payment":"continuation";
        }
        public void Advance()
        {
            if(Terminal){Phase="terminal";return;}
            if(action!="") return;
            if(Game.GameFlow.IsLastTurnOver())
            {
                if(Game.GameFlow.IsSellIndustryNecessary() && !Game.SellIndustryFlow.CanIndustrySellingBeCompleted()){Phase="debt";return;}
                var era=Game.GameProgressInformation.CurrentGameEra;
                Game.GameFlow.EndRound();
                if(era==GameEra.CanalEra && Game.GameProgressInformation.CurrentGameEra==GameEra.RailEra)
                    CaptureDiscardStarts();
            }
            else if(Game.TurnFlow.GetRemainingActionsCount()==0)
            {
                Game.GameFlow.EndTurn();
                Advance(); return;
            }
            Phase=Terminal?"terminal":"action";
            if(Phase=="action")turnOwner=(int)Player.Color;
        }
        public JObject Checkpoint()
        {
            return new JObject{["format"]="brass-trusted-checkpoint",["protocol"]=Wire.Version,["rules"]=Wire.Rules,["game_id"]=GameId,["seed"]=seed,["setup"]=initial.DeepClone(),["commands"]=new JArray(commands.Select(x=>x.DeepClone())),["hash"]=Wire.Hash(Canonical())};
        }
        public static Host Restore(JObject cp)
        {
            if((string)cp["format"]!="brass-trusted-checkpoint" || (string)cp["protocol"]!=Wire.Version || (string)cp["rules"]!=Wire.Rules) throw new InvalidOperationException("checkpoint_contract_mismatch");
            var h=new Host((int)cp["seed"],(JObject)cp["setup"]);
            h.GameId=(string)cp["game_id"];
            foreach(JObject cmd in (JArray)cp["commands"])
            {
                var c=h.Choices().SingleOrDefault(x=>x.id==(string)cmd["choice"]);
                if(c==null) throw new InvalidOperationException("replay_illegal_choice");
                h.Execute(c);
                if(!JToken.DeepEquals(h.ExportCards(),cmd["cards_after"]))
                {
                    if(h.action!="") throw new InvalidOperationException("nonchance_card_mismatch");
                    h.ImportCards((JObject)cmd["cards_after"]);
                }
                h.Revision++; h.commands.Add((JObject)cmd.DeepClone());
            }
            if(Wire.Hash(h.Canonical())!=(string)cp["hash"]) throw new InvalidOperationException("checkpoint_hash_mismatch");
            return h;
        }
    }
}
