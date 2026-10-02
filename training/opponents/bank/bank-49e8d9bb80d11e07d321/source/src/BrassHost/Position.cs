using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;
using BoardGameRules.Entities.Common;

namespace BrassHost
{
    public sealed partial class Host
    {
        // Version-pinned reconstruction adapter, not a serializer of arbitrary game objects.
        // Reflection is confined to fields for which the vendor supplies no restore API.
        static void Set(object owner,string name,object value)
        {
            for(Type t=owner.GetType();t!=null;t=t.BaseType)
            {
                var f=t.GetField(name,BindingFlags.Instance|BindingFlags.NonPublic|BindingFlags.Public|BindingFlags.DeclaredOnly);
                if(f!=null){f.SetValue(owner,value);return;}
                var p=t.GetProperty(name,BindingFlags.Instance|BindingFlags.NonPublic|BindingFlags.Public|BindingFlags.DeclaredOnly);
                if(p!=null){p.SetValue(owner,value,null);return;}
            }
            throw new InvalidOperationException("restore_member_missing:"+name);
        }
        void ImportPosition(JObject view)
        {
            if((string)view["protocol"]!=Wire.Version || (string)view["rules"]!=Wire.Rules || view["players"].Count()!=4)throw new InvalidOperationException("position_contract");
            var c=view["context"];
            if((bool)c["terminal"])throw new InvalidOperationException("sample_terminal_not_supported");
            Revision=(int)c["revision"];Phase=(string)c["phase"];turnOwner=(int)c["turn_owner"];
            var progress=Game.GameProgressInformation;
            progress.CurrentGameEra=(GameEra)Enum.Parse(typeof(GameEra),(string)c["era"]);
            progress.CurrentRoundNumber=(int)c["round"];progress.CurrentTurnNumber=(int)c["turn"];
            var players=Seats.ToArray();Game.PlayersTrack.Players=view["order"].Select(s=>players[(int)s]).ToList();
            progress.CurrentPlayer=players[Phase=="debt"?(int)c["actor"]:turnOwner];
            Game.TurnFlow.StartTurn(players[turnOwner],progress.CurrentRoundNumber,progress.CurrentGameEra);
            Set(Game.TurnFlow,"RemainingActionsCount",(int)c["actions_remaining"]);
            var used=new HashSet<string>();
            Func<JToken,BaseIndustry> tile=token=>{
                string id=(string)token["id"];
                if(!used.Add(id))throw new InvalidOperationException("duplicate_tile");
                BaseIndustry t=tileCatalog[id];t.IsFlipped=(bool)token["flipped"];
                if(t is CoalMine)((CoalMine)t).AvailableCoal=(int)token["resource"];
                if(t is IronWorks)((IronWorks)t).AvailableIron=(int)token["resource"];
                if(t is Brewery)((Brewery)t).AvailableBeer=(int)token["resource"];
                return t;
            };
            foreach(var data in view["players"])
            {
                var p=players[(int)data["seat"]];p.Money=(int)data["cash"];p.IncomeMarkerPosition=(int)data["income_marker"];p.VP=(int)data["vp"];p.VPFromBonusesInCurrentEra=(int)data["bonus_vp"];p.LinkTileCount=(int)data["link_tiles"];
                Game.PlayersTrack.SpendMoneyInCurrentRound[p.Color]=(int)data["spent"];
                foreach(var mat in data["mat"]){var list=p.Mat.GetIndustryList(Wire.Type((string)mat["type"]));list.Clear();list.AddRange(mat["tiles"].Select(tile));}
            }
            foreach(var s in view["slots"])Slot((string)s["id"]).BuildedIndustry=s["tile"]==null || s["tile"].Type==JTokenType.Null?null:tile(s["tile"]);
            Game.Board.RemovedIndustries.Clear();Game.Board.RemovedIndustries.AddRange(tileCatalog.Where(kv=>!used.Contains(kv.Key)).Select(kv=>kv.Value));
            foreach(var l in view["links"]){var link=Link((string)l["id"]);link.RemoveOwner();if(l["owner"]!=null && l["owner"].Type!=JTokenType.Null)link.SetOwner(players[(int)l["owner"]]);}
            foreach(var m in view["merchants"]){var slot=Merchant((string)m["id"]);slot.SetMerchant(m["types"]==null || m["types"].Type==JTokenType.Null?null:new MerchantTile(NumberOfPlayers.Four,m["types"].Select(t=>Wire.Type((string)t)).ToList()));if((int)m["beer"]==0)slot.GetBeer();}
            Set(Game.Board.CoalMarket,"Amount",(int)view["coal_market"]);Set(Game.Board.IronMarket,"Amount",(int)view["iron_market"]);
            publicHistory.Clear();publicHistory.AddRange(view["history"].Select(x=>(object)x.DeepClone()));
            var pending=view["pending"];action=(string)pending["kind"];target=(string)pending["target"];merchant=(string)pending["merchant"];completedParts=(int)pending["parts"];
            if(Phase=="debt")
            {
                Set(Game.GameFlow,"_shouldEndRoundBeCalled",true);Set(Game.GameFlow,"_isSellIndustryNecessary",true);
                Game.SellIndustryFlow.StartIndustrySelling();
                foreach(var p in view["players"])Game.SellIndustryFlow._moneyToPayBySelling[(PlayerColor)(int)p["seat"]]=(int)p["debt"];
                progress.CurrentPlayer=players[(int)c["actor"]];
            }
            if(action!="")
            {
                BaseAction a;
                switch(action)
                {
                    case "build":
                        a=Game.TurnFlow.StartBuildAction();Set(a,"_chosenSlot",Slot(target));Set(a,"_chosenIndustry",Slot(target).BuildedIndustry);Set(a,"_chosenBuildingType",Slot(target).BuildedIndustry.BuildingType);
                        Set(a,"NeededCoal",(int)pending["coal"]);Set(a,"NeededIron",(int)pending["iron"]);break;
                    case "network":a=Game.TurnFlow.StartNetworkAction();Set(a,"_linkCounter",completedParts);Set(a,"Link",Link(target));Set(a,"NeededCoal",(int)pending["coal"]);Set(a,"NeededBeer",(int)pending["beer"]);break;
                    case "develop":a=Game.TurnFlow.StartDevelopAction();Set(a,"DevelopCounter",completedParts);Set(a,"NeededIron",(int)pending["iron"]);break;
                    case "sell":a=Game.TurnFlow.StartSellAction();Set(a,"_industry",Slot(target).BuildedIndustry);Set(a,"inProgress",(int)pending["beer"]>0);Set(a,"NeededBeer",(int)pending["beer"]);break;
                    case "loan":a=Game.TurnFlow.StartLoanAction();break;
                    case "pass":a=Game.TurnFlow.StartPassAction();break;
                    case "scout":a=Game.TurnFlow.StartScoutAction();break;
                    default:throw new InvalidOperationException("position_action");
                }
                Set(a,"_player",Player);
            }
            Game.DevelopMerchantBonus.MerchantDevelopmentNeeded=Phase=="bonus";
        }
    }
}
