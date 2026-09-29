// Trusted R3 harness extensions. The R1 executable and installed rules are unchanged.
using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Game;
using BrassApplication.Game;
namespace BrassHost
{
    public sealed partial class Host
    {
        // Only isolated search worlds use this path. Stop BEFORE replenishment/era changes:
        // ordinary future draws must never influence either action of a current turn.
        public void SearchApply(string id)
        {
            var c=Choices().Single(x=>x.id==id);
            if(c.kind=="finish")
            {
                Game.TurnFlow.ConfirmCurrentAction(); action="";target="";merchant="";
                completedParts=0;Phase="action";
            }
            else Execute(c);
            Revision++;
        }
        public bool ActionComplete {get{return action=="";}}
        public int Remaining {get{return Game.TurnFlow.GetRemainingActionsCount();}}
        // Live harness applies a prevalidated action exactly once and owns lifecycle.
        public void CommitPlan(string[] ids)
        {
            if(Phase=="debt") {if(ids.Length!=1)throw new Exception("debt_plan");Apply(Revision,ids[0]);return;}
            var check=new Host(1,TrustedSetup());
            for(int i=0;i<ids.Length;i++)
            {
                check.SearchApply(ids[i]);
                if(check.ActionComplete && i!=ids.Length-1)throw new Exception("multiple_actions");
            }
            if(!check.ActionComplete)throw new Exception("partial_action");
            // Trial retains exact real cards, including no new draw yet. Preserve the R1
            // replay journal; then advance once on the committed world.
            foreach(var id in ids) Apply(Revision,id);
        }
        public object StockStep(BrassApplicationGame app, Func<Action> select)
        {
            if(Phase!="action" || action!="" || !ReferenceEquals(app.Game,Game))throw new Exception("stock_boundary");
            var p=Player;int seat=(int)p.Color;int n=p.DiscardPile.Count;
            string era=Game.GameProgressInformation.CurrentGameEra.ToString();
            var move=select();if(move==null)throw new Exception("stock_null_move");move();
            string kind=Game.TurnFlow.CurrentAction.GetType().Name.Replace("Action","").ToLowerInvariant();
            var discarded=p.DiscardPile.Skip(n).Select(Wire.Card).ToList();
            // Wild cards return to their deck instead of entering DiscardPile. Capture
            // the publicly played action card, not any unplayed hand contents.
            var chosen=(BoardGameRules.Entities.Cards.BaseCard)typeof(BoardGameRules.Entities.Actions.BaseAction).GetField("_chosenCard",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance).GetValue(Game.TurnFlow.CurrentAction);
            if(chosen!=null && Wire.Card(chosen).StartsWith("wild:"))discarded.Insert(0,Wire.Card(chosen));
            var cards=discarded.ToArray();
            // These are public discards, never selected-rule helper internals.
            publicHistory.Add(new{seat,era,choice=new Choice(kind,card:cards.FirstOrDefault(),cards:cards.Skip(1).ToArray())});
            Game.TurnFlow.ConfirmCurrentAction();Revision++;Advance();
            // A stock action cannot be replayed using R1 commands. Rebase at this trusted
            // boundary with public position + private cards, for transaction checks only.
            Rebase();
            return new{kind,cards};
        }
        public void Rebase()
        {
            if(Terminal)return;
            initial=ExportSetup();initial["position"]=Wire.Token(Observe(Actor<0?(int)Seats.First().Color:Actor));
            commands.Clear();
        }
        public JObject TrustedSetup()
        {
            var setup=ExportSetup();setup["position"]=Wire.Token(Observe(Actor<0?(int)Seats.First().Color:Actor));return setup;
        }
    }
}
