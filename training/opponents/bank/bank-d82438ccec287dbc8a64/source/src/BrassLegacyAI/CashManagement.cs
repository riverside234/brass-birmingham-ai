using System;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    public sealed class CashProjection
    {
        public bool collection, credit_blocked, future_credit_blocked;
        public int payment, cash_after_income, shortfall, reserve_gap;
        public double penalty;
    }

    // Uses only cash and income already achieved by this candidate. A possible
    // future flip is investment upside, never cash available to pay today's bill.
    public static class CashManagement
    {
        public static CashProjection Project(JObject view, int seat)
        {
            var context = view["context"]; var player = Conversion.Player(view, seat);
            int cash = (int)player["cash"], income = (int)player["income"];
            bool terminal = (bool)context["terminal"];
            bool canal = (string)context["era"] == "CanalEra";
            bool lastRound = (int)context["round"] == Conversion.RoundLimit(view);
            bool anotherTurn = !lastRound || (canal && !Conversion.Intro(view));
            var result = new CashProjection { collection = !terminal && (canal || !lastRound), credit_blocked = income < -7 };
            result.future_credit_blocked = !terminal && anotherTurn && result.credit_blocked;
            result.payment = result.collection ? Math.Max(0, -income) : 0;
            result.cash_after_income = cash + (result.collection ? income : 0);
            result.shortfall = Math.Max(0, -result.cash_after_income);
            // At the borrowing floor, keep one more income payment in view and
            // modest working capital for recovery. This is a soft preference,
            // not a ban on an investment that actually raises income this turn.
            if (!terminal && anotherTurn && result.credit_blocked)
                result.reserve_gap = Math.Max(0, Math.Max(0, -income) + 7 - Math.Max(0, result.cash_after_income));
            result.penalty = result.shortfall * 1.5 + result.reserve_gap * 0.75;
            return result;
        }
    }
}
