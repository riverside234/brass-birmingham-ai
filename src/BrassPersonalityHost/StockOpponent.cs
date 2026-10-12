// P5-only trusted evaluation bridge. Reuses original Hard and the R3 stock path.
// This executable is built under artifacts; it is never installed into Unity.
using System;
using System.Linq;
using BrassHost;
using BrassApplication.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.SellIndustry.Events;

namespace BrassPersonalityHost
{
    public sealed class StockOpponent
    {
        readonly GameMetadata metadata;
        public StockOpponent(int seed)
        {
            var config = new GameStartConfiguration(NumberOfPlayers.Four) { Seed = seed };
            metadata = new GameMetadata(config, GameType.Local);
            foreach (var start in config.Players)
            {
                var p = new PlayerMetadata { PlayerStartConfiguration = start,
                    PlayerNetworkId = "p5-stock-" + start.Color, NickName = "Original Hard",
                    AiType = AIType.Hard };
                p.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(p);
                metadata.JoinGame(p);
            }
        }
        public object Step(Host host)
        {
            int actor = host.Actor;
            if (host.Terminal) throw new InvalidOperationException("stock_terminal");
            if (host.Phase == "debt")
            {
                // Same original liquidation selector as the frozen R3 harness.
                var copy = new Host(1, host.TrustedSetup()); int revision = copy.Game.Revision;
                var app = new BrassApplicationGame(copy.Game, metadata);
                var selector = RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());
                selector.Select(app).MakeAction(app).Invoke();
                var sold = copy.Game.GetUncommittedEvents().OfType<IndustrySold>()
                    .First(e => e.Metadata.Revision > revision && (int)e.PlayerColor == actor);
                string slot = copy.Slots.Single(s => s.Value.UniqueId == sold.SlotUniqueId).Key;
                host.Apply(host.Revision, host.Choices().Single(c => c.target == slot).id);
                return new { kind = "liquidation" };
            }
            var live = new BrassApplicationGame(host.Game, metadata);
            return host.StockStep(live, () => metadata.GetPlayerMetadataByColor(host.Player.Color)
                .AIBehaviour.GetNextAIMove(live));
        }
    }
}
