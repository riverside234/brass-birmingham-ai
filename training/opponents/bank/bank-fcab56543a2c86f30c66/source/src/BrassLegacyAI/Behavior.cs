using System;
using System.Threading;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;
using BoardGameRules.Utils.Logging;
namespace BrassLegacyAI
{
    // Prepared-action adapter; never serialized into a vendor save. The async router
    // owns thinking and supplies this one-shot action after isolated validation.
    public sealed class LegacyBehavior:IAiBehavior
    {
        readonly BrassApplicationGame expected;
        readonly int revision;
        Action apply;
        public LegacyBehavior(BrassApplicationGame expected,Action apply){this.expected=expected;revision=expected.Game.Revision;this.apply=apply;}
        public Action GetNextAIMove(BrassApplicationGame app)
        {
            if(!ReferenceEquals(app,expected) || app.Game.Revision!=revision)throw new InvalidOperationException("stale_prepared_action");
            return ()=>{
                if(app.Game.Revision!=revision)throw new InvalidOperationException("stale_prepared_action");
                var command=Interlocked.Exchange(ref apply,null);
                if(command==null)throw new InvalidOperationException("action_already_consumed");
                command(); // Includes exactly one confirmation; caller must not add one.
            };
        }
        public void PrintNextAIMove(BrassApplicationGame app,ILog logger)
        {logger.Debug("Prepared legacy action: no stock selector or live applicability calls.");}
    }
}
