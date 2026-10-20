using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassHost
{
    // Compiled only into the additive training runtime. The qualified R1 and
    // installed-game assemblies are never rebuilt or replaced by this path.
    public sealed partial class Host
    {
        int choiceRevision=-1, wildRevision=-1;
        List<Choice> cachedChoices;
        string[][] cachedWild;
        public long ChoiceCacheHits, ChoiceCacheMisses;
        JObject trainingFork;
        public object SaveTrainingFork(bool export=false)
        {
            if(Terminal || Phase!="action")throw new InvalidOperationException("fork_requires_action");
            trainingFork=Checkpoint();
            if(export)return new{revision=Revision,hash=Wire.Hash(Canonical()),checkpoint=trainingFork};
            return new{revision=Revision,hash=Wire.Hash(Canonical())};
        }
        public object ResumeTrainingFork(int? shuffleSeed)
        {
            if(trainingFork==null)throw new InvalidOperationException("missing_training_fork");
            var saved=trainingFork;trainingFork=null;Adopt(Restore(saved));
            return ShuffleTrainingDeck(shuffleSeed);
        }
        public object ShuffleTrainingDeck(int? shuffleSeed)
        {
            if(Terminal || Phase!="action")throw new InvalidOperationException("fork_requires_action");
            if(shuffleSeed.HasValue)
            {
                // Only the unseen deck changes. Hands, public history, wild
                // decks, board state and the engine's own RNG are untouched.
                var rng=new Random(shuffleSeed.Value);var deck=Game.Board.CardDeck;
                for(int i=deck.Count-1;i>0;i--){int j=rng.Next(i+1);var card=deck[i];deck[i]=deck[j];deck[j]=card;}
                commands.Add(new JObject{["training_deck"]=ExportCards()});
            }
            InvalidateFastCaches();
            return Context();
        }
        void InvalidateFastCaches() {choiceRevision=wildRevision=-1; cachedChoices=null; cachedWild=null;}
        public List<Choice> Choices()
        {
            if(cachedChoices!=null && choiceRevision==Revision){ChoiceCacheHits++;return cachedChoices;}
            cachedChoices=UncachedChoices();choiceRevision=Revision;ChoiceCacheMisses++;return cachedChoices;
        }
        string[] KnownWild(int seat)
        {
            if(cachedWild==null || wildRevision!=Revision)
            {cachedWild=Enumerable.Range(0,4).Select(UncachedKnownWild).ToArray();wildRevision=Revision;}
            return cachedWild[seat];
        }
        // Journal entries and initial setup are append-only during Execute.
        // Copy the old prefix only on failure; retain the pre-mutation hash and
        // normal Restore verification, including cards/RNG and chance state.
        Func<JObject> RollbackMark()
        {
            int count=commands.Count,oldSeed=seed;string id=GameId;
            var setup=initial;
#if TRAINING_SEARCH
            string hash=FullHash();bool explicitChance=ExplicitChance;
#else
            string hash=Wire.Hash(Canonical());
#endif
            return ()=>{
                var cp=new JObject{["format"]="brass-trusted-checkpoint",["protocol"]=Wire.Version,
                    ["rules"]=Wire.Rules,["game_id"]=id,["seed"]=oldSeed,["setup"]=setup.DeepClone(),
                    ["commands"]=new JArray(commands.Take(count).Select(x=>x.DeepClone())),["hash"]=hash};
#if TRAINING_SEARCH
                cp["explicit_chance"]=explicitChance;
#endif
                return cp;
            };
        }
        public object Decision()
        {
            return new{context=Context(),view=Terminal?null:Observe(Actor),choices=Choices(),result=Result()};
        }
        public object StepDecision(int revision,string id,int forcedLimit)
        {
            if(forcedLimit<0 || forcedLimit>32)throw new InvalidOperationException("invalid_forced_limit");
            var start=(JObject)Wire.Token(PublicContext());int seat=Actor;
            Apply(revision,id);
            var forced=new List<object>();
            for(int i=0;i<forcedLimit && !Terminal && Actor==seat && Phase!="debt";i++)
            {
                var now=(JObject)Wire.Token(PublicContext());
                if(new[]{"era","round","turn","turn_owner"}.Any(k=>!JToken.DeepEquals(start[k],now[k])))break;
                var legal=Choices();if(legal.Count!=1)break;
                // Materialize BEFORE Execute; no lazy view can describe the next revision.
                var before=Wire.Token(Decision());Apply(Revision,legal[0].id);
                forced.Add(new{before,after=Context()});
            }
            return new{decision=Decision(),forced};
        }
#if TRAINING_SEARCH
        object[] SearchViews()
        {
            // Observe's allowlist has only two seat-dependent fields: seat and
            // own_hand. Materialize the public board once for this snapshot,
            // then replace both private fields before exposing any other seat.
            // These tokens never outlive this reply or span a rules mutation.
            var first=(JObject)Wire.Token(Observe(0));
            return Enumerable.Range(0,4).Select(seat=>{
                var view=first.Properties().ToDictionary(p=>p.Name,p=>(object)p.Value);
                view["seat"]=seat;
                view["own_hand"]=Seats.Single(p=>(int)p.Color==seat).Hand
                    .Select(Wire.Card).OrderBy(x=>x,StringComparer.Ordinal).ToArray();
                return (object)view;
            }).ToArray();
        }
        public object SearchSnapshot()
        {
            return new{context=SearchContext(),chance=ChanceOutcomes(),checkpoint=Checkpoint(),
                views=ChancePending?null:SearchViews(),
                choices=ChancePending?new List<Choice>():Choices(),result=Result()};
        }
#endif
    }
    public static class FastReply
    {
        public static void Write(string id,object data,Stopwatch elapsed,TimeSpan cpu,long hits,long misses)
        {
            var render=Stopwatch.StartNew();string payload=Wire.Json(data);render.Stop();
            var metrics=new{elapsed_seconds=elapsed.Elapsed.TotalSeconds,
                cpu_seconds=(Process.GetCurrentProcess().TotalProcessorTime-cpu).TotalSeconds,
                materialize_serialize_seconds=render.Elapsed.TotalSeconds,choice_cache_hits=hits,choice_cache_misses=misses};
            Console.WriteLine("{\"id\":"+Wire.Json(id)+",\"ok\":true,\"data\":"+payload+",\"native\":"+Wire.Json(metrics)+"}");
        }
    }
}
