using System;
using System.Collections.Generic;
using System.Linq;
using BrassHost;
using BrassApplication.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    public static class NativeInput
    {
        // Reuse the qualified public event projection; only an actor's permitted
        // view/recall crosses transport. Seed, deck, full events and game objects
        // stay on the trusted game thread.
        public static JObject Capture(BrassApplicationGame app,IList<IEvent> events,int actor,JArray recall)
        {
            int[] debts;var host=Host.CapturePersonality(app,events,out debts);
            var view=host.ObservePersonality(actor);
            return FromView(view,debts,recall);
        }
        // The clock journal consumes only these confirmed boundary fields. A
        // search_v1 continuation has no projected-successor observation to test;
        // rebuilding the full NN/history input here would serve no other reader.
        public static JObject ClockView(BrassApplicationGame app,int actor)
        {
            var progress=app.Game.GameProgressInformation;
            return new JObject{["seat"]=actor,["context"]=new JObject{
                ["revision"]=app.Game.Revision,["era"]=progress.CurrentGameEra.ToString(),
                ["round"]=progress.CurrentRoundNumber,["turn"]=progress.CurrentTurnNumber}};
        }
        public static JObject FromView(JObject view,int[] debts,JArray recall)
        {
            var spent=debts.Select(n=>Math.Min(.075,n*.02)).ToArray();
            var settled=new JArray();var c=view["context"];
            if((string)c["phase"]=="debt")settled.Add(new JArray(c["era"],c["round"]));
            return new JObject{["view"]=view.DeepClone(),["recall"]=(recall??new JArray()).DeepClone(),
                ["context"]=new JObject{["model_id"]=NativePackage.Model,["reward_id"]="a84a3fccf2421d40aefc71c80d3b8839cf3ff22900eb6c5b917b821b5bb8d72e",
                    ["profile_id"]=NativePackage.Profile,["style"]=new JArray(0.0,0.0,0.0,0.0),["intent"]=new JArray(Enumerable.Repeat(0.0,8)),
                    ["remaining"]=new JArray(spent.Select(s=>.075-s)),["max_cost"]=.075,["stable_descriptors"]=JValue.CreateNull()},
                ["reward"]=new JObject{["version"]="brass-combined-return/2.0.0",["win"]=1.0,["placement"]=.1,["vp"]=.001,["vp_cap"]=180,
                    ["insolvency"]=new JObject{["event_cost"]=.02,["max_cost"]=.075}},
                ["ledger"]=new JObject{["spent"]=new JArray(spent),["settled"]=settled},["sample_seed"]=17};
        }
        public static JObject Request(JObject input,string instance,string gameId,long generation,int allowance)
        {
            string serialized=input.ToString(Newtonsoft.Json.Formatting.None),hash;
            using(var sha=System.Security.Cryptography.SHA256.Create())hash=BitConverter.ToString(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(serialized))).Replace("-","").ToLowerInvariant();
            return new JObject{["op"]="search",["ticket"]=new JObject{["game_instance"]=instance,["game_id"]=gameId,["generation"]=generation,
                ["revision"]=input["view"]["context"]["revision"],["actor"]=input["view"]["seat"],["profile"]=NativePackage.Profile,
                ["root_sha256"]=hash,["source_sha256"]="pending"},["allowance_ms"]=allowance,["remaining_ms"]=allowance,["input_json"]=serialized};
        }
        public static string[][] Actions(JObject response)
        {
            var actions=response?["actions"] as JArray;
            if(actions==null || actions.Count<1 || actions.Count>2)throw new InvalidOperationException("Incomplete native actions");
            return actions.Select(a=>{
                var path=a as JArray;
                if(path==null || path.Count<1 || path.Count>256 || path.Any(p=>p.Type!=JTokenType.String || ((string)p).Length>512))throw new InvalidOperationException("Malformed native action");
                return path.Values<string>().ToArray();
            }).ToArray();
        }
        // Rust counts staged semantic choices; Unity counts underlying game
        // events. Compare the full permitted successor after translating only
        // those new revision stamps. The live event revision remains bound by
        // LocalTurnBridge and NativeValidation immediately before each commit.
        public static bool ContinuationMatches(JObject response,JObject actual,JArray priorRecall)
        {
            var revision=response?["search_revision"] as JObject;
            if(revision==null)return true; // Frozen search_v1 contract.
            if((string)revision["version"]!="brass-local-focus/2.0.0")throw new InvalidOperationException("Unknown native focus contract");
            var focus=revision["focus"] as JObject;
            if((bool?)focus?["switched"]!=true)return true;
            var expected=focus["permitted_successor"] as JObject;
            if(expected==null || expected["view"]==null || expected["recall"]==null)
                throw new InvalidOperationException("Missing projected continuation precondition");
            var view=(JObject)actual["view"].DeepClone();
            long liveRevision=(long)view["context"]["revision"];
            long originalRevision=(long)response["ticket"]["revision"];
            if(liveRevision<=originalRevision)return false;
            view["context"]["revision"]=expected["view"]["context"]["revision"].DeepClone();
            if(!JToken.DeepEquals(view,expected["view"]) || !JToken.DeepEquals(actual["context"],expected["context"]) ||
                !JToken.DeepEquals(actual["ledger"],expected["ledger"]))return false;
            var recall=(JArray)priorRecall.DeepClone();var hand=actual["view"]["own_hand"];
            if(recall.Count==0 || !JToken.DeepEquals(recall.Last["hand"],hand))
                recall.Add(new JObject{["revision"]=expected["recall"].Last["revision"].DeepClone(),["hand"]=hand.DeepClone()});
            return JToken.DeepEquals(recall,expected["recall"]);
        }
    }
}
