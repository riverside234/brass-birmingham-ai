using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassHost;
using BrassPatch.Persistence;
using BrassApplication.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    public static class PersonalityState
    {
        public const string Format="brass-personality-game-save/1.0.0";
        static JObject Envelope(JObject value) {
            if(value==null)return null;
            string format=(string)value["format"];
            if(format!=Format) {
                if(format!=null && format.StartsWith("brass-personality-game-save/",StringComparison.Ordinal))
                    throw new InvalidDataException("Unsupported neural save version; use its compatible package. State has not been reset.");
                return null;
            }
            if((string)value["integrity"]!=Wire.Hash(new JObject(value.Properties().Where(p=>p.Name!="integrity").Select(p=>new JProperty(p.Name,p.Value.DeepClone())))))throw new InvalidDataException("Neural save integrity differs");
            if(!PersonalityPackage.Available || (string)value["package_id"]!=(string)PersonalityPackage.Manifest["id"])throw new InvalidDataException("Restore this save's pinned neural package; it cannot be rerolled or substituted");
            if(!(value["journal"] is JArray) || value["journal"].Count()>512)throw new InvalidDataException("Invalid neural journal");
            return value;
        }
        static void Seal(JObject value) {value.Remove("integrity");value["integrity"]=Wire.Hash(value);}
        public static JObject Unwrap(JObject value) {var e=Envelope(value);return e==null?value:(JObject)e["legacy"].DeepClone();}
        public static void SetLegacy(GameMetadata metadata,JObject legacy) {
            var e=Envelope(SaveExtensions.Get(metadata));if(e==null){SaveExtensions.Set(metadata,legacy);return;}
            e["legacy"]=legacy;Seal(e);SaveExtensions.Set(metadata,e);
        }
        public static JObject Assignment(GameMetadata metadata,int seat) {
            var e=Envelope(SaveExtensions.Get(metadata));if(e==null)return null;
            return e["initial"]["seats"].Select(x=>(JObject)x["selection"]).SingleOrDefault(x=>(int)x["seat"]==seat);
        }
        public static void Attach(GameMetadata metadata,JObject legacy,Func<PlayerMetadata,AiChoice> choice,Func<PlayerMetadata,string> difficulty) {
            var assignments=new JArray(metadata.PlayerMetadata.Where(p=>p.AiType!=BrassApplication.AI.AIType.Human && choice(p).Family==AiFamily.Gamma)
                .Select(p=>new JObject{["seat"]=(int)p.PlayerStartConfiguration.Color,["archetype"]=choice(p).EntryId,["tier"]=PersonalityPackage.Tier(difficulty(p))}));
            if(assignments.Count==0){SaveExtensions.Set(metadata,legacy);return;}
            if((int)metadata.GameStartConfiguration.NumberOfPlayers!=4 || metadata.GameStartConfiguration.IsIntroductoryGame)throw new InvalidDataException("Neural personalities require standard four-player rules");
            var initial=PersonalityRpc.Call("initialize",new JObject{["game_id"]=metadata.GameId,["seed"]=Guid.NewGuid().ToString("N"),["assignments"]=assignments});
            // The metadata persistence hook binds game_id on every save. Include
            // it before sealing so that the first native save preserves integrity.
            var outer=new JObject{["format"]=Format,["game_id"]=metadata.GameId,["package_id"]=PersonalityPackage.Manifest["id"],["legacy"]=legacy,
                ["initial"]=initial,["journal"]=new JArray()};Seal(outer);SaveExtensions.Set(metadata,outer);
        }
        public static JObject Snapshot(GameMetadata metadata,int revision,IList<IEvent> events,out JArray pending) {
            var e=Envelope(SaveExtensions.Get(metadata));if(e==null)throw new InvalidDataException("No neural assignment");
            pending=null;
            foreach(var r in e["journal"].Where(x=>(int)x["revision"]<=revision).OrderByDescending(x=>(int)x["revision"])) {
                if((string)r["history"]!=LocalGameState.HistoryHash(events,(int)r["revision"]))continue;
                if((int)r["revision"]==revision)pending=(JArray)r["pending"].DeepClone();
                return (JObject)r["save"].DeepClone();
            }
            // A rewind to before the first neural decision restores the original
            // stream. Existing completed turns never get a fresh draw.
            return (JObject)e["initial"].DeepClone();
        }
        public static void Commit(GameMetadata metadata,int revision,IList<IEvent> events,JObject save,JArray pending) {
            var e=Envelope(SaveExtensions.Get(metadata));
            if(e==null || (string)save["package_id"]!=(string)e["package_id"] || (string)save["game_id"]!=metadata.GameId)throw new InvalidDataException("Foreign neural save");
            var records=new JArray(e["journal"].Where(r=>(int)r["revision"]<revision &&
                (string)r["history"]==LocalGameState.HistoryHash(events,(int)r["revision"])).Select(r=>r.DeepClone()));
            if(records.Count>=512)throw new InvalidDataException("Neural save journal limit exceeded");
            records.Add(new JObject{["revision"]=revision,["history"]=LocalGameState.HistoryHash(events,revision),["save"]=save,["pending"]=pending});
            e["journal"]=records;Seal(e);SaveExtensions.Set(metadata,e);
        }
    }
}
