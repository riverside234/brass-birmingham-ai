using System;
using System.IO;
using System.Linq;
using BrassHost;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassPatch.Persistence;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    public sealed class NativeTurnClock
    {
        static readonly System.Runtime.CompilerServices.ConditionalWeakTable<BrassApplicationGame,NativeTurnClock> clocks = new System.Runtime.CompilerServices.ConditionalWeakTable<BrassApplicationGame,NativeTurnClock>();
        string turn;
        public readonly string Instance=Guid.NewGuid().ToString("N");
        public System.Diagnostics.Stopwatch Clock=new System.Diagnostics.Stopwatch();
        public static NativeTurnClock For(BrassApplicationGame app,int actor,bool debt)
        {
            var value=clocks.GetValue(app,g=>new NativeTurnClock());var c=app.Game.GameProgressInformation;
            string key=c.CurrentGameEra+":"+c.CurrentRoundNumber+":"+c.CurrentTurnNumber+":"+actor+":"+debt;
            if(value.turn!=key){value.turn=key;value.Clock=new System.Diagnostics.Stopwatch();}
            return value;
        }
    }
    public static class NativeGameState
    {
        public const string Format="brass-local-native-save/1.0.0";
        public static bool IsNative(JObject state) {return (string)state?["format"]==Format || (string)state?["runtime"]==NativePackage.Backend;}
        public static bool Supported(GameMetadata metadata) {return metadata!=null && (int)metadata.GameStartConfiguration.NumberOfPlayers==4 && !metadata.GameStartConfiguration.IsIntroductoryGame;}
        static string AssignmentHash(JObject s) {
            var bound=new JObject{["format"]=s["format"],["runtime"]=s["runtime"],["family"]=s["family"],
                ["rules"]=s["rules"],["schema"]=s["schema"],["configuration"]=s["configuration"],["assignments"]=s["assignments"]};
            if(s["nn_search_k"]!=null)bound["nn_search_k"]=s["nn_search_k"].DeepClone();
            if(s["nn_temperature"]!=null)bound["nn_temperature"]=s["nn_temperature"].DeepClone();
            return Wire.Hash(bound);
        }
        public static int SearchK(JObject state)
        {
            var k=state["nn_search_k"];
            if(k==null)return 16; // Pre-selector saves retain the installed K16 default.
            if(k.Type!=JTokenType.Integer || ((int)k!=8 && (int)k!=16))throw new InvalidDataException("Unsupported saved NN search K");
            return (int)k;
        }
        public static JObject Create(GameMetadata metadata,Func<PlayerMetadata,string> difficulty)
        { return Create(metadata,difficulty,16); }
        public static JObject Create(GameMetadata metadata,Func<PlayerMetadata,string> difficulty,int searchK)
        { return Create(metadata,difficulty,searchK,1.0); }
        public static double Temperature(JObject state)
        {
            var temperature=state["nn_temperature"];
            if(temperature==null)return 1.0;
            if((temperature.Type!=JTokenType.Integer && temperature.Type!=JTokenType.Float) || !NativePackage.ValidTemperature((double)temperature))
                throw new InvalidDataException("Unsupported saved NN temperature");
            return (double)temperature;
        }
        public static JObject Create(GameMetadata metadata,Func<PlayerMetadata,string> difficulty,int searchK,double temperature)
        {
            if(!Supported(metadata))throw new InvalidDataException("Alpha Test requires standard four-player rules");
            if(searchK!=8 && searchK!=16)throw new InvalidDataException("Unsupported NN search K");
            if(!NativePackage.ValidTemperature(temperature))throw new InvalidDataException("Unsupported NN temperature");
            var rows=new JArray(metadata.PlayerMetadata.Select(p=>{
                var r=new JObject{["color"]=(int)p.PlayerStartConfiguration.Color,["sitting_order"]=p.PlayerStartConfiguration.SittingOrder,
                    ["kind"]=p.AiType==AIType.Human?"human":"native"};
                if(p.AiType!=AIType.Human){r["strategy"]="alpha-test";r["backend"]=NativePackage.Backend;r["profile"]=NativePackage.Profile;
                    r["model_id"]=NativePackage.Model;r["export_sha256"]=NativePackage.Export;r["execution"]=DifficultyCatalog.Snapshot(difficulty(p));}
                return r;
            }));
            var state=new JObject{["format"]=Format,["runtime"]=NativePackage.Backend,["family"]="nn",["rules"]=Wire.Rules,["schema"]=Wire.Version,
                ["game_id"]=metadata.GameId,["created_utc"]=DateTime.UtcNow.ToString("o"),["ready"]=false,
                ["configuration"]=Host.Configuration(metadata.GameStartConfiguration),["assignments"]=rows,["records"]=new JArray()};
            state["nn_search_k"]=searchK;
            state["nn_temperature"]=temperature;
            state["assignment_sha256"]=AssignmentHash(state);state["records_sha256"]=Wire.Hash(state["records"]);return state;
        }
        public static JObject Read(GameMetadata metadata,bool pending)
        {
            var s=SaveExtensions.Get(metadata);
            if(!Supported(metadata) || (string)s["format"]!=Format || (string)s["runtime"]!=NativePackage.Backend || (string)s["family"]!="nn" ||
                (string)s["rules"]!=Wire.Rules || (string)s["schema"]!=Wire.Version || (string)s["game_id"]!=metadata.GameId ||
                (!pending && (bool?)s["ready"]!=true) || !JToken.DeepEquals(s["configuration"],Host.Configuration(metadata.GameStartConfiguration)) ||
                (string)s["assignment_sha256"]!=AssignmentHash(s) || !(s["records"] is JArray) ||
                (string)s["records_sha256"]!=Wire.Hash(s["records"]))throw new InvalidDataException("Incompatible NN table; save preserved");
            SearchK(s);
            Temperature(s);
            var rows=s["assignments"].ToArray();
            if(rows.Length!=4 || rows.Select(r=>(int)r["color"]).Distinct().Count()!=4)throw new InvalidDataException("Invalid NN seat mapping");
            foreach(var p in metadata.PlayerMetadata) {
                var r=rows.Single(x=>(int)x["color"]==(int)p.PlayerStartConfiguration.Color);bool human=p.AiType==AIType.Human;
                if((int)r["sitting_order"]!=p.PlayerStartConfiguration.SittingOrder || (string)r["kind"]!=(human?"human":"native"))throw new InvalidDataException("Mixed or changed NN table");
                if(human){if(r["execution"]!=null)throw new InvalidDataException("Unexpected human execution");continue;}
                if((string)r["strategy"]!="alpha-test" || (string)r["backend"]!=NativePackage.Backend || (string)r["profile"]!=NativePackage.Profile ||
                    (string)r["model_id"]!=NativePackage.Model || (string)r["export_sha256"]!=NativePackage.Export)throw new InvalidDataException("Saved NN model differs");
                DifficultyCatalog.Read(r["execution"]);
            }
            return s;
        }
        public static JObject Clock(JObject state,JObject view,string historyHash)
        {
            var c=view["context"];
            return state["records"].OfType<JObject>().LastOrDefault(r=>(int?)r["revision"]==(int)c["revision"] &&
                (int?)r["actor"]==(int)view["seat"] && (string)r["era"]==(string)c["era"] && (int?)r["round"]==(int)c["round"] &&
                (int?)r["turn"]==(int)c["turn"] && (string)r["history_sha256"]==historyHash);
        }
        public static void Record(GameMetadata metadata,JObject view,string historyHash,int remaining,JArray recall)
        {
            var s=Read(metadata,false);var c=view["context"];
            var rows=new JArray(s["records"].Where(r=>(int)r["revision"]<(int)c["revision"]).Select(r=>r.DeepClone()));
            rows.Add(new JObject{["revision"]=c["revision"],["actor"]=view["seat"],["era"]=c["era"],["round"]=c["round"],["turn"]=c["turn"],
                ["history_sha256"]=historyHash,["remaining_ms"]=Math.Max(0,remaining),["recall"]=recall.DeepClone()});
            if(rows.Count>512)throw new InvalidDataException("NN clock journal bound");
            s["records"]=rows;s["records_sha256"]=Wire.Hash(rows);SaveExtensions.Set(metadata,s);
        }
    }
}
