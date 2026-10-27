using System;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // Error-path diagnostics only. No live game, event stream, deck or other
    // players' hands enter this writer. A full-state fixture belongs elsewhere.
    public static class NativeFailureBundle
    {
        static readonly object gate = new object();
        public const int MaximumFiles = 16;
        public const int MaximumBytes = 2 * 1024 * 1024;
        public static string Location { get { return Path.Combine(NativePackage.GameRoot,"artifacts","native-failures"); } }

        public static string Write(string directory,JObject request,JObject response,JObject package,
            JObject model,JObject readiness,string stage,string error,double elapsedMs,string[] path)
        {
            // Failure reporting must never replace the original error or cause
            // another action. Saturation preserves earlier evidence, with no deletion.
            try {
                if(request==null)return null;
                var bundle=new JObject {
                    ["format"]="brass-native-failure/1.0.0",["created_utc"]=DateTime.UtcNow.ToString("o"),
                    ["stage"]=stage,["error"]=error,["elapsed_ms"]=elapsedMs,
                    ["expected"]="Discard this result; commit nothing from this failed request; retain the NN family and original allowance.",
                    ["request"]=request.DeepClone(),["response"]=response?.DeepClone(),
                    ["semantic_path"]=path==null?JValue.CreateNull():(JToken)new JArray(path),
                    ["policy"]=package?.DeepClone(),["model_manifest"]=model?.DeepClone(),["readiness"]=readiness?.DeepClone(),
                    ["trusted_full_state_included"]=false,
                    ["replay_note"]="The permitted root, recall, pinned policy and per-owner draw receipts reproduce fixed work. Wall-clock scheduling is not deterministic. Aborted workers may have no reply/draw receipt."
                };
                byte[] bytes=new UTF8Encoding(false).GetBytes(bundle.ToString(Formatting.Indented));
                if(bytes.Length>MaximumBytes)return null;
                lock(gate) {
                    Directory.CreateDirectory(directory);
                    if(Directory.EnumerateFiles(directory,"native-failure-*.json*").Take(MaximumFiles).Count()>=MaximumFiles)return null;
                    string target=Path.Combine(directory,"native-failure-"+Guid.NewGuid().ToString("N")+".json");
                    string pending=target+".pending";
                    using(var file=new FileStream(pending,FileMode.CreateNew,FileAccess.Write,FileShare.None)) {
                        file.Write(bytes,0,bytes.Length);file.Flush();
                    }
                    File.Move(pending,target);return target;
                }
            } catch {return null;}
        }
    }
}
