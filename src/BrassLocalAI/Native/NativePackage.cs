using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // This admission is deliberately model-specific and independent of measured
    // readiness. Never alter the research manifest's null calibration receipt.
    public sealed class NativePackage
    {
        public const string Protocol = "brass-local-native/1.0.0";
        public const string Backend = "brass-gumbel-is-mcts/alpha-g890-v1";
        public const string Model = "1145fecba3289511df1c518bb71fbfb5b2f6df351777151a5eb999b89cba0761";
        public const string Export = "64e4a8340f4beb7bbc4ee70219b5e461b1185bcde9e25c4bfc78e30936c91ae9";
        public const string Checkpoint = "ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8";
        public const string Profile = "alpha-test-g890-neutral/1.0.0";
        public readonly string Directory, Source, PolicyHash;
        public readonly JObject Policy, ModelManifest;
        public static string GameRoot { get { return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(typeof(NativePackage).Assembly.Location), "../..")); } }
        public static string Location { get { return Path.Combine(GameRoot, "BrassNative", "alpha-g890"); } }
        public static bool Enabled { get { return !File.Exists(Path.Combine(GameRoot, "BrassNative", "disabled")) && File.Exists(Path.Combine(Location, "build.json")); } }
        public static string Hash(string path) { using(var sha=SHA256.Create()) using(var stream=File.OpenRead(path)) return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant(); }
        public NativePackage(string directory)
        {
            Directory=Path.GetFullPath(directory);
            var build=JObject.Parse(File.ReadAllText(Path.Combine(Directory,"build.json")));
            if((string)build["format"]!="brass-local-package/1.0.0" || (string)build["backend"]!=Backend ||
                (string)build["target"]!="x86_64-pc-windows-gnu" || (string)build["scope"]!="local-alpha-test") throw new InvalidDataException("Unsupported native package");
            var files=(JArray)build["files"];
            string[] required={"brass-ai-local.exe","policy.json","warmup.json","model/manifest.json","model/model.json","model/model.onnx","model/onnxruntime.dll"};
            if(files.Count!=required.Length || required.Any(n=>files.Count(f=>(string)f["path"]==n)!=1)) throw new InvalidDataException("Incomplete native package");
            foreach(var file in files) {
                string name=(string)file["path"];
                if(!required.Contains(name) || Hash(Path.Combine(Directory,name))!=(string)file["sha256"]) throw new InvalidDataException("Native package file changed");
            }
            Source=(string)build["source_sha256"]; PolicyHash=Hash(Path.Combine(Directory,"policy.json"));
            Policy=JObject.Parse(File.ReadAllText(Path.Combine(Directory,"policy.json")));
            ModelManifest=JObject.Parse(File.ReadAllText(Path.Combine(Directory,"model/manifest.json")));
            ValidateAcceptance((JObject)Policy["acceptance"],ModelManifest);
            if((string)Policy["protocol"]!=Protocol || (string)Policy["backend"]!=Backend ||
                (string)Policy["model_manifest_sha256"]!=Hash(Path.Combine(Directory,"model/manifest.json"))) throw new InvalidDataException("Native package contract changed");
        }
        public static void ValidateAcceptance(JObject acceptance,JObject model)
        {
            if(acceptance==null || model==null || (string)acceptance["version"]!="brass-local-utility-acceptance/1.0.0" ||
                (string)acceptance["scope"]!="part1-local-alpha-test" || (string)acceptance["decision"]!="user-2026-10-25" ||
                (string)acceptance["model_id"]!=Model || (string)acceptance["export_sha256"]!=Export || (string)acceptance["checkpoint_sha256"]!=Checkpoint ||
                (bool?)acceptance["final_model"]!=false || acceptance["utility_calibration"]?.Type!=JTokenType.Null ||
                (string)model["model_id"]!=Model || (string)model["export_sha256"]!=Export || (string)model["checkpoint_sha256"]!=Checkpoint ||
                model["utility_calibration"]?.Type!=JTokenType.Null || (bool?)model["mechanical_only"]!=true)
                throw new InvalidDataException("Model has no exact local utility-head acceptance");
        }
    }
}
