using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Runtime.CompilerServices;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrassLegacyAI
{
    public static class PersonalityPackage
    {
        public const string Protocol="personality-game-rpc/1.0.0";
        static JObject manifest,pointer,menu;static bool loaded;static string directory;
        public static string Error {get;private set;}
        public static bool ReviewMode {get;private set;}
        public static string DirectoryPath {get{Load();return directory;}}
        public static JObject Manifest {get{Load();if(manifest==null)throw new InvalidDataException(Error??"Qualified neural package not installed");return manifest;}}
        public static JObject Pointer {get{Load();return pointer;}}
        public static bool Available {get{Load();return manifest!=null;}}
        public static string Hash(string path) {using(var s=File.OpenRead(path))using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(s)).Replace("-","").ToLowerInvariant();}
        public static string Below(string root,string relative) {
            if(String.IsNullOrEmpty(relative)||Path.IsPathRooted(relative)||relative.Contains("..")||relative.Contains(":")||relative.Contains("\\"))throw new InvalidDataException("Unsafe package path");
            var path=Path.GetFullPath(Path.Combine(root,relative));
            if(!path.StartsWith(Path.GetFullPath(root)+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))throw new InvalidDataException("Package path escaped");
            return path;
        }
        public static void Load() {
            if(loaded)return;loaded=true;LoadCore();
        }
        // Keep Unity internal calls out of the cache-only path. Native save
        // contract tests can exercise a preloaded fixture outside the player.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static void LoadCore() {
            try {
                string root=Path.GetDirectoryName(Application.dataPath);
                string location=Path.Combine(root,"BrassPersonality/active.json");if(!File.Exists(location))return;
                pointer=JObject.Parse(File.ReadAllText(location));
                if((string)pointer["format"]!="personality-install-pointer/1.0.0")throw new InvalidDataException("Unknown neural installation");
                directory=Below(root,(string)pointer["package"]);
                if(Hash(Path.Combine(directory,"manifest.json"))!=(string)pointer["manifest_sha256"])throw new InvalidDataException("Neural manifest changed");
                var m=JObject.Parse(File.ReadAllText(Path.Combine(directory,"manifest.json")));
                ReviewMode=(bool?)pointer["review_only"]==true && Environment.GetCommandLineArgs().Contains("--brass-personality-review") &&
                    root.Split(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar).Contains("artifacts") && File.Exists(Path.Combine(root,".p6-fixture-sandbox"));
                if((!ReviewMode && ((bool?)m["fixture"]!=false || (bool?)m["qualified_sources"]!=true)) || (string)m["format"]!="personality-laptop-package/1.1.0")throw new InvalidDataException("Unqualified/fixture package");
                foreach(var file in m["files"])if(Hash(Below(directory,(string)file["path"]))!=(string)file["sha256"])throw new InvalidDataException("Neural package file changed");
                if(Hash(typeof(PersonalityPackage).Assembly.Location)!=(string)pointer["installed_helper_sha256"])throw new InvalidDataException("Neural helper changed");
                foreach(var f in ((JObject)pointer["installed_files"]).Properties())
                    if(f.Name!="BrassLegacyAI.dll" && f.Name!="BrassPatch.dll" || Hash(Path.Combine(Application.dataPath,"Managed/"+f.Name))!=(string)f.Value)
                        throw new InvalidDataException("Neural integration helper changed");
                if(!ReviewMode) {
                string receipt=Below(root,(string)pointer["certificate"]);
                if(Hash(receipt)!=(string)pointer["certificate_sha256"])throw new InvalidDataException("Release certificate changed");
                var certificate=JObject.Parse(File.ReadAllText(receipt));
                if((string)certificate["format"]!="personality-laptop-release/1.0.0" || (bool?)certificate["qualified"]!=true ||
                    (bool?)certificate["fixture"]!=false || (string)certificate["package_id"]!=(string)m["id"] ||
                    (string)certificate["backend"]!=(string)pointer["backend"] ||
                    Hash(typeof(PersonalityPackage).Assembly.Location)!=(string)certificate["integration_sha256"])
                    throw new InvalidDataException("Release/profile/integration differs");
                if(!JToken.DeepEquals(certificate["integration_files"],pointer["installed_files"]))throw new InvalidDataException("Released liquidation/turn bridge differs");
                }
                menu=JObject.Parse(File.ReadAllText(Path.Combine(directory,"public-menu.json")));
                if(menu["choices"].Count()!=16 || !ReviewMode && menu["choices"].Any(x=>(bool?)x["enabled"]!=true) || (int)m["profiles"]["workers"]!=2)throw new InvalidDataException("Unqualified style catalog");
                if((string)m["profiles"]["hint_mode"]=="intent-hints" && (bool?)pointer["intent_hints_opt_in"]!=true)throw new InvalidDataException("Broad intent hints require explicit opt-in");
                manifest=m;
            } catch(Exception e) {manifest=null;pointer=null;Error=e.Message;}
        }
        public static IReadOnlyList<AiCatalogEntry> Entries() {
            if(!Available)return Array.AsReadOnly(Enumerable.Range(1,16).Select(i=>new AiCatalogEntry(AiFamily.Gamma,"gamma-"+i.ToString("00"),"Gamma "+i.ToString("00"),"Qualified model pending",false,Error??"Awaiting qualified model")).ToArray());
            return Array.AsReadOnly(menu["choices"].Select(g=>new AiCatalogEntry(AiFamily.Gamma,(string)g["archetype_id"],
                (ReviewMode?((bool)manifest["fixture"]?"FIXTURE: ":"REVIEW: "):"")+(string)g["name"],
                (ReviewMode?"Unreleased. Difficulty labels are not qualification claims. ":"")+(string)g["description"],true)).ToArray());
        }
        public static string Tier(string difficulty) {return difficulty=="mid"?"medium":difficulty;}
    }
}
