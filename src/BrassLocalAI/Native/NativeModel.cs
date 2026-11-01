using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // Model selection is bound to the whole NN table and its signed saves.
    // The legacy constants/package remain valid for existing Alpha Test games.
    public sealed class NativeModel
    {
        public readonly string Key, Label, Backend, Model, Export, Checkpoint, DirectoryName, Scope, Decision;
        public readonly IReadOnlyList<string> Focuses;
        public static readonly NativeModel Alpha = new NativeModel("alpha-test","Alpha Test",NativePackage.Backend,
            NativePackage.Model,NativePackage.Export,NativePackage.Checkpoint,"alpha-g890","part1-local-alpha-test","user-2026-10-25",NativeFocus.Keys);
        public static readonly NativeModel Beta = new NativeModel("beta","Beta","brass-gumbel-is-mcts/beta-g1290-v1",
            "73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff",
            "1ee3340c0279592eb28cf2ce0bdbd24691e63be60f81d0196984bfc27c921424",
            "ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0",
            "beta-g1290","part1-local-beta-test","user-2026-11-01",Array.AsReadOnly(new[]{"neutral"}));
        NativeModel(string key,string label,string backend,string model,string export,string checkpoint,string directory,string scope,string decision,IReadOnlyList<string> focuses)
        { Key=key;Label=label;Backend=backend;Model=model;Export=export;Checkpoint=checkpoint;DirectoryName=directory;Scope=scope;Decision=decision;Focuses=focuses; }
        public string Location { get { return Path.Combine(NativePackage.GameRoot,"BrassNative",DirectoryName); } }
        public bool Enabled { get { return !File.Exists(Path.Combine(NativePackage.GameRoot,"BrassNative","disabled")) && File.Exists(Path.Combine(Location,"build.json")); } }
        public string Profile(string focus)
        {
            if(this==Alpha)return NativeFocus.Profile(focus);
            if(focus!="neutral")throw new InvalidDataException("Beta supports neutral context only");
            return "beta-g1290-neutral/1.0.0";
        }
        public static NativeModel FromChoice(string choice)
        { if(choice=="beta")return Beta;if(NativeFocus.FromChoice(choice)!=null)return Alpha;throw new InvalidDataException("Unknown NN selection"); }
        public static NativeModel FromBackend(string backend)
        { if(backend==Alpha.Backend)return Alpha;if(backend==Beta.Backend)return Beta;throw new InvalidDataException("Unknown saved NN backend"); }
        public static NativeModel FromModel(string model)
        { if(model==Alpha.Model)return Alpha;if(model==Beta.Model)return Beta;throw new InvalidDataException("Unknown NN model"); }
        public static NativeModel FromProfile(string profile)
        {
            if(profile==Beta.Profile("neutral"))return Beta;
            foreach(string focus in NativeFocus.Keys)if(profile==Alpha.Profile(focus))return Alpha;
            throw new InvalidDataException("Unknown NN request profile");
        }
        public static NativeModel FromDirectory(string directory)
        {
            string name=Path.GetFileName(Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar));
            if(name==Alpha.DirectoryName)return Alpha;if(name==Beta.DirectoryName)return Beta;
            throw new InvalidDataException("Unknown NN package directory");
        }
        public static bool ValidChoice(string choice) { return choice=="alpha-test" || choice=="beta"; }
        public static bool BetaReceipt(JToken r)
        {
            return r!=null && (string)r["version"]=="migration-utility-readonly-validation/1.0.0" &&
                (string)r["mode"]=="read-only-validation" && (bool?)r["ready"]==true && (bool?)r["completed"]==true &&
                (bool?)r["weights_unchanged"]==true && r["steps"]?.Type==JTokenType.Integer && (int)r["steps"]==0 &&
                (string)r["descriptors"]=="all-unknown" && (string)r["source_model_id"]==Beta.Model &&
                (string)r["reward_id"]=="a84a3fccf2421d40aefc71c80d3b8839cf3ff22900eb6c5b917b821b5bb8d72e" &&
                (string)r["request_id"]=="f56bd03a91e661c5be0f941765ec9420c4729ef6caca0f8d294d61b3d3550670" &&
                (string)r["source_checkpoint_sha256"]=="e14d5b7a4596944ac7dec1b30c676b52ab79beb7bbf2e55a8b8540cc8a312937";
        }
    }
}
