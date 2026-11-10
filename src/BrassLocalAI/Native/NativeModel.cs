using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // Registered models can be selected independently for every seat.
    // The legacy constants/package remain valid for existing Alpha Test games.
    public sealed class NativeModel
    {
        public readonly string Key, Label, Backend, Model, Export, Checkpoint, DirectoryName, Scope, Decision;
        public readonly IReadOnlyList<string> Focuses;
        public readonly string NeutralProfile, Description;
        public readonly string Reward;
        public readonly double InsolvencyCost, MaxCost;
        public static readonly NativeModel Alpha = new NativeModel("alpha-test","Alpha Test",NativePackage.Backend,
            NativePackage.Model,NativePackage.Export,NativePackage.Checkpoint,"alpha-g890","part1-local-alpha-test","user-2026-10-25",NativeFocus.Keys);
        public static readonly NativeModel Beta = new NativeModel("beta","Beta","brass-gumbel-is-mcts/beta-g1290-v1",
            "73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff",
            "1ee3340c0279592eb28cf2ce0bdbd24691e63be60f81d0196984bfc27c921424",
            "ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0",
            "beta-g1290","part1-local-beta-test","user-2026-11-01",Array.AsReadOnly(new[]{"neutral"}),
            "beta-g1290-neutral/1.0.0", "Test neural network g1290 · Neutral context.");
        public static readonly NativeModel IL = new NativeModel("il","IL","brass-gumbel-is-mcts/il-v11-2-v1",
            "c4cdf78a1b90ae9ade7156878009e34ee355ec71ee67226f3fd8a95d09335a94",
            "a42304fa909cbc833cf258bcbe7d046647a272a6a1424def2e6d968bb079a141",
            "657fe1901ab35d144e5d86e16b42e795d281946829c87bcb8e0245927b70a263",
            "il-v11-2","part1-local-il-test","user-2026-11-02",Array.AsReadOnly(new[]{"neutral"}),
            "il-v11-2-neutral/1.0.0", "Imitation-learning model · Neutral context. Search values combine its win, placement and VP predictions.");
        public static readonly NativeModel Beta2 = new NativeModel("beta-2","beta-2","brass-gumbel-is-mcts/beta2-g1190-v1",
            "d872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864",
            "39a536967350d02ffcaa1b9aea6a3239777bae03783fabb58d4ae38e45bb73f9",
            "f228f4dd7f1a913ea11e908cb2ccb53b76cd14ca302da9e2e5479c505d1294f1",
            "beta2-g1190","part1-local-beta2-test","user-2026-11-10",Array.AsReadOnly(new[]{"neutral"}),
            "beta2-g1190-neutral/1.0.0", "Selected neural network g1190. Neutral context.",
            "49711b4abaa95940928db567fa76ab654a5d60c417e1b5eca8dbb44ffb5327c1",.0025,.025);
        // Add future packaged models here; the roster, lookup and seat router share this registry.
        public static readonly IReadOnlyList<NativeModel> All = Array.AsReadOnly(new[]{Alpha,Beta,IL,Beta2});
        NativeModel(string key,string label,string backend,string model,string export,string checkpoint,string directory,string scope,string decision,IReadOnlyList<string> focuses,
            string neutralProfile=null,string description="Current test neural network (g890). Neutral context; this is not the final trained model.",
            string reward="a84a3fccf2421d40aefc71c80d3b8839cf3ff22900eb6c5b917b821b5bb8d72e",double insolvencyCost=.02,double maxCost=.075)
        { Key=key;Label=label;Backend=backend;Model=model;Export=export;Checkpoint=checkpoint;DirectoryName=directory;Scope=scope;Decision=decision;Focuses=focuses;NeutralProfile=neutralProfile;Description=description;Reward=reward;InsolvencyCost=insolvencyCost;MaxCost=maxCost; }
        public string Location { get { return Path.Combine(NativePackage.GameRoot,"BrassNative",DirectoryName); } }
        public bool Enabled { get { return !File.Exists(Path.Combine(NativePackage.GameRoot,"BrassNative","disabled")) && File.Exists(Path.Combine(Location,"build.json")); } }
        public string Profile(string focus)
        {
            if(this==Alpha)return NativeFocus.Profile(focus);
            if(focus!="neutral")throw new InvalidDataException(Label+" supports neutral context only");
            return NeutralProfile;
        }
        public static NativeModel FromChoice(string choice)
        { return All.FirstOrDefault(m=>m.Key==choice) ?? (NativeFocus.FromChoice(choice)!=null?Alpha:throw new InvalidDataException("Unknown NN selection")); }
        public static NativeModel FromBackend(string backend)
        { return All.FirstOrDefault(m=>m.Backend==backend) ?? throw new InvalidDataException("Unknown saved NN backend"); }
        public static NativeModel FromModel(string model)
        { return All.FirstOrDefault(m=>m.Model==model) ?? throw new InvalidDataException("Unknown NN model"); }
        public static NativeModel FromProfile(string profile)
        {
            foreach(var model in All)foreach(string focus in model.Focuses)if(profile==model.Profile(focus))return model;
            throw new InvalidDataException("Unknown NN request profile");
        }
        public static NativeModel FromDirectory(string directory)
        {
            string name=Path.GetFileName(Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar));
            return All.FirstOrDefault(m=>m.DirectoryName==name) ?? throw new InvalidDataException("Unknown NN package directory");
        }
        public static bool ValidChoice(string choice) { return All.Any(m=>m.Key==choice); }
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
