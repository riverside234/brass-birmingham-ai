using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassApplication.AI;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassPatch;
using BrassPatch.Persistence;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine.UI;

public sealed partial class LocalAiUnityDriver
{
    void Beta2Restored()
    {
        var p=Setup.PlayerMetadata;
        Check(p[0].AiType==AIType.Human,"beta-2 keeps human seat");
        Check(p.Skip(1).Select(x=>LocalMenu.Choice(x).EntryId).SequenceEqual(new[]{"beta-2","beta","il"}),"beta-2 and existing model identities persist per seat");
        Check(p.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"independent 4/8/12s persist");
        Check(LocalMenu.TableSearchK(p[1])==16 && LocalMenu.TableTemperature(p[1])==1.0 && LocalMenu.CanStart(menu),"beta-2 settings persist and table is playable");
    }
    // The existing driver invokes this method under BRASS_BETA_UI_TEST.
    IEnumerator BetaUiCases()
    {
        bool read=Environment.GetEnvironmentVariable("BRASS_BETA_UI_TEST")=="read";
        string pref=Environment.GetEnvironmentVariable("BRASS_BETA2_PREFERENCES");MixedPreferences().Clear();
        if(read)foreach(var item in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(pref)))MixedPreferences().Add(item.Key,item.Value);
        Call(menu,"CreateNewGameConfiguration");yield return null;
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);yield return null;
            MixedPick(1,"beta-2",0);yield return null;MixedPick(2,"beta",1);yield return null;MixedPick(3,"il",3);yield return null;
            Beta2Restored();
            var before=JsonConvert.SerializeObject(MixedPreferences());
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
            Check(panel.Tiles.Select(b=>b.name).OrderBy(x=>x).SequenceEqual(new[]{"alpha-test","beta","beta-2","il"}),"four distinct NN choices including exact beta-2 label");
            panel.Tiles.Single(b=>b.name=="alpha-test").onClick.Invoke();panel.DifficultyButtons[2].onClick.Invoke();panel.Close();yield return null;
            Check(before==JsonConvert.SerializeObject(MixedPreferences()),"Cancel preserves beta-2 preference");
            MixedPick(1,"balanced");yield return null;Check(LocalMenu.CanStart(menu),"existing V4 selection remains available");
            MixedPick(1,"beta-2");yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            var larger=panel.GetComponentsInChildren<Button>().Single(b=>b.name=="Larger text");larger.onClick.Invoke();
            panel.GetComponentsInChildren<Button>().Single(b=>b.name=="Larger text").onClick.Invoke();yield return null;
            panel.GridScroll.verticalNormalizedPosition=0;
            yield return null;Capture("beta2-1080-largest.png");panel.Close();yield return null;
            File.WriteAllText(pref,JsonConvert.SerializeObject(MixedPreferences()));
            Call(menu,"CreateNewGameConfiguration");yield return null;
        }
        Beta2Restored();
        foreach(int k in new[]{8,16,32})foreach(double t in new[]{1.0,1.5}) {
            var package=new NativePackage(NativeModel.Beta2.Location,k,t);
            Check(package.ModelSpec==NativeModel.Beta2 && (string)package.ModelManifest["reward_id"]==NativeModel.Beta2.Reward,"g1190 model and original reward admitted");
            var wrong=(JObject)package.ModelManifest.DeepClone();wrong["reward_id"]=NativeModel.Beta.Reward;
            bool rejected=false;try{NativePackage.ValidateAcceptance((JObject)package.Policy["acceptance"],wrong);}catch(InvalidDataException){rejected=true;}
            Check(rejected,"g1190 cannot silently use old Beta reward");
            wrong=(JObject)package.Policy["acceptance"].DeepClone();wrong["checkpoint_sha256"]=NativeModel.Beta.Checkpoint;
            rejected=false;try{NativePackage.ValidateAcceptance(wrong,package.ModelManifest);}catch(InvalidDataException){rejected=true;}
            Check(rejected,"wrong checkpoint rejected");
        }
        foreach(var model in NativeModel.All)new NativePackage(model.Location,16,1.0);
        var warm=JObject.Parse(File.ReadAllText(Path.Combine(NativeModel.Beta2.Location,"warmup.json")));
        var input=NativeInput.FromView((JObject)warm["view"],new[]{0,1,10,20},(JArray)warm["recall"],"neutral",NativeModel.Beta2);
        Check(((JArray)input["ledger"]["spent"]).Values<double>().SequenceEqual(new[]{0.0,.0025,.025,.025}),"g1190 debt penalty and cap match checkpoint");
        Check((double)input["reward"]["insolvency"]["event_cost"]==.0025 && (double)input["context"]["max_cost"]==.025,"g1190 reward reaches Rust input");
        var app=menu.StartGame(93311);Check(app!=null,"menu starts beta-2 mixed game");
        var state=LocalGameState.Read(app.Metadata);
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(state,LocalGameState.Read(loaded.Metadata)),"beta-2 disk save/load preserves identity and settings");
        Check(state["assignments"].Any(x=>(string)x["model_id"]==NativeModel.Beta2.Model),"save binds g1190 weights");
        var bad=(JObject)state.DeepClone();bad["assignments"].First(x=>(string)x["model_id"]==NativeModel.Beta2.Model)["model_id"]=NativeModel.Beta.Model;
        SaveExtensions.Set(loaded.Metadata,bad);bool invalid=false;
        try{LocalGameState.Read(loaded.Metadata);}catch(InvalidDataException){invalid=true;}
        Check(invalid,"beta-2 model substitution rejected on load");
    }
}
