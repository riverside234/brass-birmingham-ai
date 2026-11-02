using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassPatch;
using BrassPatch.Persistence;
using BrassApplication.AI;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.UI;

public sealed partial class LocalAiUnityDriver
{
    Dictionary<string,string> BetaPreferences()
    { return (Dictionary<string,string>)typeof(LocalMenu).GetField("TestPreferences",ReflectionAccess.Flags).GetValue(null); }
    string BetaPreferenceFile { get { return Path.Combine(Path.GetDirectoryName(Output),"beta-preferences.json"); } }
    Button BetaButton(string name) { return AiSelectionPanel.Active.GetComponentsInChildren<Button>().Single(b=>b.name==name); }
    void PickBeta(int seat,int difficulty)
    {
        Rows()[seat].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        panel.SwitchFamily(AiFamily.AlphaTest);BetaButton("Beta table").onClick.Invoke();
        panel.DifficultyButtons[difficulty].onClick.Invoke();Check(panel.ApplyButton.interactable,"Beta is selectable");panel.ApplyButton.onClick.Invoke();
    }
    void BetaRestored()
    {
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).EntryId=="beta"),"Beta applies to the whole table including remembered Human choices");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"Beta seats retain independent 4/8/12s settings");
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableNeuralChoice(p)=="beta") && LocalMenu.CanStart(menu),"restored Beta table is playable");
    }
    IEnumerator BetaUiCases()
    {
        bool read=Environment.GetEnvironmentVariable("BRASS_BETA_UI_TEST")=="read";BetaPreferences().Clear();
        if(read)foreach(var item in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(BetaPreferenceFile)))BetaPreferences().Add(item.Key,item.Value);
        Call(menu,"CreateNewGameConfiguration");yield return null;
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);yield return null;
            Check(AiSelectionCatalog.Entries(AiFamily.AlphaTest).Select(e=>e.Choice.EntryId).SequenceEqual(new[]{"alpha-test","beta"}),"Alpha Test and Beta are separate NN options");
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.AlphaTest);
            BetaButton("Beta table").onClick.Invoke();
            Check(panel.Tiles.Count(b=>b.gameObject.activeSelf)==1 && panel.Tiles.Any(b=>b.name=="beta" && b.gameObject.activeSelf),"Beta page shows one neutral model card");
            Check(LocalMenu.TableFamily(Setup.PlayerMetadata[1])==AiFamily.V4,"Beta remains a draft before Apply");
            panel.Close();yield return null;Check(LocalMenu.TableFamily(Setup.PlayerMetadata[1])==AiFamily.V4,"Cancel retains the previous family");
            PickBeta(1,0);yield return null;PickBeta(2,1);yield return null;PickBeta(3,3);yield return null;BetaRestored();
            Rows()[2].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;BetaButton("Alpha Test table").onClick.Invoke();panel.ApplyButton.onClick.Invoke();yield return null;
            Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).EntryId=="alpha-test"),"Alpha/Beta model switching is whole-table");
            PickBeta(2,1);yield return null;BetaRestored();
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            panel.SearchKButtons[0].onClick.Invoke();panel.TemperatureButtons[1].onClick.Invoke();panel.Close();yield return null;
            Check(LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==16 && LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0,"Cancel discards Beta K/temperature drafts");
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            BetaButton("Larger text").onClick.Invoke();BetaButton("Larger text").onClick.Invoke();yield return null;Capture("beta-1080-largest.png");panel.Close();yield return null;
            File.WriteAllText(BetaPreferenceFile,JsonConvert.SerializeObject(BetaPreferences()));
            Call(menu,"CreateNewGameConfiguration");yield return null;BetaRestored();
        }else BetaRestored();
        foreach(int k in new[]{8,16,32})foreach(double t in new[]{1.0,1.5}) {
            var package=new NativePackage(NativeModel.Beta.Location,k,t);
            Check(package.ModelSpec==NativeModel.Beta && (string)package.ModelManifest["model_id"]==NativeModel.Beta.Model,"six Beta K/temperature packages bind exact g1290 weights");
            var acceptance=(JObject)package.Policy["acceptance"].DeepClone();acceptance["utility_calibration"]["request_id"]="wrong";
            bool rejected=false;try{NativePackage.ValidateAcceptance(acceptance,package.ModelManifest);}catch(InvalidDataException){rejected=true;}
            Check(rejected,"changed utility receipt cannot be admitted");
        }
        var app=menu.StartGame(read?92941:92940);Check(app!=null,"actual menu creates Beta game");
        var state=NativeGameState.Read(app.Metadata,false);
        Check(NativeGameState.Model(state)==NativeModel.Beta && (string)state["runtime"]==NativeModel.Beta.Backend,"Beta saved runtime differs from Alpha");
        Check(state["assignments"].Where(r=>(string)r["kind"]=="native").All(r=>(string)r["model_id"]==NativeModel.Beta.Model && (string)r["profile"]==NativeModel.Beta.Profile("neutral")),"all saved NN seats use exact neutral g1290 model");
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(state,NativeGameState.Read(loaded.Metadata,false)),"Beta disk save/load preserves model identity");
        var bad=(JObject)state.DeepClone();bad["runtime"]=NativeModel.Alpha.Backend;SaveExtensions.Set(loaded.Metadata,bad);
        bool invalid=false;try{NativeGameState.Read(loaded.Metadata,false);}catch(InvalidDataException){invalid=true;}
        Check(invalid,"Beta-to-Alpha save tampering is rejected");
        var legacy=NativeGameState.Create(app.Metadata,p=>"easy");legacy["ready"]=true;SaveExtensions.Set(loaded.Metadata,legacy);
        Check(NativeGameState.Model(NativeGameState.Read(loaded.Metadata,false))==NativeModel.Alpha,"old Alpha identity remains readable");
        Check(errors.Count==0,"no Beta UI/save errors");
    }
}
