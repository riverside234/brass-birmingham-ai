using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassPatch;
using BrassPatch.Persistence;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine.UI;

public sealed partial class LocalAiUnityDriver
{
    Dictionary<string,string> IlPreferences()
    { return (Dictionary<string,string>)typeof(LocalMenu).GetField("TestPreferences",ReflectionAccess.Flags).GetValue(null); }
    string IlPreferenceFile { get { return Path.Combine(Path.GetDirectoryName(Output),"il-preferences.json"); } }
    Button IlButton(string name) { return AiSelectionPanel.Active.GetComponentsInChildren<Button>().Single(b=>b.name==name); }
    void PickIl(int seat,int difficulty)
    {
        Rows()[seat].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        panel.SwitchFamily(AiFamily.AlphaTest);IlButton("IL table").onClick.Invoke();
        panel.DifficultyButtons[difficulty].onClick.Invoke();Check(panel.ApplyButton.interactable,"IL is selectable");panel.ApplyButton.onClick.Invoke();
    }
    void IlRestored()
    {
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).EntryId=="il"),"IL applies to the whole NN table");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"IL seats retain independent 4/8/12s settings");
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableNeuralChoice(p)=="il") && LocalMenu.CanStart(menu),"restored IL table is playable");
    }
    IEnumerator IlUiCases()
    {
        bool read=Environment.GetEnvironmentVariable("BRASS_IL_UI_TEST")=="read";IlPreferences().Clear();
        if(read)foreach(var item in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(IlPreferenceFile)))IlPreferences().Add(item.Key,item.Value);
        Call(menu,"CreateNewGameConfiguration");yield return null;
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);yield return null;
            Check(AiSelectionCatalog.Entries(AiFamily.AlphaTest).Select(e=>e.Choice.EntryId).SequenceEqual(new[]{"alpha-test","beta","il"}),"all three NN models have distinct UI entries");
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.AlphaTest);
            IlButton("IL table").onClick.Invoke();
            Check(panel.Tiles.Count(b=>b.gameObject.activeSelf)==1 && panel.Tiles.Any(b=>b.name=="il" && b.gameObject.activeSelf),"IL page shows its own model card");
            panel.Close();yield return null;Check(LocalMenu.TableFamily(Setup.PlayerMetadata[1])==AiFamily.V4,"Cancel discards IL family draft");
            PickIl(1,0);yield return null;PickIl(2,1);yield return null;PickIl(3,3);yield return null;IlRestored();
            foreach(string key in new[]{"beta","alpha-test"}) {
                Rows()[2].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;panel.SwitchNeuralModel(key);panel.ApplyButton.onClick.Invoke();yield return null;
                Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).EntryId==key),"NN model change is whole-table: "+key);
                PickIl(2,1);yield return null;IlRestored();
            }
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            panel.SearchKButtons[0].onClick.Invoke();panel.TemperatureButtons[1].onClick.Invoke();panel.Close();yield return null;
            Check(LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==16 && LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0,"Cancel discards IL search settings");
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            IlButton("Larger text").onClick.Invoke();IlButton("Larger text").onClick.Invoke();yield return null;
            var tabCorners=new UnityEngine.Vector3[4];var footerCorners=new UnityEngine.Vector3[4];
            ((UnityEngine.RectTransform)IlButton("Page Gamma").transform).GetWorldCorners(tabCorners);
            panel.GetComponentsInChildren<TMPro.TextMeshProUGUI>().Single(t=>t.name=="Current AI").rectTransform.GetWorldCorners(footerCorners);
            Check(tabCorners.Min(p=>p.y)>footerCorners.Max(p=>p.y),"all NN subtabs stay above the footer at largest text size");
            Capture("il-1080-largest.png");panel.Close();yield return null;
            File.WriteAllText(IlPreferenceFile,JsonConvert.SerializeObject(IlPreferences()));
            Call(menu,"CreateNewGameConfiguration");yield return null;IlRestored();
        }else IlRestored();
        foreach(int k in new[]{8,16,32})foreach(double t in new[]{1.0,1.5}) {
            var package=new NativePackage(NativeModel.IL.Location,k,t);
            Check(package.ModelSpec==NativeModel.IL && package.ModelManifest["utility_calibration"].Type==JTokenType.Null,"IL package is exact and makes no utility calibration claim");
            var acceptance=(JObject)package.Policy["acceptance"].DeepClone();acceptance["export_sha256"]=NativeModel.Beta.Export;
            bool rejected=false;try{NativePackage.ValidateAcceptance(acceptance,package.ModelManifest);}catch(InvalidDataException){rejected=true;}
            Check(rejected,"IL cannot admit Beta's export");
        }
        var app=menu.StartGame(read?92951:92950);Check(app!=null,"actual menu starts an IL game");
        var state=NativeGameState.Read(app.Metadata,false);
        Check(NativeGameState.Model(state)==NativeModel.IL && (string)state["runtime"]==NativeModel.IL.Backend,"IL has its own saved identity");
        Check(state["assignments"].Where(r=>(string)r["kind"]=="native").All(r=>(string)r["model_id"]==NativeModel.IL.Model),"all AI seats use IL");
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(state,NativeGameState.Read(loaded.Metadata,false)),"IL disk save/load preserves exact identity");
        var bad=(JObject)state.DeepClone();bad["runtime"]=NativeModel.Beta.Backend;SaveExtensions.Set(loaded.Metadata,bad);
        bool invalid=false;try{NativeGameState.Read(loaded.Metadata,false);}catch(InvalidDataException){invalid=true;}
        Check(invalid,"IL-to-Beta saved model tampering is rejected");
        foreach(var model in new[]{NativeModel.Alpha,NativeModel.Beta}) {
            var old=NativeGameState.Create(app.Metadata,p=>"easy",16,1.0,model);old["ready"]=true;SaveExtensions.Set(loaded.Metadata,old);
            Check(NativeGameState.Model(NativeGameState.Read(loaded.Metadata,false))==model,"prior saved identity stays readable: "+model.Label);
        }
        Check(errors.Count==0,"no IL UI/save errors");
    }
}
