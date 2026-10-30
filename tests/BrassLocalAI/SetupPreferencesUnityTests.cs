using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassLegacyAI;
using BrassPatch;
using BrassApplication.AI;
using BrassLegacyAI.Native;
using BrassPatch.Persistence;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.UI;

public sealed partial class LocalAiUnityDriver
{
    const string SetupKey="BrassLocalAI.Setup.v2";
    string PreferenceFixture { get { return Path.Combine(Path.GetDirectoryName(Output),"setup-preferences.json"); } }
    Dictionary<string,string> PreferenceValues()
    { return (Dictionary<string,string>)typeof(LocalMenu).GetField("TestPreferences",ReflectionAccess.Flags).GetValue(null); }
    void PrepareSetupPreferences()
    {
        var prefs=PreferenceValues();prefs.Clear();
        if(Environment.GetCommandLineArgs().Contains("--brass-setup-preferences=read"))
            foreach(var pair in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(PreferenceFixture)))prefs.Add(pair.Key,pair.Value);
    }
    Button PanelButton(string name)
    { return AiSelectionPanel.Active.GetComponentsInChildren<Button>().Single(b=>b.name==name); }
    void PickSetup(int seat,string style,int difficulty,bool neural=false,bool human=false)
    {
        Rows()[seat].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        if(neural)PanelButton("NN table").onClick.Invoke();else panel.SwitchFamily(AiFamily.V4);
        if(!neural)panel.Tiles.Single(b=>b.name==style).onClick.Invoke();
        panel.DifficultyButtons[difficulty].onClick.Invoke();
        if(human)panel.SelectPage(PlayerSelectionPage.Human);
        Check(panel.ApplyButton.interactable,"applied choice is available");panel.ApplyButton.onClick.Invoke();
        Check(!AiSelectionPanel.IsOpen,"Apply closes chooser");
    }
    void CheckRestoredNeural(int expectedK=8)
    {
        Check(Get<int>(menu,"_numberOfPlayers")==4 && !Setup.GameStartConfiguration.IsIntroductoryGame,"restored valid NN table shape");
        Check(Setup.PlayerMetadata.Select(p=>p.AiType==AIType.Human).SequenceEqual(new[]{true,false,true,false}),"restored controllers including second Human");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"restored independent allowances including Human's remembered AI setting");
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).Family==AiFamily.AlphaTest),"restored one coherent NN family");
        Check(LocalMenu.TableFamily(Setup.PlayerMetadata[0])==AiFamily.AlphaTest && LocalMenu.CanStart(menu),"NN family retained for Human seat and Start allowed");
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableSearchK(p)==expectedK),"saved K restored for the whole table independently of seat allowances");
    }
    IEnumerator SetupPreferencesCases()
    {
        bool read=Environment.GetCommandLineArgs().Contains("--brass-setup-preferences=read");
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);
            PickSetup(1,"cotton",0);PickSetup(2,"rail-tempo",1);PickSetup(3,"random",3);yield return null;
            Check(PreferenceValues().ContainsKey(SetupKey) && history.CurrentGame==null,"Apply saves defaults before any game is started");
            string saved=PreferenceValues()[SetupKey];
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();PanelButton("NN table").onClick.Invoke();PanelButton("Page Gamma").onClick.Invoke();
            Check(!AiSelectionPanel.Active.ApplyButton.interactable && AiSelectionPanel.Active.Tiles.All(b=>!b.interactable),"Gamma remains unplayable");
            AiSelectionPanel.Active.Close();yield return null;
            Check(PreferenceValues()[SetupKey]==saved,"Cancel and Gamma browsing never persist drafts");
            Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Strategy).SequenceEqual(new[]{"cotton","rail-tempo","random"}),"V4 styles and unresolved Random restore before Start");
            Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"V4 seat difficulties restored");
            PickSetup(1,"cotton",0,true);PickSetup(2,"rail-tempo",1,true);PickSetup(3,"random",3,true);PickSetup(2,"rail-tempo",1,true,true);yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var kPanel=AiSelectionPanel.Active;
            Check(kPanel.DraftSearchK==16 && kPanel.SearchKButtons.Length==2,"existing setup defaults to K16 and offers two choices");
            Check(kPanel.SearchKButtons.Select(b=>b.name).SequenceEqual(new[]{"NN search K8","NN search K16"}),"only K8 and K16 are offered");
            kPanel.SearchKButtons[0].onClick.Invoke();
            Check(kPanel.DraftSearchK==8 && LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==16,"K8 stays a draft until Apply");
            kPanel.Close();yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();kPanel=AiSelectionPanel.Active;
            Check(kPanel.DraftSearchK==16,"Cancel discards the search K draft");
            kPanel.SearchKButtons[0].onClick.Invoke();kPanel.ApplyButton.onClick.Invoke();yield return null;
            CheckRestoredNeural();
            foreach(var difficulty in Enumerable.Range(0,4)) {
                Rows()[1].GetComponent<AiSetupRow>().OpenPicker();kPanel=AiSelectionPanel.Active;
                kPanel.DifficultyButtons[difficulty].onClick.Invoke();
                Check(kPanel.DraftSearchK==8,"K8 is independent of time budget "+difficulty);
                kPanel.SearchKButtons[1].onClick.Invoke();
                Check(kPanel.DraftSearchK==16,"K16 is available at time budget "+difficulty);
                Check(!kPanel.GetComponentsInChildren<Button>().Any(b=>b.name=="NN search K32"),"K32 is absent at time budget "+difficulty);
                kPanel.Close();yield return null;
            }
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();kPanel=AiSelectionPanel.Active;
            kPanel.SearchKButtons[1].onClick.Invoke();kPanel.ApplyButton.onClick.Invoke();yield return null;
            Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableSearchK(p)==16),"K16 Apply changes every seat together");
            Rows()[3].GetComponent<AiSetupRow>().OpenPicker();kPanel=AiSelectionPanel.Active;
            kPanel.SearchKButtons[0].onClick.Invoke();kPanel.ApplyButton.onClick.Invoke();yield return null;
            CheckRestoredNeural();
            saved=PreferenceValues()[SetupKey];
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var tPanel=AiSelectionPanel.Active;
            Check(tPanel.DraftTemperature==1.0 && tPanel.TemperatureButtons.Length==2,"temperature defaults to 1.0 with two choices");
            tPanel.TemperatureButtons[1].onClick.Invoke();
            Check(tPanel.DraftTemperature==1.5 && LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0,"temperature remains a draft before Apply");
            tPanel.Close();yield return null;
            Check(PreferenceValues()[SetupKey]==saved,"temperature Cancel leaves preferences intact");
            foreach(int kIndex in Enumerable.Range(0,NativePackage.SearchKChoices.Count))foreach(int difficulty in Enumerable.Range(0,4)) {
                Rows()[1].GetComponent<AiSetupRow>().OpenPicker();tPanel=AiSelectionPanel.Active;
                Check(tPanel.DraftTemperature==1.0,"Cancel restored T1.0");
                tPanel.SearchKButtons[kIndex].onClick.Invoke();tPanel.DifficultyButtons[difficulty].onClick.Invoke();
                tPanel.TemperatureButtons[1].onClick.Invoke();
                Check(tPanel.DraftTemperature==1.5 && tPanel.DraftSearchK==NativePackage.SearchKChoices[kIndex] && tPanel.DraftDifficulty==DifficultyCatalog.Keys[difficulty],"T1.5 works independently of K and allowance");
                tPanel.TemperatureButtons[0].onClick.Invoke();Check(tPanel.DraftTemperature==1.0,"T1.0 works at every K/allowance");
                tPanel.Close();yield return null;
            }
            Rows()[3].GetComponent<AiSetupRow>().OpenPicker();tPanel=AiSelectionPanel.Active;
            tPanel.TemperatureButtons[1].onClick.Invoke();tPanel.ApplyButton.onClick.Invoke();yield return null;
            Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableTemperature(p)==1.5),"temperature Apply sets the whole NN table");
            saved=PreferenceValues()[SetupKey];
            foreach(double invalidT in new[]{0.0,2.0,Double.NaN,Double.PositiveInfinity})
                Check(!LocalMenu.ApplyPlayerSelection(Setup.PlayerMetadata[1],menu,new PlayerSelectionDraft(PlayerSelectionPage.AlphaTest,new AiChoice(AiFamily.AlphaTest,"alpha-test"),"easy",8,invalidT)),"unsupported temperature cannot bypass UI");
            foreach(double temperature in new[]{1.0,1.5})foreach(int k in new[]{8,16,32}) {
                var package=new NativePackage(NativePackage.Location,k,temperature);
                Check(package.Temperature==temperature && package.SearchK==k,"six native K/temperature policies admitted");
                var metadata=(BrassApplication.Game.GameMetadata)Setup.Clone();
                var state=NativeGameState.Create(metadata,LocalMenu.Difficulty,k,temperature);SaveExtensions.Set(metadata,state);
                Check(NativeGameState.Temperature(NativeGameState.Read(metadata,true))==temperature,"saved game pins temperature");
                var bad=(JObject)state.DeepClone();bad["nn_temperature"]=temperature==1.0?1.5:1.0;SaveExtensions.Set(metadata,bad);
                bool rejected=false;try{NativeGameState.Read(metadata,true);}catch(InvalidDataException){rejected=true;}
                Check(rejected,"assignment integrity rejects changed temperature");
                var legacy=(JObject)state.DeepClone();legacy.Remove("nn_temperature");
                // Frozen 1.4.40 assignment schema, before the temperature field.
                legacy["assignment_sha256"]=BrassHost.Wire.Hash(new JObject {
                    ["format"]=legacy["format"],["runtime"]=legacy["runtime"],["family"]=legacy["family"],
                    ["rules"]=legacy["rules"],["schema"]=legacy["schema"],["configuration"]=legacy["configuration"],
                    ["assignments"]=legacy["assignments"],["nn_search_k"]=legacy["nn_search_k"] });
                SaveExtensions.Set(metadata,legacy);
                Check(NativeGameState.Temperature(NativeGameState.Read(metadata,true))==1.0 && JToken.DeepEquals(SaveExtensions.Get(metadata),legacy),"legacy save keeps T1.0 without rewrite");
            }
            foreach(int invalidK in new[]{0,9,32,64})
                Check(!LocalMenu.ApplyPlayerSelection(Setup.PlayerMetadata[1],menu,new PlayerSelectionDraft(PlayerSelectionPage.AlphaTest,new AiChoice(AiFamily.AlphaTest,"alpha-test"),"easy",invalidK)),"unsupported K cannot bypass UI");
            foreach(int k in new[]{8,16,32}) {
                var package=new NativePackage(NativePackage.Location,k);
                Check(package.Policy["k_by_allowance_ms"].Children<JProperty>().All(p=>(int)p.Value==k),"integrity-checked native policy maps all four allowances to K"+k);
                var metadata=(BrassApplication.Game.GameMetadata)Setup.Clone();
                var state=NativeGameState.Create(metadata,LocalMenu.Difficulty,k);SaveExtensions.Set(metadata,state);
                Check(NativeGameState.SearchK(NativeGameState.Read(metadata,true))==k,"saved-game compatibility preserves K"+k+" in its assignment");
                var bad=(JObject)state.DeepClone();bad["nn_search_k"]=k==8?16:8;SaveExtensions.Set(metadata,bad);
                bool rejected=false;try{NativeGameState.Read(metadata,true);}catch(InvalidDataException){rejected=true;}
                Check(rejected,"saved K is protected by assignment integrity");
                var legacy=(JObject)state.DeepClone();legacy.Remove("nn_search_k");
                legacy["assignment_sha256"]=(string)typeof(NativeGameState).GetMethod("AssignmentHash",ReflectionAccess.Flags).Invoke(null,new object[]{legacy});
                SaveExtensions.Set(metadata,legacy);
                Check(NativeGameState.SearchK(NativeGameState.Read(metadata,true))==16 && JToken.DeepEquals(SaveExtensions.Get(metadata),legacy),"pre-selector save reads as K16 without rewriting it");
            }
            Check(!LocalMenu.ApplyPlayerSelection(Setup.PlayerMetadata[1],new object(),new PlayerSelectionDraft(PlayerSelectionPage.V4,new AiChoice(AiFamily.V4,"cotton"),"hard")),"stale setup cannot apply");
            Check(!LocalMenu.ApplyPlayerSelection(Setup.PlayerMetadata[1],menu,new PlayerSelectionDraft(PlayerSelectionPage.Human,new AiChoice(AiFamily.Gamma,"gamma-01"),"hard")),"unavailable family cannot bypass via Human");
            Check(PreferenceValues()[SetupKey]==saved,"rejected drafts leave defaults intact");
            menu.DecreaseNumberOfPlayers();Rows()[1].GetComponent<AiSetupRow>().OpenPicker();PanelButton("NN table").onClick.Invoke();
            Check(!AiSelectionPanel.Active.GetComponentsInChildren<Button>().Any(b=>b.name=="Alpha Test table"),"unsupported NN runtime option omitted");
            Check(PanelButton("Page Gamma").transform.parent.name=="NN options","Gamma still under NN on unsupported table");
            AiSelectionPanel.Active.Close();yield return null;
            Call(menu,"CreateNewGameConfiguration");yield return null;CheckRestoredNeural();
            // Invalid and old preferences are menu defaults, never save-data errors.
            foreach(var bad in new[]{"{", "{\"version\":999}",saved.Replace("\"easy\"","\"invalid\"")}) {
                PreferenceValues()[SetupKey]=bad;Call(menu,"CreateNewGameConfiguration");yield return null;
                Check(LocalMenu.TableFamily(Setup.PlayerMetadata[0])==AiFamily.V4,"invalid preferences fall back safely");
            }
            PreferenceValues().Remove(SetupKey);PreferenceValues()["BrassLocalAI.v1.Strategy.1"]="supplier";PreferenceValues()["BrassLocalAI.v1.Difficulty.1"]="mid";
            Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(LocalMenu.Strategy(Setup.PlayerMetadata[1])=="supplier" && LocalMenu.Difficulty(Setup.PlayerMetadata[1])=="mid","older style/difficulty preferences migrate");
            PreferenceValues()[SetupKey]=saved;Call(menu,"CreateNewGameConfiguration");yield return null;CheckRestoredNeural();
            var oldPreferences=JObject.Parse(saved);oldPreferences.Remove("nn_search_k");
            PreferenceValues()[SetupKey]=oldPreferences.ToString();Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==16,"pre-selector menu defaults migrate to K16");
            var k32Preferences=JObject.Parse(saved);k32Preferences["nn_search_k"]=32;
            PreferenceValues()[SetupKey]=k32Preferences.ToString();Call(menu,"CreateNewGameConfiguration");yield return null;
            CheckRestoredNeural(16);
            Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableTemperature(p)==1.5),"K32 menu migration preserves the remembered temperature");
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var migratedPanel=AiSelectionPanel.Active;
            Check(migratedPanel.DraftSearchK==16 && migratedPanel.SearchKButtons.Length==2 && migratedPanel.SearchKButtons[1].GetComponent<AiControlSurface>().Selected,"K32 menu preference opens with K16 selected");
            migratedPanel.ApplyButton.onClick.Invoke();yield return null;
            Check((int)JObject.Parse(PreferenceValues()[SetupKey])["nn_search_k"]==16,"Apply saves the migrated menu K16");
            foreach(var invalid in new JToken[]{new JValue(7),new JValue("8"),JValue.CreateNull()}) {
                var broken=JObject.Parse(saved);broken["nn_search_k"]=invalid;
                PreferenceValues()[SetupKey]=broken.ToString();Call(menu,"CreateNewGameConfiguration");yield return null;
                Check(LocalMenu.TableFamily(Setup.PlayerMetadata[0])==AiFamily.V4 && LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==16,"invalid K preference falls back safely");
            }
            PreferenceValues()[SetupKey]=saved;Call(menu,"CreateNewGameConfiguration");yield return null;CheckRestoredNeural();
            var oldTemperature=JObject.Parse(saved);oldTemperature.Remove("nn_temperature");
            PreferenceValues()[SetupKey]=oldTemperature.ToString();Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0 && LocalMenu.TableSearchK(Setup.PlayerMetadata[1])==8,"old preferences default temperature while preserving K");
            foreach(var invalid in new JToken[]{new JValue(2.0),new JValue("1.5"),JValue.CreateNull()}) {
                var broken=JObject.Parse(saved);broken["nn_temperature"]=invalid;PreferenceValues()[SetupKey]=broken.ToString();
                Call(menu,"CreateNewGameConfiguration");yield return null;
                Check(LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0 && LocalMenu.TableFamily(Setup.PlayerMetadata[1])==AiFamily.V4,"invalid temperature preference falls back safely");
            }
            PreferenceValues()[SetupKey]=saved;Call(menu,"CreateNewGameConfiguration");yield return null;CheckRestoredNeural();
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var finalPanel=AiSelectionPanel.Active;
            finalPanel.SearchKButtons[1].onClick.Invoke();finalPanel.ApplyButton.onClick.Invoke();yield return null;
            CheckRestoredNeural(16);
        } else CheckRestoredNeural(16);
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.TableTemperature(p)==1.5),"temperature 1.5 survives menu recreation and process restart");

        foreach(var size in new[]{new Vector2Int(1920,1080),new Vector2Int(1920,1200)}) {
            router.Resize(size.x,size.y);yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
            Check(panel.CurrentPage==PlayerSelectionPage.AlphaTest,"opening restored NN seat selects Alpha Test");
            Check(panel.DraftTemperature==1.5 && panel.TemperatureButtons[1].GetComponent<AiControlSurface>().Selected && !panel.TemperatureButtons[0].GetComponent<AiControlSurface>().Selected,"temperature has exactly one selected state");
            Check(panel.DraftSearchK==16 && panel.SearchKButtons.Length==2 && panel.SearchKButtons[1].GetComponent<AiControlSurface>().Selected && panel.SearchKButtons.Count(b=>b.GetComponent<AiControlSurface>().Selected)==1,"K16 has one clear selected state");
            Check(!panel.GetComponentsInChildren<Button>().Any(b=>b.name=="NN search K32"),"K32 is absent from the restored panel");
            foreach(var button in panel.SearchKButtons) {
                AssertInsideTarget(button.GetComponent<RectTransform>(),"visible search K choice");
                Check(button.navigation.selectOnDown!=null,"search K supports keyboard navigation");
            }
            Check(panel.Window.Find("Alpha Test table")==null && panel.Window.Find("Page Gamma")==null,"neural options are not whole-table peers");
            Check(PanelButton("Alpha Test table").transform.parent.name=="NN options" && PanelButton("Page Gamma").transform.parent.name=="NN options","Alpha Test and Gamma nested under NN");
            foreach(var name in new[]{"NN table","V4 table","Alpha Test table","Page Gamma"})AssertInsideTarget(PanelButton(name).GetComponent<RectTransform>(),"visible "+name);
            panel.ChangeTextSize(1);yield return null;Capture("setup-nn-"+size.y+"-largest.png");
            foreach(var button in panel.SearchKButtons)AssertInsideTarget(button.GetComponent<RectTransform>(),"large-text search K choice inside panel");
            foreach(var button in panel.TemperatureButtons){
                AssertInsideTarget(button.GetComponent<RectTransform>(),"large-text temperature choice inside panel");
                Check(button.navigation.selectOnDown!=null,"temperature keyboard navigation");
            }
            PanelButton("Page Gamma").onClick.Invoke();yield return null;
            Check(panel.CurrentPage==PlayerSelectionPage.Gamma && !panel.ApplyButton.interactable,"Gamma preview disables Apply");
            Capture("setup-gamma-"+size.y+"-largest.png");
            PanelButton("V4 table").onClick.Invoke();
            Check(panel.Window.Find("NN options")==null && panel.Tiles.Count()==7,"V4 contains only V4 styles");
            Check(panel.SearchKButtons.Length==0,"NN search controls are scoped to the NN page");
            Check(panel.TemperatureButtons.Length==0,"temperature controls are scoped to NN");
            panel.Close();yield return null;CheckRestoredNeural(16);
        }
        // Publish exact preference bytes for a separate fresh Unity process.
        if(!read)File.WriteAllText(PreferenceFixture,JsonConvert.SerializeObject(PreferenceValues()));
        else {
            PickSetup(1,"cotton",0);yield return null;
            Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Strategy).SequenceEqual(new[]{"cotton","rail-tempo","random"}),"V4 styles survive NN roundtrip and restart");
            Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(LocalMenu.TableFamily(Setup.PlayerMetadata[0])==AiFamily.V4,"changed family persists on next menu creation");
        }
        Check(history.CurrentGame==null,"UI preference smoke never starts gameplay or touches live saves");
        Check(errors.Count==0,"no UI exceptions: "+String.Join("\n",errors));selectionCasesPassed=true;
    }
}
