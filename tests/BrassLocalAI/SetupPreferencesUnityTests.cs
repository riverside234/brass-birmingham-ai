using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassLegacyAI;
using BrassPatch;
using BrassApplication.AI;
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
    void CheckRestoredNeural()
    {
        Check(Get<int>(menu,"_numberOfPlayers")==4 && !Setup.GameStartConfiguration.IsIntroductoryGame,"restored valid NN table shape");
        Check(Setup.PlayerMetadata.Select(p=>p.AiType==AIType.Human).SequenceEqual(new[]{true,false,true,false}),"restored controllers including second Human");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"restored independent allowances including Human's remembered AI setting");
        Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).Family==AiFamily.AlphaTest),"restored one coherent NN family");
        Check(LocalMenu.TableFamily(Setup.PlayerMetadata[0])==AiFamily.AlphaTest && LocalMenu.CanStart(menu),"NN family retained for Human seat and Start allowed");
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
            CheckRestoredNeural();
            saved=PreferenceValues()[SetupKey];
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
        } else CheckRestoredNeural();

        foreach(var size in new[]{new Vector2Int(1920,1080),new Vector2Int(1920,1200)}) {
            router.Resize(size.x,size.y);yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
            Check(panel.CurrentPage==PlayerSelectionPage.AlphaTest,"opening restored NN seat selects Alpha Test");
            Check(panel.Window.Find("Alpha Test table")==null && panel.Window.Find("Page Gamma")==null,"neural options are not whole-table peers");
            Check(PanelButton("Alpha Test table").transform.parent.name=="NN options" && PanelButton("Page Gamma").transform.parent.name=="NN options","Alpha Test and Gamma nested under NN");
            foreach(var name in new[]{"NN table","V4 table","Alpha Test table","Page Gamma"})AssertInsideTarget(PanelButton(name).GetComponent<RectTransform>(),"visible "+name);
            panel.ChangeTextSize(1);yield return null;Capture("setup-nn-"+size.y+"-largest.png");
            PanelButton("Page Gamma").onClick.Invoke();yield return null;
            Check(panel.CurrentPage==PlayerSelectionPage.Gamma && !panel.ApplyButton.interactable,"Gamma preview disables Apply");
            Capture("setup-gamma-"+size.y+"-largest.png");
            PanelButton("V4 table").onClick.Invoke();
            Check(panel.Window.Find("NN options")==null && panel.Tiles.Count()==7,"V4 contains only V4 styles");
            panel.Close();yield return null;CheckRestoredNeural();
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
