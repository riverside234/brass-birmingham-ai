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
    Dictionary<string,string> IndustryPreferences()
    { return (Dictionary<string,string>)typeof(LocalMenu).GetField("TestPreferences",ReflectionAccess.Flags).GetValue(null); }
    string IndustryPreferenceFile { get { return Path.Combine(Path.GetDirectoryName(Output),"industry-preferences.json"); } }
    Button IndustryButton(string name)
    { return AiSelectionPanel.Active.GetComponentsInChildren<Button>().Single(b=>b.name==name); }
    void PickIndustry(int seat,string focus,int difficulty)
    {
        Rows()[seat].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        panel.SwitchFamily(AiFamily.AlphaTest);
        panel.Tiles.Single(b=>b.name==NativeFocus.ChoiceId(focus) && b.gameObject.activeSelf).onClick.Invoke();
        panel=AiSelectionPanel.Active;panel.DifficultyButtons[difficulty].onClick.Invoke();
        Check(panel.ApplyButton.interactable,"industry choice can be applied");panel.ApplyButton.onClick.Invoke();
    }
    void IndustryRestored()
    {
        Check(Get<int>(menu,"_numberOfPlayers")==4 && !Setup.GameStartConfiguration.IsIntroductoryGame,"restored standard four-player table");
        Check(Setup.PlayerMetadata.Select(p=>LocalMenu.Choice(p).EntryId).SequenceEqual(new[]{"alpha-test","alpha-test-cotton","alpha-test-iron","alpha-test-coal"}),"independent seat industry tags restored");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"mixed 4/8/12s budgets restored with industry tags");
        Check(LocalMenu.CanStart(menu),"mixed focus NN table starts");
    }
    IEnumerator IndustryFocusCases()
    {
        bool read=Environment.GetEnvironmentVariable("BRASS_INDUSTRY_FOCUS_TEST")=="read";
        IndustryPreferences().Clear();
        if(read)foreach(var item in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(IndustryPreferenceFile)))IndustryPreferences().Add(item.Key,item.Value);
        Call(menu,"CreateNewGameConfiguration");yield return null;
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);yield return null;
            var entries=AiSelectionCatalog.Entries(AiFamily.AlphaTest);
            Check(entries.Count==7 && entries.All(e=>e.Available),"neutral plus six playable industry tags");
            foreach(string focus in NativeFocus.Keys) {
                string before=LocalMenu.Choice(Setup.PlayerMetadata[1]).EntryId;
                Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.AlphaTest);
                Check(panel.Tiles.Count(b=>b.gameObject.activeSelf)==7,"NN displays exactly seven tag tiles");
                panel.Tiles.Single(b=>b.name==NativeFocus.ChoiceId(focus) && b.gameObject.activeSelf).onClick.Invoke();
                Check(LocalMenu.Choice(Setup.PlayerMetadata[1]).EntryId==before,"tag selection remains a draft");
                AiSelectionPanel.Active.Close();yield return null;
                Check(LocalMenu.Choice(Setup.PlayerMetadata[1]).EntryId==before,"Cancel keeps prior tag");
                PickIndustry(1,focus,0);yield return null;
                Check(LocalMenu.Choice(Setup.PlayerMetadata[1]).EntryId==NativeFocus.ChoiceId(focus),"Apply selects "+focus);
                Check(LocalMenu.Choice(Setup.PlayerMetadata[2]).EntryId=="alpha-test","other NN seat stays neutral");
                var style=NativeFocus.Style(focus);var intent=NativeFocus.Intent(focus);int index=NativeFocus.Keys.ToList().IndexOf(focus);
                Check(style.Count==4 && intent.Count==8 && style.Values<double>().Sum()==(index==0?0:1) && intent.Values<double>().Sum()==(index==0?0:1),"bounded exact context "+focus);
                if(index>0)Check((double)intent[index-1]==1 && (double)style[index<=3?0:2]==1,"training context channel order "+focus);
            }
            PickIndustry(1,"cotton",0);yield return null;PickIndustry(2,"iron",1);yield return null;PickIndustry(3,"coal",3);yield return null;
            IndustryRestored();
            Rows()[2].GetComponent<AiSetupRow>().OpenPicker();AiSelectionPanel.Active.SwitchFamily(AiFamily.V4);AiSelectionPanel.Active.ApplyButton.onClick.Invoke();yield return null;
            Check(Setup.PlayerMetadata.Where(s=>s.AiType!=AIType.Human).All(s=>LocalMenu.Choice(s).Family==AiFamily.V4),"V4 and NN cannot mix in the same table");
            Rows()[2].GetComponent<AiSetupRow>().OpenPicker();AiSelectionPanel.Active.SwitchFamily(AiFamily.AlphaTest);AiSelectionPanel.Active.ApplyButton.onClick.Invoke();yield return null;
            IndustryRestored();
            string encoded=IndustryPreferences()["BrassLocalAI.Setup.v2"];
            var decode=typeof(LocalMenu).Assembly.GetType("BrassLegacyAI.LocalSetupPreferences").GetMethod("Decode");
            var previous=JObject.Parse(encoded);foreach(JObject seat in previous["seats"])seat.Remove("nn_focus");
            Check(decode.Invoke(null,new object[]{previous.ToString()})!=null,"old preferences without focus remain readable");
            previous["seats"][1]["nn_focus"]="unknown";
            Check(decode.Invoke(null,new object[]{previous.ToString()})==null,"unknown saved preference tag is rejected");
            Rows()[2].GetComponent<AiSetupRow>().OpenPicker();var p=AiSelectionPanel.Active;
            p.Filter.text="Pottery";Check(p.Tiles.Count(b=>b.gameObject.activeSelf)==1 && p.Tiles.Last().name=="alpha-test-pottery","industry filter finds a tag");
            p.Close();yield return null;Check(IndustryPreferences()["BrassLocalAI.Setup.v2"]==encoded,"filter/Cancel never changes saved preferences");
            // Keep layout captures short; scrolling exposes the third row at large text sizes.
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();p=AiSelectionPanel.Active;
            IndustryButton("Larger text").onClick.Invoke();IndustryButton("Larger text").onClick.Invoke();yield return null;
            Check(p.GridScroll.content.rect.height>=p.GridScroll.viewport.rect.height,"industry roster scrolls at largest text size");
            Capture("industry-tags-largest.png");p.GridScroll.verticalNormalizedPosition=0;yield return null;Capture("industry-tags-bottom.png");p.Close();yield return null;
            File.WriteAllText(IndustryPreferenceFile,JsonConvert.SerializeObject(IndustryPreferences()));
            Call(menu,"CreateNewGameConfiguration");yield return null;IndustryRestored();
        }else IndustryRestored();

        var app=menu.StartGame(read?92841:92840);Check(app!=null,"actual menu creates tagged NN table");
        var state=NativeGameState.Read(app.Metadata,false);
        foreach(var player in app.Metadata.PlayerMetadata.Where(p=>p.AiType!=AIType.Human)) {
            int color=(int)player.PlayerStartConfiguration.Color;string focus=NativeFocus.FromChoice(LocalMenu.Choice(player).EntryId);
            Check(NativeGameState.Focus(state,color)==focus,"new game pins chosen industry tag");
            Check((string)state["assignments"].Single(r=>(int)r["color"]==color)["profile"]==NativeFocus.Profile(focus),"save binds focus profile");
        }
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(NativeGameState.Read(loaded.Metadata,false),state),"disk save/load retains every focus and integrity hash");
        var bad=(JObject)state.DeepClone();bad["assignments"].First(r=>(string)r["kind"]=="native")["nn_focus"]="pottery";
        SaveExtensions.Set(loaded.Metadata,bad);bool rejected=false;try{NativeGameState.Read(loaded.Metadata,false);}catch(InvalidDataException){rejected=true;}
        Check(rejected && JToken.DeepEquals(SaveExtensions.Get(loaded.Metadata),bad),"tampered focus rejected without rewriting save");
        // An old signed neutral save has no focus field; opening must not migrate its bytes.
        var old=NativeGameState.Create(app.Metadata,p=>"easy");old["ready"]=true;
        foreach(var row in old["assignments"].Where(r=>(string)r["kind"]=="native"))((JObject)row).Remove("nn_focus");
        old["assignment_sha256"]=(string)typeof(NativeGameState).GetMethod("AssignmentHash",ReflectionAccess.Flags).Invoke(null,new object[]{old});
        SaveExtensions.Set(loaded.Metadata,old);
        Check(NativeGameState.Read(loaded.Metadata,false)["assignments"].Where(r=>(string)r["kind"]=="native").All(r=>NativeGameState.Focus(old,(int)r["color"])=="neutral"),"old neutral saves keep neutral context");
        Check(JToken.DeepEquals(SaveExtensions.Get(loaded.Metadata),old),"old neutral save is read without rewriting");
        Check(!AiSelectionCatalog.CanChoose(new AiChoice(AiFamily.AlphaTest,"alpha-test-unknown")),"unknown industry tag cannot bypass UI");
        Check(errors.Count==0,"no Unity errors during industry UI and persistence smoke");
    }
}
