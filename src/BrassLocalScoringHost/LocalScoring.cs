using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Players;

namespace BrassHost
{
    // Separate training capability. Never called by Observe/Choices/Canonical.
    public sealed partial class Host
    {
        ScoringRecorder localRecorder;
        public object EnableLocalScoring(bool fresh=false)
        {
            if(localRecorder==null)localRecorder=new ScoringRecorder(this);
            if(fresh)localRecorder.Begin(Revision);
            return new{contract="local-scoring-targets/1.0.0",start_revision=localRecorder.Start};
        }
        public object LocalScoringTargets()
        {
            if(localRecorder==null || !Terminal)throw new InvalidOperationException("complete_local_scoring_required");
            return localRecorder.Result();
        }
        // These flags live outside the canonical game hash. Replaying from the
        // start reproduces captures, then resets at the saved branch boundary.
        JObject LocalScoringCheckpoint() {return localRecorder==null?null:new JObject{["start"]=localRecorder.Start};}
        sealed class ScoringRecorder
        {
            readonly Host source;
            readonly List<JObject> eras=new List<JObject>();
            JObject pending;
            int[] initial,previous,initialBonus;
            public int Start {get;private set;}
            public ScoringRecorder(Host host)
            {
                source=host;Begin(host.Revision);
                Wrap<ScoreLinksPhaseStarted>(e=>Capture(),false);
                Wrap<CanalEraEnded>(e=>Finish(e.EndOfEraIndustriesVPs,e.EndOfEraLinksVPs,e.CurrentEraBonusesVPs),true);
                Wrap<RailEraEnded>(e=>Finish(e.EndOfEraIndustriesVPs,e.EndOfEraLinksVPs,e.CurrentEraBonusesVPs),true);
            }
            void Wrap<T>(Action<T> capture,bool before) where T:class,IEvent
            {
                var routes=(IDictionary)typeof(AggregateRoot).GetField("_applyEventMethods",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(source.Game);
                var registration=(IEventRegistration)routes[typeof(T)];
                var original=registration.GetType().GetMethod("ApplyEvent");
                source.Game.UnregisterApplyEventRegistration(registration);
                source.Game.RegisterApplyEvent<T>(e=>{if(before)capture(e);original.Invoke(registration,new object[]{e});if(!before)capture(e);});
            }
            int[] Scores() {return source.Seats.Select(p=>p.VP).ToArray();}
            public void Begin(int revision) {Start=revision;eras.Clear();pending=null;initial=Scores();previous=(int[])initial.Clone();initialBonus=source.Seats.Select(p=>p.VPFromBonusesInCurrentEra).ToArray();}
            static int Checked(int value) {if(value<0 || value>65535)throw new InvalidOperationException("local_scoring_integer_bounds");return value;}
            void Capture()
            {
                if(pending!=null)throw new InvalidOperationException("duplicate_local_scoring_start");
                string era=source.Game.GameProgressInformation.CurrentGameEra.ToString();
                var slots=new JArray(source.Slots.Select(s=>{
                    var t=s.Value.BuildedIndustry;bool flipped=t!=null && t.IsFlipped;
                    return new JObject{["id"]=s.Key,["owner"]=flipped?(int)t.Color+1:0,
                        ["vp"]=flipped?Checked(t.VPForBuilding):0,["valid"]=true,
                        ["instance"]=t==null?null:source.tileIds[t.UniqueId]};}));
                var links=new JArray(source.Links.Select(l=>{
                    bool valid=l.GetType().Name==(era=="CanalEra"?"Canal":"Rail");
                    if(l.HasOwner && !valid)throw new InvalidOperationException("invalid_era_owned_link");
                    int points=0;
                    if(l.HasOwner)foreach(var region in new[]{l.Region1,l.Region2,l.Region3})
                        if(region!=null)points+=region is BorderRegion?2:region.BuildingSlots.Where(s=>s.BuildedIndustry!=null && s.BuildedIndustry.IsFlipped).Sum(s=>s.BuildedIndustry.VPPerLink);
                    return new JObject{["id"]=l.LinkId,["owner"]=l.HasOwner?(int)l.Owner.Color+1:0,
                        ["vp"]=Checked(points),["valid"]=valid};}));
                pending=new JObject{["era"]=era,["slots"]=slots,["links"]=links,["before"]=new JArray(Scores())};
            }
            static int[] Totals(JArray rows) {return Enumerable.Range(1,4).Select(s=>rows.Where(r=>(int)r["owner"]==s).Sum(r=>(int)r["vp"])).ToArray();}
            void Finish(Dictionary<PlayerColor,int> industry,Dictionary<PlayerColor,int> links,Dictionary<PlayerColor,int> bonuses)
            {
                if(pending==null)throw new InvalidOperationException("missing_local_scoring_start");
                int[] i=Totals((JArray)pending["slots"]),l=Totals((JArray)pending["links"]),after=Scores(),before=pending["before"].Select(x=>(int)x).ToArray();
                int[] b=Enumerable.Range(0,4).Select(s=>bonuses.ContainsKey((PlayerColor)s)?bonuses[(PlayerColor)s]:0).ToArray();
                for(int s=0;s<4;s++)if(i[s]!=industry[(PlayerColor)s] || l[s]!=links[(PlayerColor)s] || after[s]-before[s]!=i[s]+l[s])
                    throw new InvalidOperationException("official_local_scoring_audit_failed");
                pending["audit"]=new JObject{["industry"]=new JArray(i),["network"]=new JArray(l),["bonus"]=new JArray(b),
                    ["bonus_since_recording"]=new JArray(b.Zip(initialBonus,(a,z)=>a-z)),
                    ["outside_adjustment"]=new JArray(Enumerable.Range(0,4).Select(s=>before[s]-previous[s]-b[s]+initialBonus[s])),
                    ["outside_scoring"]=new JArray(before.Zip(previous,(a,z)=>a-z)),["after"]=new JArray(after)};
                eras.Add(pending);pending=null;previous=after;initialBonus=new int[4];
            }
            public object Result()
            {
                if(pending!=null || !eras.Any() || (string)eras.Last()["era"]!="RailEra")throw new InvalidOperationException("incomplete_local_scoring");
                return new{contract="local-scoring-targets/1.0.0",rules=Wire.Rules,producer="local-scoring-host/1.0.0",
                    start_revision=Start,initial_vp=initial,eras=eras.ToArray(),final_vp=Scores()};
            }
        }
    }
}
