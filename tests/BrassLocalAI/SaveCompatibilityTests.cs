using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.Filesystem;

public static class SaveCompatibilityTests
{
    static int checks;
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    public static int Main(string[] args)
    {
        try
        {
            var manifest=JObject.Parse(File.ReadAllText(args[1]));
            var repo=new BaseRepository(new FilesystemGameStorage(args[0],new Frameworks.Persistence.Serialization.JsonSerializer()),new FilesystemSnapshotStorage(args[0],new Frameworks.Persistence.Serialization.JsonSerializer()),new BrassBirminghamFactory());
            foreach(var entry in manifest["games"])
            {
                var app=repo.Load((string)entry["id"]);var state=LocalGameState.Read(app.Metadata);string before=Wire.Json(state);
                Check((string)state["runtime"]==(string)manifest["runtime"],"load genuine old-runtime fixture");
                int seat=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;var assignment=LocalGameState.Assignment(state,seat);
                Check((string)state["format"] == LocalGameState.PreviousFormat && LocalGameState.Difficulty(state, assignment) == DifficultyCatalog.Existing && DifficultyCatalog.Milliseconds(DifficultyCatalog.Existing) == 10000, "genuine older save keeps current 10-second compatibility allowance");
                var cfg=ProfileCatalog.ForPlay((JObject)assignment["profile"]);
                Check(cfg.cash_policy==1 && cfg.conversion_policy==1 && cfg.id==(string)assignment["profile"]["id"],"apply v3 cash repair with saved intent identity");
                Check(before==Wire.Json(LocalGameState.Read(app.Metadata)),"reading and preparing old saves does not rewrite assignments");
                var events=LocalGameState.History(app,repo.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision));
                var live=Host.Capture(app,events);var composition=Host.Composition((JObject)state["configuration"]);cfg.work_limit=1;
                var prior=LocalGameState.Intent(state,seat,app.Game.Revision,events);Check(prior!=null,"old intent remains readable");
                var plan=new Planner(composition).Decide((JObject)Wire.Token(live.Observe(seat)),live.Choices(),cfg,true,1,default(System.Threading.CancellationToken),null,prior).selected;
                foreach(var action in plan.actions){live.ValidateLiveAction(action,composition);live.LiveCommit(action,composition);}
                events=LocalGameState.History(app,repo.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision));
                LocalGameState.CommitIntent(app.Metadata,plan.intent,app.Game.Revision,events);repo.Save(app);var loaded=repo.Load(app.Metadata.GameId);var next=LocalGameState.Read(loaded.Metadata);
                Check(JToken.DeepEquals(state["assignments"],next["assignments"]),"v2 assignment and source hashes survive v3 play/save/reload");
                Check(LocalGameState.Intent(next,seat,loaded.Game.Revision,events)!=null && loaded.Game.PlayersTrack.Players.Count==4,"four-player state and updated intent restore");
                var corrupt=(JObject)next.DeepClone();corrupt["assignments"][0]["profile"]["conversion_policy"]=42;SaveExtensions.Set(loaded.Metadata,corrupt);
                bool rejected=false;try{LocalGameState.Read(loaded.Metadata);}catch(InvalidDataException){rejected=true;}
                Check(rejected,"corrupt old assignment still rejected");
                var newProfile=ProfileCatalog.Create((string)entry["strategy"],4);
                Check(newProfile.id.StartsWith(ProfileCatalog.Runtime+"/") && newProfile.cash_policy==1 && newProfile.conversion_policy==1,"new game has explicit v3 identity");
            }
            File.WriteAllText(args[2],Wire.Json(new{passed=true,checks,games=manifest["games"],scope="Generated pre-update v2/v3 saves only; no user save opened or modified"}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[2],Wire.Json(new{passed=false,checks,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
