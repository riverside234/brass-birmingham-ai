using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.Filesystem;

// Compile and run against the requested archived release, not the new serializer.
public static class V2SaveFixture
{
    public static int Main(string[] args)
    {
        if(ProfileCatalog.Runtime!="brass-local-ai/1.3.0" && ProfileCatalog.Runtime!="brass-local-ai/1.4.0" && ProfileCatalog.Runtime!="brass-local-ai/1.4.1")throw new Exception("wrong fixture producer");
        var repository=new BaseRepository(new FilesystemGameStorage(args[0],new Frameworks.Persistence.Serialization.JsonSerializer()),new FilesystemSnapshotStorage(args[0],new Frameworks.Persistence.Serialization.JsonSerializer()),new BrassBirminghamFactory());
        var games=new JArray();
        foreach(string key in ProfileCatalog.Keys)
        {
            var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=1301};var metadata=new GameMetadata(config,GameType.Local);
            foreach(var start in config.Players)
            {
                var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="fixture-"+start.Color,NickName="V2 compatibility fixture",AiType=AIType.Hard};
                p.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(p);metadata.JoinGame(p);
            }
            var state=LocalGameState.Create(metadata,p=>key);SaveExtensions.Set(metadata,state);
            var app=repository.Create(metadata);app.Game.StartGame();state["game_id"]=app.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(app.Metadata,state);repository.Save(app);
            int color=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            var events=repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision).ToList();
            LocalGameState.CommitIntent(app.Metadata,new IntentState{profile=(string)LocalGameState.Assignment(state,color)["profile"]["id"],seat=color,revision=app.Game.Revision},app.Game.Revision,events);
            repository.Save(app);games.Add(new JObject{["id"]=app.Metadata.GameId,["strategy"]=key});
        }
        File.WriteAllText(args[1],Wire.Json(new{passed=true,runtime=ProfileCatalog.Runtime,games}));return 0;
    }
}
