using System;
using System.IO;
using System.Linq;
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

public static class EraSaveProducer
{
    public static int Main(string[] args)
    {
        var rows=new JArray();
        var repository=new BaseRepository(new FilesystemGameStorage(args[1],new Frameworks.Persistence.Serialization.JsonSerializer()),new FilesystemSnapshotStorage(args[1],new Frameworks.Persistence.Serialization.JsonSerializer()),new BrassBirminghamFactory());
        foreach(string key in ProfileCatalog.Keys)foreach(string difficulty in DifficultyCatalog.Keys)
        {
            var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=842100};var metadata=new GameMetadata(config,GameType.Local);
            foreach(var start in config.Players)
            {
                var player=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="fixture-"+start.Color,AiType=AIType.Hard,NickName="Archive save fixture"};
                player.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(player);metadata.JoinGame(player);
            }
            var state=LocalGameState.Create(metadata,p=>key,p=>difficulty);
            SaveExtensions.Set(metadata,state);var app=repository.Create(metadata);app.Game.StartGame();state["game_id"]=app.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(app.Metadata,state);repository.Save(app);
            rows.Add(new JObject{["id"]=app.Metadata.GameId,["strategy"]=key,["difficulty"]=difficulty});
        }
        File.WriteAllText(args[0],Wire.Json(new{runtime=ProfileCatalog.Runtime,rows}));return 0;
    }
}
public static class EraSaveConsumer
{
    public static int Main(string[] args)
    {
        int checks=0;
        try
        {
            var input=JObject.Parse(File.ReadAllText(args[0]));
            var repository=new BaseRepository(new FilesystemGameStorage(args[2],new Frameworks.Persistence.Serialization.JsonSerializer()),new FilesystemSnapshotStorage(args[2],new Frameworks.Persistence.Serialization.JsonSerializer()),new BrassBirminghamFactory());
            foreach(var row in input["rows"])
            {
                var app=repository.Load((string)row["id"]);var metadata=app.Metadata;var before=Wire.Json(SaveExtensions.Get(metadata));var restored=LocalGameState.Read(metadata);
                if((string)restored["runtime"]!=(string)input["runtime"])throw new Exception("archive runtime changed");checks++;
                foreach(var assignment in restored["assignments"])
                {
                    string difficulty=(string)row["difficulty"];if(LocalGameState.Difficulty(restored,(JObject)assignment)!=difficulty)throw new Exception("clock changed");checks++;
                    var saved=(JObject)assignment["profile"];var playable=ProfileCatalog.ForPlay(saved);
                    if(playable.id!=(string)saved["id"] || playable.mode!="masked_ismcts")throw new Exception("saved identity changed");checks++;
                    if(ProfileCatalog.SourceHash((string)row["strategy"],(string)input["runtime"])!=(string)assignment["source_sha256"])throw new Exception("source hash changed");checks++;
                }
                if(before!=Wire.Json(SaveExtensions.Get(metadata)))throw new Exception("reading rewrote save");checks++;
            }
            File.WriteAllText(args[1],Wire.Json(new{passed=true,checks,runtime=(string)input["runtime"],scope="genuine archived repository saves, all six styles and three clocks; no user save touched"}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[1],Wire.Json(new{passed=false,checks,error=e.ToString()}));return 1;}
    }
}
