using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Reflection;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Players;
using BrassApplication.Game;
using BrassApplication.Persistence.Serialization;
using Frameworks.Persistence.RepositoryBackends.Filesystem;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.GameFactory;
using BrassApplication.Persistence.Repository;

public static class PersistenceTests
{
    static int checks;
    static void Check(bool condition, string label) { checks++; if (!condition) throw new Exception(label); }
    static void Reject(Action action, string label)
    {
        bool rejected=false;try{action();}catch(Exception){rejected=true;}Check(rejected,label);
    }
    public sealed class FaultSerializer : ISerialize
    {
        readonly ISerialize inner=new Frameworks.Persistence.Serialization.JsonSerializer();
        public string Fail;
        public bool BlockMetadataRead;
        public readonly ManualResetEventSlim ReadEntered=new ManualResetEventSlim(false),ContinueRead=new ManualResetEventSlim(false);
        public void Serialize<T>(Stream output,T graph)
        {
            bool events=graph is IList<IEvent>;
            if(Fail=="events"&&events || Fail=="metadata"&&graph is GameMetadata)
            { output.WriteByte(0x7B); throw new IOException("Injected serializer/write failure."); }
            inner.Serialize(output,graph);
        }
        public T Deserialize<T>(Stream input)
        {
            T value=inner.Deserialize<T>(input);
            if(BlockMetadataRead && typeof(T)==typeof(GameMetadata))
            {BlockMetadataRead=false;ReadEntered.Set();if(!ContinueRead.Wait(5000))throw new Exception("Timed out releasing metadata read fixture.");}
            return value;
        }
    }
    static GameMetadata Metadata(string id)
    {
        var cfg=new GameStartConfiguration(NumberOfPlayers.Two);cfg.Seed=20260927;
        var metadata=new GameMetadata(cfg,GameType.Local);metadata.GameId=id;return metadata;
    }
    static IEvent Event(int revision,int amount=30)
    {
        var result=new MoneyLoaned(PlayerColor.Blue,amount);result.Metadata.Revision=revision;return result;
    }
    static string PathFor(string root,string id,string name) { return Path.Combine(Path.Combine(root,id),name); }
    static IRepository OfflineRepository(string root)
    {
        return (IRepository)typeof(RepositoryFactory).GetMethod("CreateOfflineGameRepository",BindingFlags.Instance|BindingFlags.NonPublic)
            .Invoke(new RepositoryFactory(),new object[]{root});
    }
    public static object Stock(string root)
    {
        Directory.CreateDirectory(root);string run=Guid.NewGuid().ToString("N");
        var serializer=new FaultSerializer();var storage=new FilesystemGameStorage(root,serializer);
        string a="stock-truncate-"+run;storage.SaveGame(a,new List<IEvent>{Event(1)},Metadata(a));
        byte[] old=File.ReadAllBytes(PathFor(root,a,"events.storage"));serializer.Fail="events";
        try{storage.SaveGame(a,new List<IEvent>{Event(2)},Metadata(a));}catch(IOException){}
        long damagedLength=new FileInfo(PathFor(root,a,"events.storage")).Length;
        serializer.Fail=null;string b="stock-duplicate-"+run;var next=Event(2);
        storage.SaveGame(b,new List<IEvent>{Event(1)},Metadata(b));serializer.Fail="metadata";
        try{storage.SaveGame(b,new List<IEvent>{next},Metadata(b));}catch(IOException){}
        serializer.Fail=null;storage.SaveGame(b,new List<IEvent>{next},Metadata(b));
        var duplicate=storage.LoadEvents(b,0,int.MaxValue).Select(e=>e.Metadata.Revision).ToArray();
        string c="stock-delete-"+run;storage.SaveGame(c,new List<IEvent>{Event(1)},Metadata(c));
        File.WriteAllBytes(PathFor(root,c,"metadata.storage"),new byte[0]);storage.LoadGameMetadata(c);
        var production=OfflineRepository(Path.Combine(root,"stock-factory-"+run));
        var persistent=production.Advanced.GameStorage.GetType().BaseType.GetField("PersistentGameStorage",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(production.Advanced.GameStorage);
        typeof(FilesystemGameStorage).GetProperty("Serializer",BindingFlags.Instance|BindingFlags.NonPublic).SetValue(persistent,serializer,null);
        string d="factory";serializer.Fail=null;production.Advanced.GameStorage.SaveGame(d,new List<IEvent>{Event(1)},Metadata(d));
        serializer.Fail="events";try{production.Advanced.GameStorage.SaveGame(d,new List<IEvent>{Event(2)},Metadata(d));}catch(IOException){}
        int failedCacheCount=production.Advanced.GameStorage.LoadEvents(d,0,int.MaxValue).Count;
        return new{assembly=typeof(FilesystemGameStorage).Assembly.Location,previousEventBytes=old.Length,failedSaveEventBytes=damagedLength,retryRevisions=duplicate,emptyMetadataDeletedHistory=!Directory.Exists(Path.Combine(root,c)),offlineFactoryStorage=production.Advanced.GameStorage.GetType().FullName,cacheEventsAfterFailedSave=failedCacheCount};
    }
    public static object Patched(string root)
    {
        checks=0;Directory.CreateDirectory(root);string run=Guid.NewGuid().ToString("N");
        var serializer=new FaultSerializer();var storage=new FilesystemGameStorage(root,serializer);
        foreach(string failure in new[]{"events","metadata"})
        {
            string id="serialize-"+failure+"-"+run;var md=Metadata(id);storage.SaveGame(id,new List<IEvent>{Event(1)},md);
            string ep=PathFor(root,id,"events.storage"),mp=PathFor(root,id,"metadata.storage");
            byte[] oldEvents=File.ReadAllBytes(ep),oldMeta=File.ReadAllBytes(mp);serializer.Fail=failure;
            Reject(()=>storage.SaveGame(id,new List<IEvent>{Event(2)},md),"Injected serialization failed");
            Check(oldEvents.SequenceEqual(File.ReadAllBytes(ep))&&oldMeta.SequenceEqual(File.ReadAllBytes(mp)),"Serialization failure preserves both previous files");
            Check(!File.Exists(PathFor(root,id,"brasspatch-save-v1.pending")),"Serialization failure publishes no journal");
            serializer.Fail=null;storage.SaveGame(id,new List<IEvent>{Event(2)},md);storage.SaveGame(id,new List<IEvent>{Event(2)},md);
            Check(storage.LoadEvents(id,0,int.MaxValue).Count==2,"Repeated retry is idempotent");
        }
        foreach(string locked in new[]{"events.storage","metadata.storage"})
        {
            string id="commit-"+locked+"-"+run;var md=Metadata(id);storage.SaveGame(id,new List<IEvent>{Event(1)},md);
            md.GameProgressInformation.CurrentRoundNumber=99;
            // Blocks first or second live-file replacement after a complete durable journal exists.
            using(var blocker=new FileStream(PathFor(root,id,locked),FileMode.Open,FileAccess.Read,FileShare.Read))
                Reject(()=>storage.SaveGame(id,new List<IEvent>{Event(2)},md),"Locked live file fails commit");
            Check(File.Exists(PathFor(root,id,"brasspatch-save-v1.pending")),"Incomplete commit retains recovery journal");
            // A new storage instance simulates a restarted process and recovers on the first read.
            var restarted=new FilesystemGameStorage(root,new FaultSerializer());
            Check(restarted.LoadGameMetadata(id).GameId==id,"Recovery metadata remains readable");
            Check(restarted.LoadGameMetadata(id).GameProgressInformation.CurrentRoundNumber==99,"Recovery publishes new metadata with new events");
            Check(restarted.LoadEvents(id,0,int.MaxValue).Count==2,"Recovery completes both-file commit");
            Check(!File.Exists(PathFor(root,id,"brasspatch-save-v1.pending")),"Completed recovery removes journal");
            restarted.SaveGame(id,new List<IEvent>{Event(2)},md);
            Check(restarted.LoadEvents(id,0,int.MaxValue).Count==2,"Retry after recovery does not append duplicate");
            Check(File.Exists(PathFor(root,id,"events.storage.brasspatch-backup")),"Previous events retained as backup");
        }
        string conflict="conflict-"+run;var cm=Metadata(conflict);storage.SaveGame(conflict,new List<IEvent>{Event(1)},cm);
        byte[] committed=File.ReadAllBytes(PathFor(root,conflict,"events.storage"));
        Reject(()=>storage.SaveGame(conflict,new List<IEvent>{Event(1,99)},cm),"Conflicting revision rejected");
        Reject(()=>storage.SaveGame(conflict,new List<IEvent>{Event(3)},cm),"Revision gap rejected");
        Check(committed.SequenceEqual(File.ReadAllBytes(PathFor(root,conflict,"events.storage"))),"Rejected history edits preserve bytes");
        storage.SaveGame(conflict,new List<IEvent>{Event(2)},cm);
        Reject(()=>storage.SaveGame(conflict,new List<IEvent>{Event(1)},cm),"Stale batch cannot overwrite current metadata");
        // Existing incomplete saves must remain recoverable; do not recursively delete them.
        string empty="empty-metadata-"+run;storage.SaveGame(empty,new List<IEvent>{Event(1)},Metadata(empty));
        File.WriteAllBytes(PathFor(root,empty,"metadata.storage"),new byte[0]);
        Reject(()=>storage.LoadGameMetadata(empty),"Empty legacy metadata reports corruption");
        Check(File.Exists(PathFor(root,empty,"events.storage")),"Empty metadata does not delete history");
        string corrupt="corrupt-journal-"+run;var corruptMd=Metadata(corrupt);storage.SaveGame(corrupt,new List<IEvent>{Event(1)},corruptMd);
        using(var blocker=new FileStream(PathFor(root,corrupt,"events.storage"),FileMode.Open,FileAccess.Read,FileShare.Read))
            Reject(()=>storage.SaveGame(corrupt,new List<IEvent>{Event(2)},corruptMd),"Create interrupted transaction");
        string jp=PathFor(root,corrupt,"brasspatch-save-v1.pending");byte[] bytes=File.ReadAllBytes(jp);bytes[bytes.Length-1]^=0x01;File.WriteAllBytes(jp,bytes);
        committed=File.ReadAllBytes(PathFor(root,corrupt,"events.storage"));
        Reject(()=>storage.LoadGameMetadata(corrupt),"Corrupt journal is rejected");
        Check(committed.SequenceEqual(File.ReadAllBytes(PathFor(root,corrupt,"events.storage"))),"Corrupt journal does not change prior history");
        Reject(()=>storage.SaveGame("../escape",new List<IEvent>(),cm),"Paths cannot escape storage root");
        // Real repository command/event lifecycle, not only handcrafted serialization payloads.
        var repository=new BaseRepository(storage,new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var template=Metadata("new");
        foreach(var start in template.GameStartConfiguration.Players)
            template.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="test-"+start.Color,NickName="Test"});
        var app=repository.Create(template);app.Game.StartGame();repository.Save(app);
        var player=app.Game.GameProgressInformation.CurrentPlayer;int hand=player.Hand.Count;
        app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0],player);app.Game.TurnFlow.ConfirmCurrentAction();
        serializer.Fail="metadata";Reject(()=>repository.Save(app),"Production repository failed save is observable");
        Check(app.Game.GetUncommittedEvents().Count>0,"Failed repository save retains events for retry");
        serializer.Fail=null;repository.Save(app);repository.Save(app);
        var loaded=repository.Load(app.Metadata.GameId);
        Check(loaded.Game.Revision==app.Game.Revision,"Repository roundtrip preserves exact revision");
        Check(loaded.Game.GameProgressInformation.CurrentPlayer.Hand.Count==hand-1,"Repository retry never repeats card discard");
        Check(loaded.Game.TurnFlow.GetRemainingActionsCount()==app.Game.TurnFlow.GetRemainingActionsCount(),"Repository retry preserves action count");
        foreach(int count in new[]{2,3,4})
        {
            string id="shared-local-id-"+count+"-"+run;
            var config=new GameStartConfiguration((NumberOfPlayers)count);
            var local=new GameMetadata(config,GameType.Local){GameId=id};
            foreach(var start in config.Players) local.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="Player",NickName="Local"});
            storage.SaveGame(id,new List<IEvent>{Event(1)},local);
            for(int seat=count-1;seat>=0;seat--)
            {
                var update=(PlayerMetadata)local.PlayerMetadata[seat].Clone(); update.LastSeenEventRevision=seat+1;
                storage.SavePlayerMetadata(id,update);
                var roundtrip=storage.LoadGameMetadata(id);
                for(int other=0;other<count;other++) Check(roundtrip.PlayerMetadata[other].LastSeenEventRevision==(other>=seat?other+1:0),"Shared offline ID updates only its own seat");
            }
            byte[] before=File.ReadAllBytes(PathFor(root,id,"metadata.storage"));
            Reject(()=>storage.SavePlayerMetadata(id,new PlayerMetadata{PlayerNetworkId="Player",LastSeenEventRevision=99}),"Ambiguous local seat is rejected");
            var wrongGame=(PlayerMetadata)local.PlayerMetadata[0].Clone();wrongGame.GameId="another-game";
            Reject(()=>storage.SavePlayerMetadata(id,wrongGame),"Cross-game player metadata rejected");
            Check(before.SequenceEqual(File.ReadAllBytes(PathFor(root,id,"metadata.storage"))),"Invalid seat update preserves save bytes");
        }
        // Deterministic race: pause asynchronous last-seen read while a full game save starts.
        string race="last-seen-race-"+run;var rm=Metadata(race);
        rm.PlayerMetadata.Add(new PlayerMetadata{PlayerNetworkId="reader",NickName="Reader"});
        storage.SaveGame(race,new List<IEvent>{Event(1)},rm);serializer.BlockMetadataRead=true;
        var reading=Task.Run(()=>storage.SavePlayerMetadata(race,new PlayerMetadata{PlayerNetworkId="reader",LastSeenEventRevision=1}));
        Check(serializer.ReadEntered.Wait(5000),"Last-seen read reached deterministic pause");
        rm.GameProgressInformation.CurrentRoundNumber=42;
        var saving=Task.Run(()=>storage.SaveGame(race,new List<IEvent>{Event(2)},rm));
        try{Check(!saving.Wait(100),"Concurrent full save cannot pass an in-progress player read/modify/write");}
        finally{serializer.ContinueRead.Set();}
        Task.WaitAll(reading,saving);
        storage.SavePlayerMetadata(race,new PlayerMetadata{PlayerNetworkId="reader",LastSeenEventRevision=2});
        var final=storage.LoadGameMetadata(race);
        Check(final.GameProgressInformation.CurrentRoundNumber==42 && final.PlayerMetadata[0].LastSeenEventRevision==2,"Player-only metadata update preserves newer game metadata");
        var production=OfflineRepository(Path.Combine(root,"factory-"+run));
        Check(production.Advanced.GameStorage is FilesystemGameStorage,"Production offline factory uses durable storage without cache-first publication");
        typeof(FilesystemGameStorage).GetProperty("Serializer",BindingFlags.Instance|BindingFlags.NonPublic).SetValue(production.Advanced.GameStorage,serializer,null);
        serializer.Fail=null;string fid="factory";var fm=Metadata(fid);production.Advanced.GameStorage.SaveGame(fid,new List<IEvent>{Event(1)},fm);
        serializer.Fail="events";Reject(()=>production.Advanced.GameStorage.SaveGame(fid,new List<IEvent>{Event(2)},fm),"Factory-backed save fails visibly");
        serializer.Fail=null;Check(production.Advanced.GameStorage.LoadEvents(fid,0,int.MaxValue).Count==1,"Factory read never exposes failed uncommitted append");
        production.Advanced.GameStorage.SaveGame(fid,new List<IEvent>{Event(2)},fm);production.Advanced.GameStorage.SaveGame(fid,new List<IEvent>{Event(2)},fm);
        Check(production.Advanced.GameStorage.LoadEvents(fid,0,int.MaxValue).Count==2,"Factory-backed retry is idempotent in memory and disk");
        return new{assertions=checks,failures=new string[0],runDirectory=root};
    }
}
