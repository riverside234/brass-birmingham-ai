using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using BrassApplication.Game;
using BrassApplication.AI;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public static class NativeP56SaveAudit
{
    public static int Main(string[] args) {
        if(args.Contains("--rpc-fixture")) {
            string line;
            while((line=Console.ReadLine())!=null) {
                var request=JObject.Parse(line);
                var initial=new JObject{["package_id"]="fixture-package",["game_id"]=request["body"]["game_id"],
                    ["seats"]=new JArray(((JArray)request["body"]["assignments"]).Select(s=>new JObject{["selection"]=s.DeepClone()}))};
                Console.WriteLine(new JObject{["protocol"]=PersonalityPackage.Protocol,["id"]=request["id"],["ok"]=true,["data"]=initial}.ToString(Formatting.None));
            }
            return 0;
        }
        try {
            var host=new Host(327);var metadata=new GameMetadata(host.Game.GameStartConfiguration,GameType.Local){GameId="save-audit"};
            foreach(var player in host.Game.GameStartConfiguration.Players)
                metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=player,AiType=AIType.Hard,PlayerNetworkId="audit-"+player.Color});
            var flags=BindingFlags.NonPublic|BindingFlags.Static;
            typeof(PersonalityPackage).GetField("loaded",flags).SetValue(null,true);
            typeof(PersonalityPackage).GetField("manifest",flags).SetValue(null,new JObject{["id"]="fixture-package"});
            var info=new ProcessStartInfo(Assembly.GetExecutingAssembly().Location,"--rpc-fixture") {
                UseShellExecute=false,CreateNoWindow=true,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true};
            var process=Process.Start(info);typeof(PersonalityRpc).GetField("process",flags).SetValue(null,process);
            try {
                var legacy=LocalGameState.Create(metadata,p=>"balanced");legacy["ready"]=true;
                PersonalityState.Attach(metadata,legacy,p=>new AiChoice(AiFamily.Gamma,"fixture-style"),p=>"easy");
                var before=SaveExtensions.Get(metadata);
                var bytes=SaveExtensions.Merge(metadata,metadata.GameId,Path.GetDirectoryName(args[0]),System.Text.Encoding.UTF8.GetBytes("{}"));
                SaveExtensions.Read(metadata,bytes);
                var restored=PersonalityState.Unwrap(SaveExtensions.Get(metadata));
                if(!JToken.DeepEquals(legacy,restored))throw new Exception("Legacy payload changed");
                if(!JToken.DeepEquals(before,SaveExtensions.Get(metadata)))throw new Exception("Native metadata merge changed sealed state");
                var events=host.Game.GetUncommittedEvents().ToList();JArray pending;
                var initial=PersonalityState.Snapshot(metadata,host.Game.Revision,events,out pending);
                var plan=new JArray(new JArray("pass|fixture","finish|||||"));
                PersonalityState.Commit(metadata,host.Game.Revision,events,initial,plan);
                var committed=SaveExtensions.Get(metadata);
                bytes=SaveExtensions.Merge(metadata,metadata.GameId,Path.GetDirectoryName(args[0]),System.Text.Encoding.UTF8.GetBytes("{}"));
                var clone=(GameMetadata)metadata.Clone();SaveExtensions.Read(clone,bytes);
                restored=PersonalityState.Snapshot(clone,host.Game.Revision,events,out pending);
                if(!JToken.DeepEquals(restored,initial) || !JToken.DeepEquals(plan,pending))throw new Exception("Pending action changed after metadata save/clone/load");
                if(!JToken.DeepEquals(committed,SaveExtensions.Get(clone)))throw new Exception("Committed envelope changed");
                restored=PersonalityState.Snapshot(clone,0,events,out pending);
                if(!JToken.DeepEquals(restored,initial) || pending!=null)throw new Exception("Rewind reused a future plan");
                bool rejected=false;
                try{SaveExtensions.Merge(metadata,"foreign-game",Path.GetDirectoryName(args[0]),System.Text.Encoding.UTF8.GetBytes("{}"));}
                catch(InvalidDataException){rejected=true;}
                if(!rejected || !JToken.DeepEquals(committed,SaveExtensions.Get(metadata)))throw new Exception("Foreign game save changed envelope");
                File.WriteAllText(args[0],"{\"passed\":true,\"checks\":[\"actual_attach_merge_read_roundtrip\",\"legacy_state_unchanged\",\"sealed_state_unchanged\",\"pending_action_clone_load\",\"rewind_no_future_plan\",\"foreign_game_rejected_without_mutation\"]}");
                Console.WriteLine("Native P6 save roundtrip passed");return 0;
            }finally{PersonalityRpc.Shutdown();}
        }catch(Exception error){Console.Error.WriteLine(error);return 1;}
    }
}
