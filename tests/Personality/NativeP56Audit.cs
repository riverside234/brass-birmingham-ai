using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassApplication.Game;
using BrassApplication.AI;
using Newtonsoft.Json.Linq;

public static class NativeP56Audit
{
    static BrassApplicationGame App(Host host) {
        var metadata=new GameMetadata(host.Game.GameStartConfiguration,GameType.Local){GameId="p56-history-audit"};
        foreach(var player in host.Game.GameStartConfiguration.Players)
            metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=player,AiType=AIType.Hard,PlayerNetworkId="audit-"+player.Color});
        return new BrassApplicationGame(host.Game,metadata);
    }
    public static int Main(string[] args) {
        try {
            var results=new List<object>();
            foreach(int seed in new[]{327,638,1987}) {
                var host=new Host(seed);var app=App(host);var random=new Random(seed);
                var kinds=new Dictionary<string,int>();int boundaries=0,steps=0;
                while(!host.Terminal && steps++<1800) {
                    var legal=host.Choices().ToArray();
                    var choices=host.Phase=="action" ? legal.Where(c=>c.kind!="pass" && c.kind!="scout").ToArray() : legal;
                    if(choices.Length==0)choices=legal;
                    var choice=choices[random.Next(choices.Length)];
                    int count;kinds.TryGetValue(choice.kind,out count);kinds[choice.kind]=count+1;
                    host.Apply(host.Revision,choice.id);
                    if(choice.kind!="finish" || host.Terminal)continue;
                    int[] debts;var captured=Host.CapturePersonality(app,null,out debts);
                    var expected=(JObject)Wire.Token(host.Observe(host.Actor));
                    var actual=(JObject)Wire.Token(captured.Observe(captured.Actor));
                    if(!JToken.DeepEquals(expected["history"],actual["history"])) {
                        File.WriteAllText(args[0],Wire.Json(new{passed=false,seed,steps,kind=choice.kind,expected=expected["history"],actual=actual["history"]}));
                        throw new Exception("Public history mismatch: seed="+seed+" step="+steps);
                    }
                    boundaries++;
                }
                if(!host.Terminal)throw new Exception("Game did not complete");
                results.Add(new{seed,steps,boundaries,kinds});
            }
            File.WriteAllText(args[0],Wire.Json(new{passed=true,results}));
            Console.WriteLine("Native P5/P6 history audit passed");return 0;
        }catch(Exception error){Console.Error.WriteLine(error);return 1;}
    }
}
