using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class EraChecks
{
    static int checks;
    static void Check(bool ok,string label){checks++;if(!ok)throw new Exception(label);}
    public static int Main(string[] args)
    {
        try
        {
            var cards=JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/contracts/v0.2.0/card-composition.json")));
            var host=new Host(842100,null);
            while(host.Game.GameProgressInformation.CurrentRoundNumber==1){host.SearchApply(host.Choices().First(c=>c.kind=="pass").id);host.SearchApply(host.Choices().Single(c=>c.kind=="finish").id);host.Advance();}
            var view=(JObject)Wire.Token(host.Observe(host.Actor));
            var rail=(JObject)view.DeepClone();rail["context"]["era"]="RailEra";
            var network=host.Choices().First(c=>c.kind=="network");
            Check(EraEconomy.NetworkCost(view,network)==3,"canal fee");
            rail["coal_market"]=0;Check(EraEconomy.CoalPrice(rail,1)>0,"empty market quote");
            var single=EraEconomy.NetworkCost(rail,network);
            rail["pending"]["kind"]="network";rail["pending"]["parts"]=1;
            Check(EraEconomy.NetworkCost(rail,network)==single+5,"second rail base costs ten");
            Check(EraEconomy.Strategy(view)!=EraEconomy.Strategy(rail),"separate era priorities");
            var unflipped=(JObject)rail["slots"][0].DeepClone();unflipped["tile"]=Conversion.Player(rail,(int)rail["seat"])["mat"][0]["tiles"][0].DeepClone();unflipped["tile"]["flipped"]=false;
            Check(EraEconomy.LinkProbability(rail,unflipped,(int)rail["seat"],.42)==.42,"one shared uncertain flip event");
            var noTime=(JObject)rail.DeepClone();noTime["context"]["round"]=8;noTime["context"]["actions_remaining"]=1;
            Check(InvestmentOpportunity.Development(noTime,"CottonMill")<0,"development cannot unlock a last-action sale");
            Check(InvestmentOpportunity.FreeDevelopment(noTime,"CottonMill")<0,"free development still needs a later investment action");
            Check(Conversion.MerchantBonus(noTime,new JObject{["id"]="Gloucester:0",["bonus"]="Develop"})==0,"unusable free bonus is zero, not a phantom reward");
            Check(Math.Abs(EraEconomy.Finance(view,0,20,40,10)-40*.32*(1-Math.Exp(-.5)))<1e-9,"canal finance unchanged");
            var queue=new HeuristicBatchQueue();var start=new ManualResetEventSlim();int computes=0;
            var jobs=Enumerable.Range(0,4).Select(i=>Task.Run(()=>{start.Wait();using(queue.Bind())return queue.Evaluate("same-permitted-input",()=>{Interlocked.Increment(ref computes);Thread.Sleep(15);return new[]{1.0,2.0,3.0,4.0};});})).ToArray();
            start.Set();Task.WaitAll(jobs);
            Check(computes==1,"dynamic queue coalesces duplicate work");
            jobs[0].Result[0]=99;using(queue.Bind())Check(queue.Evaluate("same-permitted-input",()=>new[]{0.0})[0]==1,"queue output ownership");
            Check((int)queue.Report()["pending"]==0,"queue drained");
            using(queue.Bind())try{queue.Evaluate("error",()=>{throw new Exception("expected");});}catch(Exception e){Check(e.GetBaseException().Message=="expected","queue error delivered");}
            using(queue.Bind())Check(queue.Evaluate("after-error",()=>new[]{.25})[0]==.25,"queue recovers after error");
            var profile=ProfileCatalog.Create("balanced",4);profile.milliseconds=350;profile.opponent_turn_limit=0;
            var before=Wire.Hash(host.Canonical());var d=RootForest.Decide(cards,view,host.Choices(),profile,4);
            Check(d.search.workers==4 && d.failures==0,"four private owners finish");
            Check(d.search.opponent_nodes==0 && (int)d.search.cpu_batching["pending"]==0,"only complete observer evidence; queue drained");
            Check(d.selected.actions.Count==2,"whole turn incumbent");
            Check(before==Wire.Hash(host.Canonical()),"live state untouched");
            Check(d.search.root.Values.All(e=>e.visits>0 && e.returns.Length==4),"raw completed forest evidence");
            Check(TurnCoordinator.SearchWorkers==2,"game allowance is two, independent of four lab owners");
            var game=RootForest.Decide(cards,view,host.Choices(),profile,TurnCoordinator.SearchWorkers);
            Check(game.search.workers==2 && game.failures==0 && game.search.cpu_batching["pending"]==0,"two game owners retire with complete legal incumbent");
            Check(game.selected.actions.Count==2 && before==Wire.Hash(host.Canonical()),"two-worker game preserves state and full turn");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,workers=4,game_workers=TurnCoordinator.SearchWorkers,cpu_batching=d.search.cpu_batching}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));return 1;}
    }
}
