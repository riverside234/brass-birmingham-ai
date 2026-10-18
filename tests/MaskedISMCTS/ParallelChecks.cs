using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BrassLegacyAI;

public static class ParallelChecks
{
    static int checks;
    static void Check(bool ok,string message){Interlocked.Increment(ref checks);if(!ok)throw new Exception(message);}
    static Task<double[]> Job(HeuristicBatchQueue q,string key,Func<double[]> work)
    {return Task.Factory.StartNew(()=>q.Evaluate(key,work),CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default);}
    public static int Main()
    {
        var q=new HeuristicBatchQueue();var both=new CountdownEvent(2);
        Func<double[]> overlap=()=>{both.Signal();Check(both.Wait(3000),"different keys must compute concurrently");return new[]{.25};};
        var a=Job(q,"a",overlap);var b=Job(q,"b",overlap);Task.WaitAll(a,b);
        Check(q.Report()["max_concurrent"]==2 && q.Report()["active"]==0,"concurrency retires");
        var entered=new ManualResetEventSlim();var release=new ManualResetEventSlim();int calls=0;
        var first=Job(q,"same",()=>{Interlocked.Increment(ref calls);entered.Set();release.Wait();return new[]{.75};});
        Check(entered.Wait(3000),"first owner starts");
        var second=Job(q,"same",()=>{throw new Exception("duplicate computed");});
        try{Check(SpinWait.SpinUntil(()=>q.Report()["coalesced"]==1,3000),"same key waits");}
        finally{release.Set();}
        Task.WaitAll(first,second);first.Result[0]=0;
        Check(second.Result[0]==.75 && calls==1,"coalesced outputs detached");
        Check(q.Evaluate("same",()=>{throw new Exception();})[0]==.75,"cached output detached");
        // Exceptions wake every waiter, remove the pending key and allow retry.
        entered.Reset();release.Reset();
        var failure=Job(q,"failure",()=>{entered.Set();release.Wait();throw new Exception("injected");});
        Check(entered.Wait(3000),"failing owner starts");
        var follower=Job(q,"failure",()=>new[]{0.0});
        try{Check(SpinWait.SpinUntil(()=>q.Report()["coalesced"]==2,3000),"failing request coalesces");}
        finally{release.Set();}
        try{Task.WaitAll(failure,follower);}catch(AggregateException){}
        Check(failure.IsFaulted && follower.IsFaulted && q.Report()["pending"]==0,"failure retires both owners");
        Check(q.Evaluate("failure",()=>new[]{1.0})[0]==1.0,"failed key can retry");
        Check(q.Evaluate("outer",()=>q.Evaluate("inner",()=>new[]{.5}))[0]==.5,"different-key nesting");
        bool recursive=false;try{q.Evaluate("cycle",()=>q.Evaluate("cycle",()=>new[]{1.0}));}catch(InvalidOperationException){recursive=true;}
        Check(recursive && q.Report()["pending"]==0,"same-key recursion fails promptly");
        bool invalid=false;try{q.Evaluate("nan",()=>new[]{double.NaN});}catch(InvalidOperationException){invalid=true;}
        Check(invalid && q.Report()["pending"]==0,"nonfinite not cached");
        for(int i=0;i<520;i++)q.Evaluate("bounded-"+i,()=>new[]{1.0});
        Check(q.Report()["cache_entries"]==512,"cache bound");
        using(q.Bind()){Check(HeuristicBatchQueue.Current==q,"thread-local bind");using(new HeuristicBatchQueue().Bind()){}Check(HeuristicBatchQueue.Current==q,"nested binding restored");}
        Check(HeuristicBatchQueue.Current==null,"binding retired");
        var tail=new DeadlineAdmission();Check(tail.Admit(.001) && !tail.Admit(0),"cold admission/deadline");
        foreach(var n in new[]{10.0,20.0,30.0})tail.Observe(n);
        Check(tail.Admit(1),"need four complete timing samples");tail.Observe(1000.0);
        Check(tail.Estimate==10 && !tail.Admit(9) && tail.Admit(10),"optimistic boundary");
        tail.Observe(double.NaN);tail.Observe(0);Check(tail.Estimate==10,"bad timing samples ignored");
        for(int i=0;i<8;i++)tail.Observe(50.0);
        Check(tail.Estimate==50,"timing window bounded to latest eight");
        Console.WriteLine("PASS "+checks+" parallel/cache/deadline checks");return 0;
    }
}
