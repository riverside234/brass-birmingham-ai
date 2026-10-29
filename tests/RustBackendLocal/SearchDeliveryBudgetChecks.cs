using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using BrassLegacyAI.Native;
using Newtonsoft.Json.Linq;

public static class SearchDeliveryBudgetChecks
{
    static int checks;
    static void Check(bool okay,string message){checks++;if(!okay)throw new Exception(message);}
    public static int Main(string[] args)
    {
        Exception error=null;
        try {
            foreach(int search in new[]{4000,8000,10000,12000})foreach(int k in new[]{8,16,32}) {
                var request=new JObject{["allowance_ms"]=search};
                var j=new NativeJob(request,new Stopwatch(),search,k,1.0);
                Check(j.DeadlineMilliseconds==search+1000,"one additional delivery second for every setting");
                var stamp=typeof(NativeJob).GetProperty("ResponseReceivedMilliseconds");
                stamp.SetValue(j,(double)search,null);
                Check(j.DeliveryDeadlineMilliseconds==search+1000,"late result retains the fixed total deadline");
                stamp.SetValue(j,100.0,null);
                Check(j.DeliveryDeadlineMilliseconds==1100,"early result has at most one second of delivery");
                var partial=NativeJob.WithDeadline(request,new Stopwatch(),300,k,1.0);
                Check(partial.DeadlineMilliseconds==300,"restored partial allowance receives no new grace");
                bool refused=false;try{NativeJob.WithDeadline(request,new Stopwatch(),search+1001,k,1.0);}catch(ArgumentOutOfRangeException){refused=true;}
                Check(refused,"caller cannot grant a second delivery allowance");
            }
            var lease=new NativeTurnClock();var original=lease.Clock;
            Check(lease.BindDeadline(5000)==5000,"fresh deadline binds once");
            Check(lease.BindDeadline(5000)==5000 && ReferenceEquals(original,lease.Clock),"retry preserves the same deadline and clock");
            Check(lease.BindDeadline(300)==300,"restored budget narrows the lease");
            Check(lease.BindDeadline(5000)==300 && ReferenceEquals(original,lease.Clock),"later retry cannot replenish a restored lease");
        }catch(Exception e){error=e;}
        File.WriteAllText(args[0],new JObject{["passed"]=error==null,["checks"]=checks,["error"]=error?.ToString()}.ToString());
        if(error!=null)Console.Error.WriteLine(error);return error==null?0:1;
    }
}
