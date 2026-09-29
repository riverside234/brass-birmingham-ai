using System;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI {
 public sealed class ReplyTelemetry { public string status="not_enabled_in_local_release"; }
 public static class ReplySearch {
  public static Decision Decide(Planner planner,JObject view,List<Choice> legal,Profile profile,JObject composition,bool fixedWork,uint seed,CancellationToken cancel,Action<Plan> incumbent,IntentState intent) { throw new NotSupportedException("S2 is not enabled in the local game release."); }
 }
}
