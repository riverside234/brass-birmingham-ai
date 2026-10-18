using System;
using System.Collections.Generic;
using System.Linq;

namespace BrassLegacyAI
{
    // Launch control only: never shorten a trajectory's horizon or back up a
    // partial return. Learn from this root, not an unrelated previous position.
    public sealed class DeadlineAdmission
    {
        readonly Queue<double> completed=new Queue<double>();
        public void Observe(double milliseconds)
        {
            if(milliseconds<=0 || double.IsNaN(milliseconds) || double.IsInfinity(milliseconds))return;
            completed.Enqueue(milliseconds);while(completed.Count>8)completed.Dequeue();
        }
        // An optimistic lower quartile avoids reserving a slow outlier's time.
        // Four complete paths are required before withholding any new launch.
        public double Estimate {get{return completed.Count<4?0:completed.OrderBy(x=>x).ElementAt((completed.Count-1)/4);}}
        public bool Admit(double remainingMilliseconds){return remainingMilliseconds>0 && remainingMilliseconds>=Estimate;}
    }
}
