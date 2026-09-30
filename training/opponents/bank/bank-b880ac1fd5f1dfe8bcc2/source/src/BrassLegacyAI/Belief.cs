using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public static class Belief
    {
        // Same conservation contract as R1, with a portable xorshift ordering.
        // This accepts NO trusted game/checkpoint/seed. The seed is policy-owned.
        public static JObject Setup(JObject view,JObject composition,uint seed)
        {
            if((string)view["protocol"]!=Wire.Version || (string)view["rules"]!=Wire.Rules || view["players"].Count()!=4 || (bool)view["context"]["terminal"])throw new Exception("observation_contract");
            int own=(int)view["seat"];var pool=new List<string>();
            foreach(var p in composition.Properties())for(int i=0;i<(int)p.Value;i++)pool.Add(p.Name);
            var hands=Enumerable.Range(0,4).Select(_=>new List<string>()).ToArray();
            var discards=Enumerable.Range(0,4).Select(_=>new List<string>()).ToArray();
            foreach(var p in view["players"])
            {
                int s=(int)p["seat"];hands[s].AddRange((s==own?view["own_hand"]:p["known_wild"]).Select(x=>(string)x));
                discards[s].AddRange(p["discards"].Select(x=>(string)x));
                foreach(string c in hands[s].Concat(discards[s]))if(!c.StartsWith("wild:") && !pool.Remove(c))throw new Exception("card_conservation");
            }
            uint state=seed==0?0x9e3779b9:seed;
            for(int i=pool.Count-1;i>0;i--){state^=state<<13;state^=state>>17;state^=state<<5;int j=(int)(state%(uint)(i+1));var v=pool[i];pool[i]=pool[j];pool[j]=v;}
            Func<int,List<string>> take=n=>{if(n<0 || n>pool.Count)throw new Exception("card_count");var a=pool.Take(n).ToList();pool.RemoveRange(0,n);return a;};
            foreach(var p in view["players"]){int s=(int)p["seat"];if(s!=own)hands[s].AddRange(take((int)p["hand_count"]-hands[s].Count));if(hands[s].Count!=(int)p["hand_count"])throw new Exception("hand_count");}
            foreach(var p in view["players"]){int s=(int)p["seat"];discards[s].InsertRange(0,take((int)p["hidden_discards"]));}
            if(pool.Count!=(int)view["deck_count"])throw new Exception("deck_count");
            foreach(var kind in new[]{"industry","location"})if((int)view["wild_"+kind+"_count"]+hands.Sum(h=>h.Count(x=>x=="wild:"+kind))!=4)throw new Exception("wild_count");
            return JObject.FromObject(new{cards=new{hands,discards,discard_starts=new int[4],deck=pool,wild_industry=Enumerable.Repeat("wild:industry",(int)view["wild_industry_count"]).ToArray(),wild_location=Enumerable.Repeat("wild:location",(int)view["wild_location_count"]).ToArray()},order=view["order"],merchants=view["merchants"],position=view.DeepClone()});
        }
    }
}
