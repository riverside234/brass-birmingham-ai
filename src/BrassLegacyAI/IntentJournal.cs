using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    // Game-thread persistence. Workers receive one detached seat record only.
    public sealed class IntentJournal
    {
        readonly string path;
        readonly JObject header;
        JArray records=new JArray();
        public IntentJournal(string path,string game,JArray profiles)
        {
            this.path=path;
            header=new JObject{["format"]="legacy-intent-journal/1.0.0",["game_id"]=game,["rules"]=Wire.Rules,["schema"]=Wire.Version,["profiles"]=profiles.DeepClone()};
            if(File.Exists(path))
            {
                var saved=JObject.Parse(File.ReadAllText(path));
                if(!JToken.DeepEquals(header,saved["header"]) || (string)saved["sha256"]!=Wire.Hash(saved["records"]))throw new Exception("intent_journal_mismatch");
                records=(JArray)saved["records"].DeepClone();
            }
        }
        public IntentState Read(int seat,int revision,string profile)
        {
            var item=records.Where(r=>(int)r["seat"]==seat && (int)r["revision"]<=revision).OrderByDescending(r=>(int)r["revision"]).FirstOrDefault();
            if(item==null)return null;
            var state=item["intent"].ToObject<IntentState>();
            if(state.seat!=seat || state.profile!=profile)throw new Exception("intent_seat_profile_mismatch");
            return state;
        }
        public void Commit(int revision,IntentState intent)
        {
            if(intent==null)return;
            if(revision<intent.revision)throw new Exception("intent_future_revision");
            // A save/undo branch can read an older record; committing there removes
            // abandoned future records for every seat, without borrowing their intent.
            records=new JArray(records.Where(r=>(int)r["revision"]<=revision && !((int)r["revision"]==revision && (int)r["seat"]==intent.seat)).Select(r=>r.DeepClone()));
            records.Add(new JObject{["seat"]=intent.seat,["revision"]=revision,["intent"]=JObject.FromObject(intent)});
            var payload=new JObject{["header"]=header.DeepClone(),["records"]=records,["sha256"]=Wire.Hash(records)};
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            string temporary=path+".tmp";File.WriteAllText(temporary,Wire.Json(payload));
            if(File.Exists(path))File.Replace(temporary,path,null);else File.Move(temporary,path);
        }
    }
}
