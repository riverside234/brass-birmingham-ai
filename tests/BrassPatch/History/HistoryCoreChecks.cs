using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using BrassPatch.History;
using BrassPatch.Statistics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public static class HistoryCoreChecks
{
    static readonly DateTime epoch = new DateTime(2026, 10, 3, 12, 0, 0, DateTimeKind.Utc);
    public static void Set(object value, string name, object data) { value.GetType().GetProperty(name).SetValue(value, data, null); }
    static JToken Observe(object value)
    {
        if (value == null) return JValue.CreateNull();
        var type = value.GetType();
        if (type.IsPrimitive || type.IsEnum || value is string || value is decimal) return JToken.FromObject(value);
        if (value is IDictionary) {
            var values=new List<DictionaryEntry>();var iterator=((IDictionary)value).GetEnumerator();while(iterator.MoveNext())values.Add(iterator.Entry);
            return new JArray(values.OrderBy(p=>p.Key.ToString()).Select(p=>new JObject{["key"]=Observe(p.Key),["value"]=Observe(p.Value)}));
        }
        if (value is IEnumerable) return new JArray(((IEnumerable)value).Cast<object>().Select(Observe));
        var result = new JObject();
        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(p => p.GetIndexParameters().Length == 0).OrderBy(p => p.Name)) result[property.Name] = Observe(property.GetValue(value, null));
        foreach (var field in type.GetFields(BindingFlags.NonPublic | BindingFlags.Instance).Where(f => !f.Name.Contains("BackingField")).OrderBy(f => f.Name)) result["backing:" + field.Name] = Observe(field.GetValue(value));
        return result;
    }
    static CompletedMatchStatistics Copy(CompletedMatchStatistics report, string id)
    { var result = GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(report)); Set(result, "GameId", id); return result; }
    static void Reject(Action action, Action<bool, string> check, string reason)
    { bool rejected = false; try { action(); } catch (Exception) { rejected = true; } check(rejected, reason); }
    static JObject Json(string text)
    { using(var reader=new JsonTextReader(new StringReader(text)){DateParseHandling=DateParseHandling.None})return JObject.Load(reader); }
    public static void Roundtrip(HistoryFixture fixture, Action<bool, string> check)
    {
        var report = fixture.Report; check(report.CompleteHistory, "H-DATA completed fixture history");
        var copy = GameHistoryStatisticsCodec.Record(GameHistoryStatisticsCodec.Envelope(report, 1, epoch, "fixture")).Report;
        var before = (JObject)Observe(report); var after = (JObject)Observe(copy);
        foreach (var property in before.Properties()) check(JToken.DeepEquals(property.Value, after[property.Name]), "H-DATA all values/backing data: " + report.Players.Count + "/" + report.Introductory + "/" + property.Name);
        foreach (var player in report.Players) check(copy.Rounds.All(r => r.Spent(player.Color) == report.Rounds.First(o => o.Era == r.Era && o.Round == r.Round).Spent(player.Color)), "H-DATA round spending " + player.Color);
        var signed=Copy(report,"signed-fixture");Set(signed.Players[0],"FinalVP",-7);Set(signed.Players[0],"FinalIncome",-3);
        Set(signed.Players[0],"Rank",1);Set(signed.Players[1],"Rank",1);
        var signedCopy=GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(signed));
        check(signedCopy.Players[0].FinalVP==-7&&signedCopy.Players[0].FinalIncome==-3&&signedCopy.Players[0].Rank==signedCopy.Players[1].Rank,"H-DATA signed final values and tied ranks preserved");
        var missing=GameHistoryStatisticsCodec.Encode(report);((JObject)missing["actions"].First).Remove("cards");
        Reject(()=>GameHistoryStatisticsCodec.Decode(missing),check,"H-DATA missing card backing data rejected");
        foreach (int era in report.Introductory ? new[] { 0, 1 } : new[] { 0, 1, 2 })
        foreach (bool cash in new[] { false, true })
        foreach (string resources in new[] { "Supply", "Market", "Merchants" }) {
            var filters = new StatisticsFilters(); filters.Reset(report); filters.Era = era; filters.AllCashPayments = cash; filters.ResourceView = resources;
            for (int page = 0; page < 10; page++) {
                check(JToken.DeepEquals(JToken.FromObject(StatisticsPages.Build(page, report, filters)), JToken.FromObject(StatisticsPages.Build(page, copy, filters))), "H-DATA ledger parity " + page + "/" + era + "/" + cash + "/" + resources);
                check(JToken.DeepEquals(JToken.FromObject(StatisticsPlotPages.Build(page, report, filters)), JToken.FromObject(StatisticsPlotPages.Build(page, copy, filters))), "H-DATA plot/evidence parity " + page + "/" + era + "/" + cash + "/" + resources);
            }
        }
    }
    public static void Store(string root, HistoryFixture fixture, Action<bool, string> check)
    {
        Directory.CreateDirectory(root); string protectedFile = Path.Combine(root, "metadata.storage"); File.WriteAllText(protectedFile, "protected saved game");
        var summary = StatisticsCollector.Summary(fixture.App, "Fixture summary only"); int clock = 0;
        var store = new GameHistoryStore(Path.Combine(root, "retention"), () => epoch.AddMinutes(50 - ++clock));
        check(store.List().Games.Count == 0, "H-RETENTION empty archive");
        for (int n = 1; n <= 25; n++) {
            Set(summary, "GameId", "retained-" + n); check(store.Record(summary) == HistoryAdmission.Added, "H-RETENTION admission " + n);
            var list = store.List(); check(list.Error == null && list.Games.Count == Math.Min(10, n) && list.Games.Select(g => g.Sequence).SequenceEqual(Enumerable.Range(Math.Max(1, n - 9), Math.Min(10, n)).Reverse().Select(v => (long)v)), "H-RETENTION completion order/count " + n);
        }
        var reopened = new GameHistoryStore(store.Root); check(reopened.List().Games.Count == 10, "H-RETENTION durable ten after restart");
        Set(summary, "GameId", "retained-1"); check(reopened.Record(summary) == HistoryAdmission.AlreadyRetired && reopened.List().Games.Count == 10, "H-RETENTION evicted game cannot return as newest");
        Set(summary, "GameId", "retained-25"); var date = reopened.List().Games[0].RecordedUtc;
        check(reopened.Record(summary) == HistoryAdmission.Unchanged && reopened.List().Games[0].Sequence == 25 && reopened.List().Games[0].RecordedUtc == date, "H-RETENTION repeated review preserves sequence/date");
        var full = Copy(fixture.Report, "retained-25");
        check(reopened.Record(full) == HistoryAdmission.Updated && reopened.Load(GameHistoryStatisticsCodec.Key(full.GameId)).Report.CompleteHistory && reopened.List().Games[0].RecordedUtc == date, "H-DATA summary upgrade preserves time/order");
        check(reopened.Record(summary) == HistoryAdmission.Unchanged && reopened.Load(GameHistoryStatisticsCodec.Key(full.GameId)).Report.CompleteHistory, "H-DATA weak recapture cannot downgrade details");
        check(File.ReadAllText(protectedFile) == "protected saved game" && Directory.GetFiles(store.Root).Select(Path.GetFileName).SequenceEqual(new[] { "history.json" }), "H-RETENTION no save deletion or hidden report backups");
        Reject(() => reopened.Load("../../metadata.storage"), check, "H-RETENTION selection paths rejected");
        var locked=new GameHistoryStore(Path.Combine(root,"locked"),()=>epoch);Set(summary,"GameId","lock-1");locked.Record(summary);
        var lockPath=Path.Combine(locked.Root,"history.json");string intact=File.ReadAllText(lockPath);
        using(var handle=new FileStream(lockPath,FileMode.Open,FileAccess.Read,FileShare.Read)){
            Set(summary,"GameId","lock-2");Reject(()=>locked.Record(summary),check,"H-RECOVERY locked publication fails safely");
            check(File.ReadAllText(lockPath)==intact,"H-RECOVERY locked canonical report unchanged");
        }
        check(locked.List().Games.Count==2&&locked.List().Games[0].Sequence==2,"H-RECOVERY complete pending write recovers after lock release");
        foreach (string stage in new[] { "before-write", "after-flush", "before-publish", "after-publish" }) {
            var interrupted = new GameHistoryStore(Path.Combine(root, "fault-" + stage), () => epoch);
            for (int n = 1; n <= 10; n++) { Set(summary, "GameId", stage + n); interrupted.Record(summary); }
            string path = Path.Combine(interrupted.Root, "history.json"), previous = File.ReadAllText(path);
            interrupted.Fault = boundary => { if (boundary == stage) throw new IOException("Injected publication interruption"); };
            Set(summary, "GameId", stage + "11"); Reject(() => interrupted.Record(summary), check, "H-RECOVERY fault injection " + stage);
            if (stage != "after-publish") check(File.ReadAllText(path) == previous, "H-RECOVERY prior ten preserved until publication " + stage);
            var recovered = new GameHistoryStore(interrupted.Root); var list = recovered.List();
            check(list.Error == null && list.Games.Count == 10 && list.Games[0].Sequence == (stage == "before-write" ? 10 : 11), "H-RECOVERY restart resolves whole admission/retirement " + stage);
            check(!File.Exists(Path.Combine(interrupted.Root, "history.pending.json")), "H-RECOVERY no completed pending backup " + stage);
        }
        var broken = new GameHistoryStore(Path.Combine(root, "corrupt"), () => epoch);
        for (int n = 1; n <= 3; n++) { Set(summary, "GameId", "corrupt-" + n); broken.Record(summary); }
        var bankPath = Path.Combine(broken.Root, "history.json"); string good=File.ReadAllText(bankPath);var bank = Json(good);
        var first = (JObject)bank["games"][0]; first["schema"] = 99;
        File.WriteAllText(bankPath, bank.ToString(Formatting.None));
        check(broken.List().Games.Count == 3 && broken.List().Games[0].Unavailable != null && broken.List().Games.Skip(1).All(g => g.Unavailable == null), "H-RECOVERY unknown report version isolates one row");
        Reject(() => broken.Load((string)first["key"]), check, "H-RECOVERY unknown version cannot become zero-filled plots");
        broken.Fault=stage=>{if(stage=="after-flush")throw new IOException("Interrupted admission alongside unavailable older report");};Set(summary,"GameId","corrupt-4");
        Reject(()=>broken.Record(summary),check,"H-RECOVERY unavailable older row with interrupted new admission");broken.Fault=null;
        var alongside=broken.List();check(alongside.Error==null&&alongside.Games.Count==4&&alongside.Games[0].Sequence==4&&alongside.Games[1].Unavailable!=null,"H-RECOVERY new valid report recovers alongside unavailable older row");
        bank = Json(good); first = (JObject)bank["games"][0]; first["payload"]["players"][0]["FinalVP"] = 99999; File.WriteAllText(bankPath, bank.ToString(Formatting.None));
        check(broken.List().Games[0].Unavailable != null && broken.List().Games[1].Unavailable == null, "H-RECOVERY payload integrity checked per report");
        File.WriteAllText(bankPath, good); File.WriteAllText(Path.Combine(broken.Root, "history.pending.json"), "{half written");
        check(broken.List().Games.Count == 3 && !File.Exists(Path.Combine(broken.Root, "history.pending.json")), "H-RECOVERY truncated temporary write leaves canonical bank intact");
        bank = Json(good); bank["format"] = "future/2"; var future = bank.ToString(Formatting.None); File.WriteAllText(bankPath, future);
        check(broken.List().Error != null, "H-RECOVERY unknown bank has a recoverable error state");
        Reject(() => broken.Record(summary), check, "H-RECOVERY unsupported bank cannot be overwritten");
        check(File.ReadAllText(bankPath) == future, "H-RECOVERY future file preserved");
    }
}
