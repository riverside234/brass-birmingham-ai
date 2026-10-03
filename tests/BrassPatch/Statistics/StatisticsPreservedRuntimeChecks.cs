using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Newtonsoft.Json;

public static class PreservedRuntimeChecks
{
    static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types)
    { foreach(var type in types){yield return type;foreach(var child in Types(type.NestedTypes))yield return child;} }
    static string Body(MethodDefinition method)
    {
        if(!method.HasBody)return "no body";
        var instructions=method.Body.Instructions;
        Func<Instruction,int> index=i=>i==null?-1:instructions.IndexOf(i);
        Func<object,string> operand=o=>{
            if(o==null)return "";
            var branch=o as Instruction;if(branch!=null)return "branch "+index(branch);
            var branches=o as Instruction[];if(branches!=null)return "switch "+String.Join(",",branches.Select(index));
            var variable=o as VariableDefinition;if(variable!=null)return "variable "+variable.Index;
            var parameter=o as ParameterDefinition;if(parameter!=null)return "parameter "+parameter.Index;
            return o.ToString();
        };
        return String.Join("\n",instructions.Select(i=>i.OpCode.Name+" "+operand(i.Operand)))+
            "\nvariables "+String.Join(",",method.Body.Variables.Select(v=>v.VariableType.FullName))+
            "\nhandlers "+String.Join(";",method.Body.ExceptionHandlers.Select(h=>h.HandlerType+":"+h.CatchType+":"+index(h.TryStart)+":"+index(h.TryEnd)+":"+index(h.HandlerStart)+":"+index(h.HandlerEnd)+":"+index(h.FilterStart)));
    }
    static bool StatisticsGeneratedOnly(MethodDefinition method,Dictionary<string,MethodDefinition> methods)
    {
        var type=method.DeclaringType;
        if(!type.FullName.StartsWith("<>f__AnonymousType")||!type.CustomAttributes.Any(a=>a.AttributeType.FullName=="System.Runtime.CompilerServices.CompilerGeneratedAttribute"))return false;
        // Anonymous type ordinals are assigned across the compiled helper. New
        // Statistics groupings can renumber them without changing game helpers.
        // Exempt only types with no references from any non-Statistics runtime
        // method/field. Never exempt an arbitrary compiler-generated closure.
        foreach(var candidate in methods.Values){
            if(candidate.DeclaringType.FullName.StartsWith("BrassPatch.Statistics.")||candidate.DeclaringType.FullName.StartsWith("<>f__AnonymousType"))continue;
            if(candidate.FullName.Contains(type.FullName)||Body(candidate).Contains(type.FullName))return false;
        }
        foreach(var owner in Types(type.Module.Types)){
            if(owner.FullName.StartsWith("BrassPatch.Statistics.")||owner.FullName.StartsWith("<>f__AnonymousType"))continue;
            if(owner.Fields.Any(f=>f.FieldType.FullName.Contains(type.FullName)))return false;
        }
        return true;
    }
    public static int Main(string[] args)
    {
        bool informationUi=args.Length>3&&args[3]=="information-ui";
        bool appearance=informationUi||args.Length>3&&(args[3]=="appearance"||args[3]=="statistics");
        string allowedNamespace=args.Length>3&&args[3]=="statistics"?"BrassPatch.Statistics.":"BrassPatch.CardTracker.";
        bool statisticsOnly=args.Length>3&&args[3]=="statistics";
        string[] allowedTypes={"BrassPatch.CardTracker.CardTrackerOverlay","BrassPatch.Statistics.StatisticsEntryPoints","BrassPatch.Statistics.StatisticsEntryGeometry","BrassPatch.Statistics.StatisticsEntrySurface","BrassPatch.Statistics.StatisticsPanel"};
        Func<MethodDefinition,bool> uiMethod=m=>informationUi?allowedTypes.Any(t=>m.DeclaringType.FullName==t||m.DeclaringType.FullName.StartsWith(t+"/")):m.DeclaringType.FullName.StartsWith(allowedNamespace);
        var results=new List<object>();bool passed=true;
        foreach(string file in new[]{"Assembly-CSharp.dll","BrassPatch.dll"}) {
            using(var before=AssemblyDefinition.ReadAssembly(Path.Combine(args[0],file)))
            using(var after=AssemblyDefinition.ReadAssembly(Path.Combine(args[1],file))) {
                var oldMethods=Types(before.MainModule.Types).SelectMany(t=>t.Methods).ToDictionary(m=>m.FullName);
                var methods=Types(after.MainModule.Types).SelectMany(t=>t.Methods).ToDictionary(m=>m.FullName);
                var changed=new List<string>();var missing=new List<string>();
                foreach(var pair in oldMethods) {
                    MethodDefinition method;if(!methods.TryGetValue(pair.Key,out method)){missing.Add(pair.Key);continue;}
                    if(Body(pair.Value)!=Body(method))changed.Add(pair.Key);
                }
                var added=methods.Where(p=>!oldMethods.ContainsKey(p.Key)).Select(p=>p.Value).ToArray();
                Func<MethodDefinition,bool> allowedOld=m=>uiMethod(m)||statisticsOnly&&StatisticsGeneratedOnly(m,oldMethods);
                Func<MethodDefinition,bool> allowedNew=m=>uiMethod(m)||statisticsOnly&&StatisticsGeneratedOnly(m,methods);
                bool allowed=appearance?(file=="Assembly-CSharp.dll"?missing.Count==0&&changed.Count==0&&added.Length==0:
                    missing.All(m=>allowedOld(oldMethods[m]))&&
                    changed.All(m=>allowedOld(oldMethods[m])&&allowedNew(methods[m]))&&
                    added.All(m=>allowedNew(m)||m.DeclaringType.FullName.StartsWith("<PrivateImplementationDetails>"))):missing.Count==0 && (file=="Assembly-CSharp.dll"?
                    changed.Count==1&&methods[changed[0]].DeclaringType.FullName=="InterfaceAdapters.Game.Board.BoardPresenter"&&methods[changed[0]].Name=="UpdateBoardView"&&
                    added.Length==0:
                    changed.Count==1&&methods[changed[0]].DeclaringType.FullName=="BrassPatch.Statistics.StatisticsEntryPoints"&&methods[changed[0]].Name=="ProcessInput"&&
                    added.All(m=>m.DeclaringType.FullName.StartsWith(allowedNamespace)||m.DeclaringType.FullName.StartsWith("<PrivateImplementationDetails>")));
                var generated=statisticsOnly?missing.Concat(changed).Select(m=>oldMethods[m]).Where(m=>!uiMethod(m)&&allowedOld(m)).Select(m=>m.DeclaringType.FullName).Concat(added.Where(m=>!uiMethod(m)&&StatisticsGeneratedOnly(m,methods)).Select(m=>m.DeclaringType.FullName)).Distinct().ToArray():new string[0];
                passed&=allowed;results.Add(new{file,passed=allowed,existingMethods=oldMethods.Count,unchangedMethods=oldMethods.Count-changed.Count-missing.Count,changed,missing,added=added.Select(m=>m.FullName).ToArray(),statisticsOwnedAnonymousTypes=generated});
            }
        }
        File.WriteAllText(args[2],JsonConvert.SerializeObject(new{passed,assemblies=results},Formatting.Indented));
        return passed?0:1;
    }
}
