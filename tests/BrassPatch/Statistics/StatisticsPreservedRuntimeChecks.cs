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
    public static int Main(string[] args)
    {
        bool informationUi=args.Length>3&&args[3]=="information-ui";
        bool appearance=informationUi||args.Length>3&&(args[3]=="appearance"||args[3]=="statistics");
        string allowedNamespace=args.Length>3&&args[3]=="statistics"?"BrassPatch.Statistics.":"BrassPatch.CardTracker.";
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
                bool allowed=appearance?(file=="Assembly-CSharp.dll"?missing.Count==0&&changed.Count==0&&added.Length==0:
                    missing.All(m=>uiMethod(oldMethods[m]))&&
                    changed.All(m=>uiMethod(methods[m]))&&
                    added.All(m=>uiMethod(m)||m.DeclaringType.FullName.StartsWith("<PrivateImplementationDetails>"))):missing.Count==0 && (file=="Assembly-CSharp.dll"?
                    changed.Count==1&&methods[changed[0]].DeclaringType.FullName=="InterfaceAdapters.Game.Board.BoardPresenter"&&methods[changed[0]].Name=="UpdateBoardView"&&
                    added.Length==0:
                    changed.Count==1&&methods[changed[0]].DeclaringType.FullName=="BrassPatch.Statistics.StatisticsEntryPoints"&&methods[changed[0]].Name=="ProcessInput"&&
                    added.All(m=>m.DeclaringType.FullName.StartsWith(allowedNamespace)||m.DeclaringType.FullName.StartsWith("<PrivateImplementationDetails>")));
                passed&=allowed;results.Add(new{file,passed=allowed,existingMethods=oldMethods.Count,unchangedMethods=oldMethods.Count-changed.Count-missing.Count,changed,missing,added=added.Select(m=>m.FullName).ToArray()});
            }
        }
        File.WriteAllText(args[2],JsonConvert.SerializeObject(new{passed,assemblies=results},Formatting.Indented));
        return passed?0:1;
    }
}
