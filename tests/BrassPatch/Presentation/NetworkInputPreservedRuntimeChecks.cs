using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Newtonsoft.Json;

public static class NetworkInputPreservedRuntimeChecks
{
    static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types)
    { foreach(var type in types){yield return type;foreach(var nested in Types(type.NestedTypes))yield return nested;} }
    static string Body(MethodDefinition method)
    {
        if(!method.HasBody)return "no body";
        var code=method.Body.Instructions;Func<Instruction,int> index=i=>i==null?-1:code.IndexOf(i);
        Func<object,string> operand=o=>{
            if(o==null)return "";var branch=o as Instruction;if(branch!=null)return "branch "+index(branch);
            var branches=o as Instruction[];if(branches!=null)return "switch "+String.Join(",",branches.Select(index));
            var variable=o as VariableDefinition;if(variable!=null)return "variable "+variable.Index;
            var parameter=o as ParameterDefinition;if(parameter!=null)return "parameter "+parameter.Index;return o.ToString();
        };
        return String.Join("\n",code.Select(i=>i.OpCode.Name+" "+operand(i.Operand)))+
            "\nvariables "+String.Join(",",method.Body.Variables.Select(v=>v.VariableType.FullName))+
            "\nhandlers "+String.Join(";",method.Body.ExceptionHandlers.Select(h=>h.HandlerType+":"+h.CatchType+":"+index(h.TryStart)+":"+index(h.TryEnd)+":"+index(h.HandlerStart)+":"+index(h.HandlerEnd)+":"+index(h.FilterStart)));
    }
    static bool Allowed(string file,MethodDefinition method)
    {
        string name=method.DeclaringType.FullName;
        if(file=="BrassPatch.dll")return name.StartsWith("BrassPatch.NetworkInput.NetworkInputFixes",StringComparison.Ordinal);
        if(file!="Assembly-CSharp.dll")return false;
        return (name=="BoardGameRules.Entities.Actions.NetworkAction" && method.Name=="PlaceLink") ||
            (name=="BrassApplication.UseCases.Game.Actions.NetworkAction.NetworkActionUseCase" &&
             (method.Name=="LinkSelected" || method.Name=="BrassPatch_Original_LinkSelected"));
    }
    static string Hash(byte[] bytes){using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-","");}
    public static int Main(string[] args)
    {
        bool passed=true;var reports=new List<object>();
        foreach(string file in new[]{"Assembly-CSharp.dll","BrassPatch.dll","BrassLegacyAI.dll"}){
            using(var before=AssemblyDefinition.ReadAssembly(Path.Combine(args[0],file)))
            using(var after=AssemblyDefinition.ReadAssembly(Path.Combine(args[1],file))){
                var oldMethods=Types(before.MainModule.Types).SelectMany(t=>t.Methods).ToDictionary(m=>m.FullName);
                var methods=Types(after.MainModule.Types).SelectMany(t=>t.Methods).ToDictionary(m=>m.FullName);
                var changed=new List<string>();var missing=new List<string>();
                foreach(var pair in oldMethods){MethodDefinition current;if(!methods.TryGetValue(pair.Key,out current))missing.Add(pair.Key);else if(Body(pair.Value)!=Body(current))changed.Add(pair.Key);}
                var added=methods.Values.Where(m=>!oldMethods.ContainsKey(m.FullName)).ToArray();
                bool scoped=missing.All(m=>Allowed(file,oldMethods[m]))&&changed.All(m=>Allowed(file,oldMethods[m])&&Allowed(file,methods[m]))&&
                    added.All(m=>Allowed(file,m));
                var oldResources=before.MainModule.Resources.OfType<EmbeddedResource>().ToDictionary(r=>r.Name,r=>Hash(r.GetResourceData()));
                var resources=after.MainModule.Resources.OfType<EmbeddedResource>().ToDictionary(r=>r.Name,r=>Hash(r.GetResourceData()));
                bool frozen=oldResources.Count==resources.Count&&oldResources.All(p=>resources.ContainsKey(p.Key)&&resources[p.Key]==p.Value);
                bool originalPreserved=true;
                if(file=="Assembly-CSharp.dll") {
                    var old=oldMethods.Values.Single(m=>m.DeclaringType.FullName=="BrassApplication.UseCases.Game.Actions.NetworkAction.NetworkActionUseCase" && m.Name=="LinkSelected");
                    var saved=methods.Values.Single(m=>m.DeclaringType.FullName==old.DeclaringType.FullName && m.Name=="BrassPatch_Original_LinkSelected");
                    originalPreserved=Body(old)==Body(saved) && changed.Count==2 && missing.Count==0 && added.Length==1;
                }
                if(file=="BrassLegacyAI.dll")originalPreserved=Hash(File.ReadAllBytes(Path.Combine(args[0],file)))==Hash(File.ReadAllBytes(Path.Combine(args[1],file)));
                passed&=scoped&&frozen&&originalPreserved;
                reports.Add(new{file,passed=scoped&&frozen&&originalPreserved,originalPreserved,existingMethods=oldMethods.Count,unchangedMethods=oldMethods.Count-changed.Count-missing.Count,
                    changed,missing,added=added.Select(m=>m.FullName).ToArray(),embeddedResources=resources.Count,embeddedResourcesPreserved=frozen});
            }
        }
        File.WriteAllText(args[2],JsonConvert.SerializeObject(new{passed,assemblies=reports},Formatting.Indented));return passed?0:1;
    }
}

