using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Newtonsoft.Json;

public static class PreservedRuntimeChecks
{
    static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types)
    { foreach (var type in types) { yield return type; foreach (var nested in Types(type.NestedTypes)) yield return nested; } }
    static string Body(MethodDefinition method)
    {
        if (!method.HasBody) return "no body";
        var code = method.Body.Instructions; Func<Instruction, int> index = i => i == null ? -1 : code.IndexOf(i);
        Func<object, string> operand = o => {
            if (o == null) return "";
            var branch = o as Instruction; if (branch != null) return "branch " + index(branch);
            var branches = o as Instruction[]; if (branches != null) return "switch " + String.Join(",", branches.Select(index));
            var variable = o as VariableDefinition; if (variable != null) return "variable " + variable.Index;
            var parameter = o as ParameterDefinition; if (parameter != null) return "parameter " + parameter.Index;
            return o.ToString();
        };
        return String.Join("\n", code.Select(i => i.OpCode.Name + " " + operand(i.Operand))) +
            "\nvariables " + String.Join(",", method.Body.Variables.Select(v => v.VariableType.FullName)) +
            "\nhandlers " + String.Join(";", method.Body.ExceptionHandlers.Select(h => h.HandlerType + ":" + h.CatchType + ":" + index(h.TryStart) + ":" + index(h.TryEnd) + ":" + index(h.HandlerStart) + ":" + index(h.HandlerEnd) + ":" + index(h.FilterStart)));
    }
    static string Hash(byte[] bytes) { using (var sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", ""); }
    public static int Main(string[] args)
    {
        bool passed = true; var rows = new List<object>();
        foreach (var file in new[] { "Assembly-CSharp.dll", "BrassPatch.dll", "BrassLegacyAI.dll" })
        using (var a = AssemblyDefinition.ReadAssembly(Path.Combine(args[0], file)))
        using (var b = AssemblyDefinition.ReadAssembly(Path.Combine(args[1], file)))
        {
            var before = Types(a.MainModule.Types).SelectMany(t => t.Methods).ToDictionary(m => m.FullName);
            var after = Types(b.MainModule.Types).SelectMany(t => t.Methods).ToDictionary(m => m.FullName);
            var missing = before.Keys.Except(after.Keys).ToArray();
            var changed = before.Values.Where(m => after.ContainsKey(m.FullName) && Body(m) != Body(after[m.FullName])).ToArray();
            var added = after.Values.Where(m => !before.ContainsKey(m.FullName)).ToArray();
            Func<string,bool> menu = name => name.StartsWith("BrassLegacyAI.LocalMenu") || name.StartsWith("BrassLegacyAI.AiSelectionPanel") || name.StartsWith("BrassLegacyAI.LocalSetupPreferences");
            bool ok = missing.All(n => menu(before[n].DeclaringType.FullName)) && changed.All(m => menu(m.DeclaringType.FullName)) && added.All(m => menu(m.DeclaringType.FullName));
            if (file != "BrassLegacyAI.dll") ok &= Hash(File.ReadAllBytes(Path.Combine(args[0], file))) == Hash(File.ReadAllBytes(Path.Combine(args[1], file)));
            var oldResources = a.MainModule.Resources.OfType<EmbeddedResource>().ToDictionary(r => r.Name, r => Hash(r.GetResourceData()));
            var resources = b.MainModule.Resources.OfType<EmbeddedResource>().ToDictionary(r => r.Name, r => Hash(r.GetResourceData()));
            ok &= oldResources.Count == resources.Count && oldResources.All(p => resources.ContainsKey(p.Key) && resources[p.Key] == p.Value);
            passed &= ok;
            rows.Add(new { file, passed = ok, existingMethods = before.Count, unchangedMethods = before.Count - changed.Length,
                changed = changed.Select(m => m.FullName), added = added.Select(m => m.FullName), missing, embeddedResources = resources.Count });
        }
        File.WriteAllText(args[2], JsonConvert.SerializeObject(new { passed, assemblies = rows }, Formatting.Indented)); return passed ? 0 : 1;
    }
}
