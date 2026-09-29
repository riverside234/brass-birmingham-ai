using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Newtonsoft.Json;

public sealed class PatchSpec
{
    public string targetType, targetMethod, helperType, helperMethod, mode;
    public string[] parameterTypes;
}

public static class Program
{
    const string StockHash = "600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB";
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 6 && args[0] == "--overlay") return Overlay(args);
            if (args.Length != 4) throw new ArgumentException("stock.dll helper.dll manifest-directory output.dll");
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(args[0]))
                if (BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "") != StockHash)
                    throw new InvalidOperationException("Unsupported original game assembly. No output written.");
            var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0])));
            resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1])));
            var settings = new ReaderParameters { AssemblyResolver = resolver, ReadWrite = false };
            using (var assembly = AssemblyDefinition.ReadAssembly(args[0], settings))
            using (var helper = AssemblyDefinition.ReadAssembly(args[1], settings))
            {
                var module = assembly.MainModule;
                var config = FindType(module, "BoardGameRules.Entities.Game.GameStartConfiguration");
                config.Fields.Add(new FieldDefinition("BrassRulesVersion", FieldAttributes.Public, module.TypeSystem.Int32));
                // Liquidation events include brewery summary objects. The callback
                // belongs to the live setup, and cannot be restored from JSON.
                var callback = FindType(module, "BoardGameRules.Entities.Industries.Brewery").Fields.Single(f => f.Name == "_numberOfBeersFunc");
                callback.CustomAttributes.Add(new CustomAttribute(module.ImportReference(typeof(JsonIgnoreAttribute).GetConstructor(Type.EmptyTypes))));
                int scopedSites = ScopeBonusState(module, helper.MainModule);
                var patched = new List<string>();
                foreach (var manifest in Directory.GetFiles(args[2], "*.json", SearchOption.AllDirectories).OrderBy(p => p))
                foreach (var spec in JsonConvert.DeserializeObject<PatchSpec[]>(File.ReadAllText(manifest)))
                {
                    var type = FindType(module, spec.targetType);
                    var matches = type.Methods.Where(m => m.Name == spec.targetMethod &&
                        (spec.parameterTypes == null || m.Parameters.Select(p => p.ParameterType.FullName).SequenceEqual(spec.parameterTypes))).ToList();
                    if (matches.Count != 1) throw new InvalidOperationException("Ambiguous/missing target " + spec.targetType + "." + spec.targetMethod + " (" + matches.Count + ")");
                    var method = matches[0];
                    string key = method.FullName;
                    if (patched.Contains(key)) throw new InvalidOperationException("Duplicate patch " + key);
                    var hooks = FindType(helper.MainModule, spec.helperType).Methods.Where(m => m.Name == spec.helperMethod).ToList();
                    if (hooks.Count != 1 || !hooks[0].IsStatic) throw new InvalidOperationException("Missing/ambiguous static hook " + spec.helperType + "." + spec.helperMethod);
                    var hook = hooks[0];
                    Validate(method, hook, spec.mode ?? "replace");
                    Rewrite(module, method, module.ImportReference(hook), spec.mode ?? "replace");
                    patched.Add(key);
                }
                if (patched.Count == 0) throw new InvalidOperationException("No patch manifests found.");
                PatchIntroSort(module, helper.MainModule);
                foreach (var method in AllTypes(module).SelectMany(t => t.Methods).Where(m => m.HasBody))
                foreach (var instruction in method.Body.Instructions)
                    if (instruction.OpCode.OperandType == OperandType.ShortInlineBrTarget)
                        instruction.OpCode = (OpCode)typeof(OpCodes).GetFields().Single(f => f.FieldType == typeof(OpCode) && ((OpCode)f.GetValue(null)).Name == instruction.OpCode.Name.Substring(0, instruction.OpCode.Name.Length - 2)).GetValue(null);
                assembly.Write(args[3]);
                File.WriteAllText(args[3] + ".manifest.json", JsonConvert.SerializeObject(new { stockSha256 = StockHash, patchVersion = "1.4.2", gameOwnedBonusSites = scopedSites, jsonIgnoredFields = new[] { callback.FullName }, methods = patched }, Formatting.Indented));
                Console.WriteLine("Patched " + patched.Count + " methods; scoped " + scopedSites + " bonus references.");
            }
            return 0;
        }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
    static int Overlay(string[] args)
    {
        using (var sha = SHA256.Create()) using (var stream = File.OpenRead(args[1]))
            if (BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "") != args[5].ToUpperInvariant())
                throw new InvalidOperationException("Staged rules assembly changed before AI weaving.");
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1])));
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[2])));
        var settings = new ReaderParameters { AssemblyResolver = resolver };
        using (var game = AssemblyDefinition.ReadAssembly(args[1], settings))
        using (var helper = AssemblyDefinition.ReadAssembly(args[2], settings))
        {
            var before = AllTypes(game.MainModule).SelectMany(t => t.Methods).Where(m => m.HasBody).ToDictionary(m => m.FullName, Body);
            var patched = new List<string>();
            foreach (var file in Directory.GetFiles(args[3], "*.json").OrderBy(p => p))
            foreach (var spec in JsonConvert.DeserializeObject<PatchSpec[]>(File.ReadAllText(file)))
            {
                var type = FindType(game.MainModule, spec.targetType);
                var method = type.Methods.Single(m => m.Name == spec.targetMethod && m.Parameters.Select(p => p.ParameterType.FullName).SequenceEqual(spec.parameterTypes));
                if (patched.Contains(method.FullName) || method.Body.Instructions.Any(i => i.Operand is MethodReference mr && mr.DeclaringType.Scope.Name == helper.Name.Name))
                    throw new InvalidOperationException("Duplicate AI hook: " + method.FullName);
                var hook = FindType(helper.MainModule, spec.helperType).Methods.Single(m => m.Name == spec.helperMethod);
                Validate(method, hook, spec.mode);
                Rewrite(game.MainModule, method, game.MainModule.ImportReference(hook), spec.mode);
                patched.Add(method.FullName);
            }
            foreach (var method in AllTypes(game.MainModule).SelectMany(t => t.Methods).Where(m => m.HasBody))
                foreach (var instruction in method.Body.Instructions)
                    if (instruction.OpCode.OperandType == OperandType.ShortInlineBrTarget)
                        instruction.OpCode = (OpCode)typeof(OpCodes).GetFields().Single(f => f.FieldType == typeof(OpCode) && ((OpCode)f.GetValue(null)).Name == instruction.OpCode.Name.Substring(0, instruction.OpCode.Name.Length - 2)).GetValue(null);
            var changed = AllTypes(game.MainModule).SelectMany(t => t.Methods).Where(m => before.ContainsKey(m.FullName) && before[m.FullName] != Body(m)).Select(m => m.FullName).OrderBy(x => x).ToArray();
            if (!changed.SequenceEqual(patched.OrderBy(x => x))) throw new InvalidOperationException("Unexpected changes outside the AI hook allowlist.");
            game.Write(args[4]);
            var manifest = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(args[1] + ".manifest.json"));
            manifest["localAiMethods"] = new Newtonsoft.Json.Linq.JArray(patched);
            manifest["methods"] = new Newtonsoft.Json.Linq.JArray(manifest["methods"].Select(x => (string)x).Concat(patched));
            File.WriteAllText(args[4] + ".manifest.json", manifest.ToString(Formatting.Indented));
            Console.WriteLine("Added " + patched.Count + " AI/UI hooks; retained all rules hooks.");
            return 0;
        }
    }
    static string Body(MethodDefinition method) { return string.Join("\n", method.Body.Instructions.Select(i => i.OpCode.Name + " " + i.Operand)); }
    static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
    {
        foreach (var type in module.Types) foreach (var item in Walk(type)) yield return item;
    }
    static IEnumerable<TypeDefinition> Walk(TypeDefinition type)
    {
        yield return type;
        foreach (var child in type.NestedTypes) foreach (var item in Walk(child)) yield return item;
    }
    static TypeDefinition FindType(ModuleDefinition module, string name)
    {
        return AllTypes(module).Single(t => t.FullName == name || t.FullName.Replace('/', '+') == name);
    }
    static void Validate(MethodDefinition method, MethodDefinition hook, string mode)
    {
        if (!method.HasBody || method.HasGenericParameters) throw new InvalidOperationException("Unsupported target " + method.FullName);
        int offset = method.IsStatic ? 0 : 1;
        if (hook.Parameters.Count != method.Parameters.Count + offset) throw new InvalidOperationException("Hook parameter count " + hook.FullName);
        if (offset == 1 && hook.Parameters[0].ParameterType.FullName != method.DeclaringType.FullName && hook.Parameters[0].ParameterType.FullName != "System.Object")
            throw new InvalidOperationException("Hook owner type " + hook.FullName);
        for (int i = 0; i < method.Parameters.Count; i++)
            if (hook.Parameters[i + offset].ParameterType.FullName != method.Parameters[i].ParameterType.FullName)
                throw new InvalidOperationException("Hook argument mismatch " + hook.FullName + " expected " + method.Parameters[i].ParameterType.FullName);
        if (hook.ReturnType.FullName != (mode == "replace" ? method.ReturnType.FullName : "System.Void"))
            throw new InvalidOperationException("Hook return type " + hook.FullName);
        if (method.IsConstructor && mode == "replace") throw new InvalidOperationException("Cannot replace constructors.");
        if (mode != "replace" && mode != "prefix" && mode != "postfix") throw new InvalidOperationException("Unknown mode " + mode);
        if (mode == "postfix" && method.ReturnType.FullName != "System.Void") throw new InvalidOperationException("Postfix requires void target.");
    }
    static List<Instruction> CallInstructions(MethodDefinition method, MethodReference hook)
    {
        var list = new List<Instruction>();
        if (!method.IsStatic) list.Add(Instruction.Create(OpCodes.Ldarg_0));
        foreach (var p in method.Parameters) list.Add(Instruction.Create(OpCodes.Ldarg, p));
        list.Add(Instruction.Create(OpCodes.Call, hook));
        return list;
    }
    static void Rewrite(ModuleDefinition module, MethodDefinition method, MethodReference hook, string mode)
    {
        if (mode == "replace")
        {
            var saved = new MethodDefinition("BrassPatch_Original_" + method.Name,
                MethodAttributes.Private | MethodAttributes.HideBySig | (method.IsStatic ? MethodAttributes.Static : 0), method.ReturnType);
            foreach (var p in method.Parameters) saved.Parameters.Add(new ParameterDefinition(p.Name, p.Attributes, p.ParameterType));
            saved.Body = method.Body;
            foreach (var instruction in saved.Body.Instructions)
                if (instruction.Operand is ParameterDefinition p) instruction.Operand = saved.Parameters[p.Index];
            method.DeclaringType.Methods.Add(saved);
            method.Body = new MethodBody(method);
            foreach (var instruction in CallInstructions(method, hook)) method.Body.Instructions.Add(instruction);
            method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
        }
        else
        {
            var il = method.Body.GetILProcessor();
            if (mode == "prefix")
            {
                var first = method.Body.Instructions[0];
                foreach (var instruction in CallInstructions(method, hook)) il.InsertBefore(first, instruction);
            }
            else
            {
                foreach (var ret in method.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray())
                {
                    // Keep the existing return instruction as the branch target.
                    ret.OpCode = OpCodes.Nop;
                    Instruction previous = ret;
                    foreach (var instruction in CallInstructions(method, hook).Concat(new[] { Instruction.Create(OpCodes.Ret) }))
                    { il.InsertAfter(previous, instruction); previous = instruction; }
                }
            }
            method.Body.MaxStackSize = Math.Max(method.Body.MaxStackSize, method.Parameters.Count + 2);
        }
    }
    static int ScopeBonusState(ModuleDefinition module, ModuleDefinition helper)
    {
        var ownerHook = module.ImportReference(FindType(helper, "BrassPatch.StateFixes").Methods.Single(m => m.Name == "BonusForOwner"));
        int reads = 0, writes = 0;
        foreach (var method in AllTypes(module).SelectMany(t => t.Methods).Where(m => m.HasBody))
        foreach (var instruction in method.Body.Instructions.ToArray())
        {
            var field = instruction.Operand as FieldReference;
            if (field == null || field.DeclaringType.FullName != "BoardGameRules.Entities.Actions.DevelopMerchantBonus" || field.Name != "Instance") continue;
            if (instruction.OpCode == OpCodes.Ldsfld)
            {
                var il = method.Body.GetILProcessor();
                if (method.DeclaringType.FullName == "BrassApplication.AI.AIHelpers.SellActionAIHelper" && method.Name == "SellIndustries")
                { instruction.OpCode = OpCodes.Ldarg; instruction.Operand = method.Parameters[4]; }
                else
                {
                    if (method.IsStatic) throw new InvalidOperationException("Unexpected static bonus reader " + method.FullName);
                    instruction.OpCode = OpCodes.Ldarg_0; instruction.Operand = null;
                    il.InsertAfter(instruction, Instruction.Create(OpCodes.Call, ownerHook));
                }
                reads++;
            }
            else if (instruction.OpCode == OpCodes.Stsfld && method.IsConstructor)
            { instruction.OpCode = OpCodes.Pop; instruction.Operand = null; writes++; }
            else throw new InvalidOperationException("Unexpected global bonus access " + method.FullName);
        }
        if (reads != 6 || writes != 1) throw new InvalidOperationException("Unexpected bonus references: " + reads + "/" + writes);
        return reads;
    }
    static void PatchIntroSort(ModuleDefinition module, ModuleDefinition helper)
    {
        var hook = FindType(helper, "BrassPatch.StateFixes").Methods.Single(m => m.Name == "CompareIntroResultRows");
        var row = hook.Parameters[0].ParameterType.FullName;
        var candidates = Walk(FindType(module, "Frameworks.View.Game.EndOfGameView")).SelectMany(t => t.Methods)
            .Where(m => m.Parameters.Count == 2 && m.Parameters.All(p => p.ParameterType.FullName == row) && m.ReturnType.FullName == "System.Int32").ToList();
        if (candidates.Count != 1) throw new InvalidOperationException("Unexpected introductory result comparator shape.");
        var method = candidates[0];
        method.Body = new MethodBody(method);
        method.Body.Instructions.Add(Instruction.Create(OpCodes.Ldarg, method.Parameters[0]));
        method.Body.Instructions.Add(Instruction.Create(OpCodes.Ldarg, method.Parameters[1]));
        method.Body.Instructions.Add(Instruction.Create(OpCodes.Call, module.ImportReference(hook)));
        method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
    }
}
