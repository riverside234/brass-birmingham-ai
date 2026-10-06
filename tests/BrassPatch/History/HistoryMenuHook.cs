using System;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

public static class HistoryMenuHook
{
    public static int Main(string[] args)
    {
        var resolver=new DefaultAssemblyResolver();resolver.AddSearchDirectory(args[3]);resolver.AddSearchDirectory(Path.GetDirectoryName(args[1]));
        var settings=new ReaderParameters{AssemblyResolver=resolver};
        using (var game = AssemblyDefinition.ReadAssembly(args[0],settings))
        using (var helper = AssemblyDefinition.ReadAssembly(args[1],settings)) {
            var type = game.MainModule.Types.Single(t => t.FullName == "Frameworks.View.Menu.MainMenuView");
            var method = type.Methods.Single(m => m.Name == "PrepareReferences" && m.Parameters.Count == 0);
            var bind = helper.MainModule.Types.Single(t => t.FullName == "BrassPatch.History.HistoryEntryPoints").Methods.Single(m => m.Name == "Bind");
            if (!method.HasBody || bind.Parameters.Count != 1 || bind.Parameters[0].ParameterType.FullName != type.FullName || method.Body.Instructions.Any(i => i.Operand is MethodReference && ((MethodReference)i.Operand).DeclaringType.FullName == bind.DeclaringType.FullName))
                throw new InvalidOperationException("Unexpected or duplicate main-menu hook.");
            var returns = method.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray();
            if (returns.Length != 1 || method.Body.ExceptionHandlers.Count != 0) throw new InvalidOperationException("Main-menu initialization shape changed.");
            var processor = method.Body.GetILProcessor(); var load = Instruction.Create(OpCodes.Ldarg_0);
            processor.InsertBefore(returns[0], load); processor.InsertBefore(returns[0], Instruction.Create(OpCodes.Call, game.MainModule.ImportReference(bind)));
            foreach (var code in method.Body.Instructions) if (ReferenceEquals(code.Operand, returns[0])) code.Operand = load;
            game.Write(args[2]);
        }
        Console.WriteLine("Added one main-menu initialization hook; existing offline click routing retained."); return 0;
    }
}
