// Optional presentation/AI routing overlay. No rules methods are changed.
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Mono.Cecil;
using Mono.Cecil.Cil;
class LegacyWeaver
{
    static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types){foreach(var t in types){yield return t;foreach(var n in Types(t.NestedTypes))yield return n;}}
    static Dictionary<string,string> Bodies(AssemblyDefinition a){return Types(a.MainModule.Types).SelectMany(t=>t.Methods).Where(m=>m.HasBody).ToDictionary(m=>m.FullName,m=>string.Join("\n",m.Body.Instructions.Select(i=>i.OpCode.Name+" "+i.Operand)));}
    static int Main(string[] args)
    {
        var resolver=new DefaultAssemblyResolver();resolver.AddSearchDirectory(Path.GetDirectoryName(args[0]));resolver.AddSearchDirectory(Path.GetDirectoryName(args[1]));
        var rp=new ReaderParameters{AssemblyResolver=resolver};
        using(var game=AssemblyDefinition.ReadAssembly(args[0],rp))using(var helper=AssemblyDefinition.ReadAssembly(args[1],rp))
        {
            var before=Bodies(game);var bridge=helper.MainModule.Types.Single(t=>t.FullName=="BrassLegacyAI.UnityBridge");
            var turn=game.MainModule.Types.Single(t=>t.FullName=="BrassApplication.UseCases.Game.PlayAiTurn.PlayAiTurnUseCase").Methods.Single(m=>m.Name=="PlayAITurn");
            var il=turn.Body.GetILProcessor();var first=turn.Body.Instructions[0];
            foreach(var instruction in new[]{il.Create(OpCodes.Ldarg_0),il.Create(OpCodes.Ldarg_1),il.Create(OpCodes.Call,game.MainModule.ImportReference(bridge.Methods.Single(m=>m.Name=="TryStart"))),il.Create(OpCodes.Brfalse,first),il.Create(OpCodes.Ret)})il.InsertBefore(first,instruction);
            var factory=game.MainModule.Types.Single(t=>t.FullName=="BrassApplication.AI.AiBehaviors.AiBehaviorFactory").Methods.Single(m=>m.Name=="CreateAiBehavior");
            il=factory.Body.GetILProcessor();
            foreach(var ret in factory.Body.Instructions.Where(i=>i.OpCode==OpCodes.Ret).ToArray())
            {il.InsertBefore(ret,il.Create(OpCodes.Ldarg_0));il.InsertBefore(ret,il.Create(OpCodes.Call,game.MainModule.ImportReference(bridge.Methods.Single(m=>m.Name=="Factory"))));}
            game.Write(args[2]);
            using(var check=AssemblyDefinition.ReadAssembly(args[2],rp))
            {
                var after=Bodies(check);var changes=before.Keys.Where(k=>before[k]!=after[k]).ToArray();
                if(changes.Length!=2 || changes.Any(k=>k!=turn.FullName && k!=factory.FullName))throw new Exception("Unexpected overlay changes");
                File.WriteAllLines(args[2]+".methods.txt",changes);
            }
        }
        return 0;
    }
}
