// A startup-only hook in a disposable COPY. No game rules methods are changed.
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

public static class UnityWeaver
{
    static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types){foreach(var t in types){yield return t;foreach(var n in Types(t.NestedTypes))yield return n;}}
    static Dictionary<string,string> Bodies(AssemblyDefinition a){return Types(a.MainModule.Types).SelectMany(t=>t.Methods).Where(m=>m.HasBody).ToDictionary(m=>m.FullName,m=>string.Join("\n",m.Body.Instructions.Select(i=>i.OpCode.Name+" "+(i.Operand==null?"":i.Operand.ToString()))));}
    static string Hash(string file){using(var s=File.OpenRead(file))using(var h=SHA256.Create())return BitConverter.ToString(h.ComputeHash(s)).Replace("-","").ToLowerInvariant();}
    public static int Main(string[] args)
    {
        var resolver=new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0])));
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1])));
        var settings=new ReaderParameters{AssemblyResolver=resolver};
        using(var game=AssemblyDefinition.ReadAssembly(args[0],settings))
        using(var helper=AssemblyDefinition.ReadAssembly(args[1],settings))
        {
            var before=Bodies(game);
            var type=game.MainModule.Types.Single(t=>t.FullName=="Frameworks.Init.InitUnity");
            var run=helper.MainModule.Types.Single(t=>t.Name=="UnityEntry").Methods.Single(m=>m.Name=="Run");
            foreach(string name in new[]{"Awake","OnDestroy"})
            {
                var method=type.Methods.Single(m=>m.Name==name);method.Body=new MethodBody(method);
                var il=method.Body.GetILProcessor();
                if(name=="Awake")il.Emit(OpCodes.Call,game.MainModule.ImportReference(run));
                il.Emit(OpCodes.Ret);
            }
            game.Write(args[2]);
            using(var output=AssemblyDefinition.ReadAssembly(args[2],settings))
            {
                var after=Bodies(output);
                var changed=before.Keys.Where(k=>before[k]!=after[k]).ToArray();
                if(before.Count!=after.Count || changed.Length!=2 || changed.Any(k=>!k.Contains("Frameworks.Init.InitUnity::Awake(") && !k.Contains("Frameworks.Init.InitUnity::OnDestroy(")))throw new Exception("Startup copy changed an unexpected method");
                File.WriteAllText(args[2]+".startup-hook.json","{\"input_sha256\":\""+Hash(args[0])+"\",\"output_sha256\":\""+Hash(args[2])+"\",\"only_two_startup_methods_changed\":true}");
            }
        }
        return 0;
    }
}
