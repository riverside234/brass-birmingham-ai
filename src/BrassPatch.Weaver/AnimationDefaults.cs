using System;
using System.Linq;
using System.IO;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Scoped overlay: preserve every instruction except these two initializer operands.
public static class AnimationDefaults
{
    public static int Main(string[] args)
    {
        try {
            if (args.Length != 2) throw new ArgumentException("input.dll output.dll");
            var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0])));
            using (var assembly = AssemblyDefinition.ReadAssembly(args[0], new ReaderParameters { AssemblyResolver = resolver })) {
                var type = assembly.MainModule.Types.Single(t => t.FullName == "Frameworks.Utils.UIProperties");
                var ctor = type.Methods.Single(m => m.Name == ".cctor");
                foreach (string name in new[] { "CurrentAnimationSpeed", "DefaultAnimationSpeed" }) {
                    var store = ctor.Body.Instructions.Single(i => i.OpCode == OpCodes.Stsfld && ((FieldReference)i.Operand).Name == name);
                    if (store.Previous.OpCode != OpCodes.Ldc_R4 || (float)store.Previous.Operand != 1f)
                        throw new InvalidOperationException("Unexpected animation baseline: " + name);
                    store.Previous.Operand = 2f;
                }
                assembly.Write(args[1]);
            }
            Console.WriteLine("Animation startup and reset defaults: 2x. Fast-forward remains 2x.");
            return 0;
        } catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
