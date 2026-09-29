using System;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace BrassPatch
{
    public static class ReflectionAccess
    {
        public const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        public static object Get(object owner, string name)
        {
            if (owner == null) throw new ArgumentNullException("owner");
            for (Type t = owner.GetType(); t != null; t = t.BaseType)
            {
                var field = t.GetField(name, Flags | BindingFlags.DeclaredOnly);
                if (field != null) return field.GetValue(owner);
                var property = t.GetProperty(name, Flags | BindingFlags.DeclaredOnly);
                if (property != null) return property.GetValue(owner, null);
            }
            throw new MissingMemberException(owner.GetType().FullName, name);
        }
        public static T Get<T>(object owner, string name) { return (T)Get(owner, name); }
        public static void Set(object owner, string name, object value)
        {
            for (Type t = owner.GetType(); t != null; t = t.BaseType)
            {
                var field = t.GetField(name, Flags | BindingFlags.DeclaredOnly);
                if (field != null) { field.SetValue(owner, value); return; }
                var property = t.GetProperty(name, Flags | BindingFlags.DeclaredOnly);
                if (property != null) { property.SetValue(owner, value, null); return; }
            }
            throw new MissingMemberException(owner.GetType().FullName, name);
        }
        public static object Original(object owner, string name, Type[] parameters, params object[] arguments)
        {
            MethodInfo method = null;
            // Saved originals are private. A patched base method can be invoked on
            // a derived Scout/Build/etc. instance, so inherited lookup is insufficient.
            for (Type type = owner.GetType(); type != null && method == null; type = type.BaseType)
                method = type.GetMethod("BrassPatch_Original_" + name, Flags | BindingFlags.DeclaredOnly, null, parameters, null);
            if (method == null) throw new MissingMethodException(owner.GetType().FullName, name);
            try { return method.Invoke(owner, arguments); }
            catch (TargetInvocationException ex) { ExceptionDispatchInfo.Capture(ex.InnerException ?? ex).Throw(); throw; }
        }
    }
}
