using System;
using System.IO;
using UnityEngine;
using BrassHost;
public static class UnityEntry
{
    public static void Run()
    {
        string root=Path.GetDirectoryName(Application.dataPath);string output=Path.Combine(root,"legacy-unity-result.json");
        try{var result=BridgeChecks.Run(root);File.WriteAllText(output,Wire.Json(new{passed=true,unity=Application.unityVersion,result}));Application.Quit(0);}
        catch(Exception e){File.WriteAllText(output,Wire.Json(new{passed=false,error=e.ToString()}));Debug.LogException(e);Application.Quit(1);}
    }
}
