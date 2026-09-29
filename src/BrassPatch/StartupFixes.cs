using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Frameworks.Init;
using UnityEngine;

namespace BrassPatch
{
    public static class StartupFixes
    {
        private static bool selfTestStarted;
        public static bool IsSelfTest { get { return Environment.GetCommandLineArgs().Any(arg => arg == "--brass-patch-selftest" || arg == "--brass-city-locator-test" || arg == "--brass-city-locator-render" || arg == "--brass-local-ai-test"); } }
        public static void Awake(InitUnity owner)
        {
            if (!IsSelfTest) { ReflectionAccess.Original(owner, "Awake", Type.EmptyTypes); return; }
            if (selfTestStarted) return;
            selfTestStarted = true;
            string gameRoot = Path.GetDirectoryName(Application.dataPath);
            string output = Path.Combine(gameRoot, "artifacts", "patch-build", "unity-selftest-result.json");
            try
            {
                string tests = Path.Combine(gameRoot, "artifacts", "patch-build", "Managed");
                if (Environment.GetCommandLineArgs().Contains("--brass-local-ai-test"))
                {
                    Assembly.LoadFrom(Path.Combine(tests, "LocalAiUnityTests.dll")).GetType("LocalAiUnityTests").GetMethod("Begin").Invoke(null, new object[] { Path.Combine(gameRoot, "artifacts", "patch-build", "local-ai-native.json") });
                    return;
                }
                if (Environment.GetCommandLineArgs().Contains("--brass-city-locator-render") || Environment.GetCommandLineArgs().Contains("--brass-city-locator-test"))
                {
                    StartCityLocator(tests, output, Newtonsoft.Json.JsonConvert.SerializeObject(new { passed = true, unityVersion = Application.unityVersion }));
                    return;
                }
                var core = Assembly.LoadFrom(Path.Combine(tests, "CoreRegression.exe"));
                int coreResult = (int)core.GetType("CoreRegression").GetMethod("Main").Invoke(null, new object[] { new[] { "unity" } });
                if (coreResult != 0) throw new InvalidOperationException("Core regression failed in Unity Mono. Inspect the Unity test log.");
                var state = Assembly.LoadFrom(Path.Combine(tests, "StateRegression.exe"));
                int stateResult = (int)state.GetType("StateRegression").GetMethod("Main").Invoke(null, new object[] { new string[0] });
                if (stateResult != 0) throw new InvalidOperationException("State regression failed in Unity Mono. Inspect the Unity test log.");
                var ai = Assembly.LoadFrom(Path.Combine(tests, "AiRegression.exe"));
                int aiChecks = (int)ai.GetType("AiRegression").GetMethod("Run").Invoke(null, null);
                var presentation = Assembly.LoadFrom(Path.Combine(tests, "UnityPresentationRegression.dll"));
                int popupChecks = (int)presentation.GetType("BrassPatch.Tests.Liquidation.UnityPresentationRegression").GetMethod("Run").Invoke(null, null);
                var persistence = Assembly.LoadFrom(Path.Combine(tests, "Persistence.Tests.dll"));
                persistence.GetType("PersistenceTests").GetMethod("Patched").Invoke(null, new object[] { Path.Combine(gameRoot, "artifacts", "patch-build", "unity-save-fixtures") });
                var resources = Assembly.LoadFrom(Path.Combine(tests, "ResourcesRegression.Tests.dll"));
                resources.GetType("ResourcesRegression").GetMethod("Run").Invoke(null, null);
                var liquidation = Assembly.LoadFrom(Path.Combine(tests, "LiquidationTests.dll"));
                int liquidationChecks = (int)liquidation.GetType("BrassPatch.Tests.Liquidation.LiquidationTests").GetMethod("Run").Invoke(null, null);
                var scout = Assembly.LoadFrom(Path.Combine(tests, "ScoutTests.dll"));
                int scoutChecks = (int)scout.GetType("BrassPatch.Tests.Scout.ScoutTests").GetMethod("Run").Invoke(null, null);
                var scoutPresentation = Assembly.LoadFrom(Path.Combine(tests, "UnityScoutRegression.dll"));
                int scoutPresentationChecks = (int)scoutPresentation.GetType("BrassPatch.Tests.Scout.UnityScoutRegression").GetMethod("Run").Invoke(null, null);
                StartCityLocator(tests, output, Newtonsoft.Json.JsonConvert.SerializeObject(new { passed = true, unityVersion = Application.unityVersion,
                    runtime = Environment.Version.ToString(), persistentDataPath = Application.persistentDataPath, popupChecks = popupChecks, liquidationChecks = liquidationChecks,
                    aiChecks = aiChecks, scoutChecks = scoutChecks, scoutPresentationChecks = scoutPresentationChecks,
                    suites = new[] { "Core", "State", "AI", "ActualPopup", "Persistence", "Resources", "Liquidation", "Scout", "ScoutPresentation", "CityLocator" } }));
            }
            catch (Exception error)
            {
                File.WriteAllText(output, Newtonsoft.Json.JsonConvert.SerializeObject(new { passed = false, persistentDataPath = Application.persistentDataPath, error = error.ToString() }));
                Debug.LogException(error);
                Application.Quit(1);
            }
        }
        private static void StartCityLocator(string tests, string output, string result)
        {
            Assembly.LoadFrom(Path.Combine(tests, "UnityCityLocatorRegression.dll"))
                .GetType("BrassPatch.Tests.CityLocator.UnityCityLocatorRegression").GetMethod("Begin").Invoke(null, new object[] { output, result });
        }
        public static void OnDestroy(InitUnity owner)
        {
            if (!IsSelfTest) ReflectionAccess.Original(owner, "OnDestroy", Type.EmptyTypes);
        }
    }
}
