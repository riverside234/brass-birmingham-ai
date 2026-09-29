using System;
using BrassApplication.UseCases.Common.SaveGame;
using InterfaceAdapters.Common.SaveGame;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils.MainThreadDispatcher;

namespace BrassPatch.Persistence
{
    public static class SaveCompletion
    {
        public static void Completed(SaveGamePresenter presenter, bool successful, ISaveGameDelegate callback)
        {
            if (successful || !Liquidation.LiquidationFixes.HandlesSaveFailure(callback))
            {
                ReflectionAccess.Original(presenter, "SaveGameCompleted", new[] { typeof(bool), typeof(ISaveGameDelegate) }, successful, callback);
                return;
            }
            // The original presenter consumes failure instead of notifying the local
            // turn coordinator, leaving its SaveInProgress flag latched forever.
            ReflectionAccess.Get<IMainThreadDispatcher>(presenter, "_dispatcher").InvokeAsync(() => {
                if (!Liquidation.LiquidationFixes.IsCurrentSave(callback)) return;
                var router = ReflectionAccess.Get<IRouter>(presenter, "_router");
                if (router.IsViewVisible<ISaveGameView>()) router.HideView<ISaveGameView>();
                callback.SaveGameComplete(false);
            });
        }
    }
}
