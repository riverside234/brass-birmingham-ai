using BrassApplication.UseCases.Menu.MainMenu;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;

namespace BrassPatch
{
    public static class OfflineMode
    {
        public static void ShowOnlineDisabled(MainMenuUseCase menu)
        {
            var presenter = ReflectionAccess.Get<object>(menu, "_presenter");
            var router = ReflectionAccess.Get<IRouter>(presenter, "_router");
            ShowNotice(router);
        }

        public static void ShowNotice(IRouter router)
        {
            router.ShowView<IInfoPopupView>();
            router.GetView<IInfoPopupView>().Setup(
                "Online play is disabled while the local rules patch is installed. Restore the original game files to play online. Games created with this patch require the patch to continue.",
                "Close", () => router.HideView<IInfoPopupView>());
        }
    }
}
