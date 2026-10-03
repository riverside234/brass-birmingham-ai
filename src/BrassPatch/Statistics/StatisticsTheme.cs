using BoardGameRules.Entities.Players;
using UnityEngine;
namespace BrassPatch.Statistics
{
    // Opt-in on the statistics root. Shared widgets retain their legacy defaults
    // when called by Card Tracker or another information surface.
    public sealed class StatisticsTheme : MonoBehaviour
    {
        public static readonly Color Window = Hex(0xF5F1E7), Paper = Hex(0xFFFCF5), Alternate = Hex(0xEEE8DB);
        public static readonly Color Ink = Hex(0x283238), Muted = Hex(0x586269), Accent = Hex(0x8A672B);
        public static readonly Color Selected = Hex(0xE9D8B4), Border = Hex(0xC5BCA9), Grid = Hex(0xDCD6C9), Focus = Hex(0x286D83);
        public static Color Hex(int rgb) { return new Color(((rgb >> 16)&255)/255f,((rgb >> 8)&255)/255f,(rgb&255)/255f,1); }
        public static bool Light(Component component) { return component != null && component.GetComponentInParent<StatisticsTheme>() != null; }
        public static Color Player(PlayerColor player)
        {
            switch(player.ToString()) {
                case "Blue": return Hex(0x1D6F9D); case "Yellow": return Hex(0x8B6413);
                case "Purple": case "Violet": return Hex(0x77539D); case "Red": return Hex(0xB34F45);
                case "Green": return Hex(0x397652); default: return Ink;
            }
        }
        public static Color ReadableOn(Color background)
        {
            System.Func<float,double> linear = c => c <= .04045f ? c/12.92 : System.Math.Pow((c+.055)/1.055,2.4);
            double lum=.2126*linear(background.r)+.7152*linear(background.g)+.0722*linear(background.b);
            return lum > .25 ? Ink : Paper;
        }
    }
}
