using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsBoardRoute
    {
        public readonly int Era; public readonly string[] Cities;
        public StatisticsBoardRoute(int era,params string[] cities){Era=era;Cities=cities;}
    }
    public static class StatisticsBoardLayout
    {
        // Board-relative coordinates, stable across eras and report filters.
        public static readonly Dictionary<string,Vector2> Points=new Dictionary<string,Vector2>{
            {"Warrington",new Vector2(0.160f,0.955f)},
            {"StokeOnTrent",new Vector2(0.345f,0.850f)},
            {"Leek",new Vector2(0.560f,0.965f)},
            {"Belper",new Vector2(0.800f,0.950f)},
            {"Nottingham",new Vector2(0.965f,0.835f)},
            {"Stone",new Vector2(0.235f,0.735f)},
            {"Uttoxeter",new Vector2(0.560f,0.745f)},
            {"Derby",new Vector2(0.825f,0.745f)},
            {"Stafford",new Vector2(0.370f,0.625f)},
            {"BurtonOnTrent",new Vector2(0.745f,0.610f)},
            {"Central",new Vector2(0.220f,0.535f)},
            {"Cannock",new Vector2(0.470f,0.515f)},
            {"Tamworth",new Vector2(0.805f,0.465f)},
            {"Shrewsbury",new Vector2(0.045f,0.415f)},
            {"Coalbrookdale",new Vector2(0.225f,0.385f)},
            {"Wolverhampton",new Vector2(0.385f,0.415f)},
            {"Walsall",new Vector2(0.580f,0.385f)},
            {"Nuneaton",new Vector2(0.955f,0.365f)},
            {"Dudley",new Vector2(0.430f,0.275f)},
            {"Birmingham",new Vector2(0.715f,0.270f)},
            {"Coventry",new Vector2(0.955f,0.205f)},
            {"Kidderminster",new Vector2(0.320f,0.235f)},
            {"SouthWest",new Vector2(0.205f,0.140f)},
            {"Worcester",new Vector2(0.395f,0.115f)},
            {"Redditch",new Vector2(0.695f,0.115f)},
            {"Oxford",new Vector2(0.900f,0.055f)},
            {"Gloucester",new Vector2(0.550f,0.040f)},
        };
        // Explicit era topology copied from Board.CreateLinks, including its one three-endpoint link.
        public static readonly StatisticsBoardRoute[] Routes={
            new StatisticsBoardRoute(2, "Birmingham", "Walsall"),
            new StatisticsBoardRoute(1, "Birmingham", "Walsall"),
            new StatisticsBoardRoute(2, "Birmingham", "Redditch"),
            new StatisticsBoardRoute(2, "Birmingham", "Coventry"),
            new StatisticsBoardRoute(1, "Birmingham", "Coventry"),
            new StatisticsBoardRoute(2, "Birmingham", "Nuneaton"),
            new StatisticsBoardRoute(2, "Birmingham", "Tamworth"),
            new StatisticsBoardRoute(1, "Birmingham", "Tamworth"),
            new StatisticsBoardRoute(2, "Birmingham", "Dudley"),
            new StatisticsBoardRoute(1, "Birmingham", "Dudley"),
            new StatisticsBoardRoute(2, "Birmingham", "Worcester"),
            new StatisticsBoardRoute(1, "Birmingham", "Worcester"),
            new StatisticsBoardRoute(2, "Birmingham", "Oxford"),
            new StatisticsBoardRoute(1, "Birmingham", "Oxford"),
            new StatisticsBoardRoute(2, "Redditch", "Oxford"),
            new StatisticsBoardRoute(1, "Redditch", "Oxford"),
            new StatisticsBoardRoute(2, "Redditch", "Gloucester"),
            new StatisticsBoardRoute(1, "Redditch", "Gloucester"),
            new StatisticsBoardRoute(2, "Nuneaton", "Tamworth"),
            new StatisticsBoardRoute(1, "Nuneaton", "Tamworth"),
            new StatisticsBoardRoute(2, "Kidderminster", "Dudley"),
            new StatisticsBoardRoute(1, "Kidderminster", "Dudley"),
            new StatisticsBoardRoute(2, "Kidderminster", "Coalbrookdale"),
            new StatisticsBoardRoute(1, "Kidderminster", "Coalbrookdale"),
            new StatisticsBoardRoute(2, "Coalbrookdale", "Wolverhampton"),
            new StatisticsBoardRoute(1, "Coalbrookdale", "Wolverhampton"),
            new StatisticsBoardRoute(2, "Dudley", "Wolverhampton"),
            new StatisticsBoardRoute(1, "Dudley", "Wolverhampton"),
            new StatisticsBoardRoute(2, "Wolverhampton", "Cannock"),
            new StatisticsBoardRoute(1, "Wolverhampton", "Cannock"),
            new StatisticsBoardRoute(2, "Wolverhampton", "Walsall"),
            new StatisticsBoardRoute(1, "Wolverhampton", "Walsall"),
            new StatisticsBoardRoute(2, "Walsall", "Cannock"),
            new StatisticsBoardRoute(1, "Walsall", "Cannock"),
            new StatisticsBoardRoute(2, "Walsall", "Tamworth"),
            new StatisticsBoardRoute(1, "Walsall", "BurtonOnTrent"),
            new StatisticsBoardRoute(2, "Coventry", "Nuneaton"),
            new StatisticsBoardRoute(2, "Worcester", "Gloucester"),
            new StatisticsBoardRoute(1, "Worcester", "Gloucester"),
            new StatisticsBoardRoute(2, "Worcester", "SouthWest", "Kidderminster"),
            new StatisticsBoardRoute(1, "Worcester", "SouthWest", "Kidderminster"),
            new StatisticsBoardRoute(2, "Coalbrookdale", "Shrewsbury"),
            new StatisticsBoardRoute(1, "Coalbrookdale", "Shrewsbury"),
            new StatisticsBoardRoute(2, "Cannock", "Central"),
            new StatisticsBoardRoute(1, "Cannock", "Central"),
            new StatisticsBoardRoute(2, "Cannock", "Stafford"),
            new StatisticsBoardRoute(1, "Cannock", "Stafford"),
            new StatisticsBoardRoute(2, "Cannock", "BurtonOnTrent"),
            new StatisticsBoardRoute(2, "Tamworth", "BurtonOnTrent"),
            new StatisticsBoardRoute(1, "Tamworth", "BurtonOnTrent"),
            new StatisticsBoardRoute(2, "Stone", "Stafford"),
            new StatisticsBoardRoute(1, "Stone", "Stafford"),
            new StatisticsBoardRoute(2, "Stone", "Uttoxeter"),
            new StatisticsBoardRoute(2, "Stone", "BurtonOnTrent"),
            new StatisticsBoardRoute(1, "Stone", "BurtonOnTrent"),
            new StatisticsBoardRoute(2, "Stone", "StokeOnTrent"),
            new StatisticsBoardRoute(1, "Stone", "StokeOnTrent"),
            new StatisticsBoardRoute(2, "StokeOnTrent", "Warrington"),
            new StatisticsBoardRoute(1, "StokeOnTrent", "Warrington"),
            new StatisticsBoardRoute(2, "StokeOnTrent", "Leek"),
            new StatisticsBoardRoute(1, "StokeOnTrent", "Leek"),
            new StatisticsBoardRoute(2, "Leek", "Belper"),
            new StatisticsBoardRoute(2, "Belper", "Derby"),
            new StatisticsBoardRoute(1, "Belper", "Derby"),
            new StatisticsBoardRoute(2, "Derby", "Nottingham"),
            new StatisticsBoardRoute(1, "Derby", "Nottingham"),
            new StatisticsBoardRoute(2, "Derby", "Uttoxeter"),
            new StatisticsBoardRoute(2, "Derby", "BurtonOnTrent"),
            new StatisticsBoardRoute(1, "Derby", "BurtonOnTrent"),
        };
        public static string Identity(IEnumerable<string> cities){return String.Join("|",cities.OrderBy(c=>c,StringComparer.Ordinal));}
        public static Vector2[][] Paths(string[] cities)
        {
            if(cities.Any(c=>!Points.ContainsKey(c)))return new Vector2[0][];
            if(cities.Length==3){var hub=new Vector2(.315f,.135f);return cities.Select(c=>new[]{Points[c],hub}).ToArray();}
            var a=Points[cities[0]];var b=Points[cities[1]];var key=Identity(cities);
            // Route bends keep long southern corridors away from unrelated city centers.
            if(key=="Birmingham|Worcester")return new[]{new[]{a,new Vector2(.55f,.185f),b}};
            if(key=="Birmingham|Oxford")return new[]{new[]{a,new Vector2(.83f,.180f),b}};
            if(key=="Stone|BurtonOnTrent"||key=="BurtonOnTrent|Stone")return new[]{new[]{a,new Vector2(.50f,.695f),b}};
            return new[]{new[]{a,b}};
        }
    }
}
