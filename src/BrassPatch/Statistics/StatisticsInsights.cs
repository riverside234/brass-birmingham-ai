using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Players;

namespace BrassPatch.Statistics
{
    public static class StatisticsInsights
    {
        public static List<StatisticsObservation> Build(CompletedMatchStatistics r,PlayerColor p)
        {
            var result=new List<StatisticsObservation>();if(!r.CompleteHistory)return result;
            var assets=r.Assets.Where(a=>a.Player==p).ToArray();var actions=r.Actions.Where(a=>a.Player==p).ToArray();
            Action<string,string,string,string,bool,int,IEnumerable<string>> add=(title,fact,interpretation,experiment,strength,page,keys)=>{
                var drill=new StatisticsDrill{Page=page,Player=p,Era=0};foreach(var key in keys)drill.Keys.Add(key);
                if(title=="Resources helped other players"){drill.Outgoing=true;drill.Resource=r.Entries.Where(e=>e.Kind=="Resource use"&&e.Supplier==p&&e.Player!=p).GroupBy(e=>e.Resource).OrderByDescending(g=>g.Sum(e=>e.Quantity)).First().Key;}
                if(title=="Repeated market purchases")drill.Resource=r.Entries.Where(e=>e.Player==p&&e.Kind=="Market purchase").GroupBy(e=>e.Resource).OrderByDescending(g=>g.Sum(e=>e.TotalPrice)).First().Key;
                result.Add(new StatisticsObservation{Id=title.ToLowerInvariant().Replace(" ","-"),Player=p,SampleSize=drill.Keys.Count,Coverage=r.Availability,Title=title,Fact=fact,Interpretation=interpretation,Experiment=experiment,Strength=strength,Evidence=drill});
            };
            var finished=assets.Where(a=>a.Flipped).ToArray();var goods=assets.Where(a=>a.Sellable&&a.Remaining&&!a.Flipped).ToArray();
            if(assets.Length>=4&&finished.Length>=assets.Length*.8)
                add("Strong industry conversion",finished.Length+" of "+assets.Length+" built tiles flipped ("+StatisticsFormat.Percent(finished.Length,assets.Length)+").","Most construction reached its income-producing side. Scoring still depends on era boundaries.","Compare the Industries and Scoring pages before repeating the same build order.",true,2,finished.Select(a=>a.Key));
            if(goods.Length>0)
                add("Unsold goods at the finish",goods.Length+" goods tiles remained unflipped; construction and bought resources cost "+StatisticsFormat.Money(goods.Sum(a=>a.BaseCost+a.MarketCost))+".","Their printed points were not awarded. This does not prove a sale was legal or that building them was a mistake.","Before a late goods build, check the card, merchant connection and beer needed to sell it.",false,2,goods.Select(a=>a.Key));
            var suppliers=assets.Where(a=>!a.Sellable&&a.Produced>0).ToArray();int shared=r.Entries.Where(e=>e.Kind=="Resource use"&&e.Supplier==p&&e.Player!=p).Sum(e=>e.Quantity);
            if(shared>0)add("Resources helped other players",shared+" units from your suppliers were consumed by opponents.","Shared supply can flip your industry and increase income; it can also enable an opponent's score.","Compare the timing of those flips with the outgoing resource matrix before changing supplier placement.",true,5,suppliers.Select(a=>a.Key));
            var routes=r.Links.Where(l=>l.Player==p).ToArray();double median=StatisticsDetailMetrics.Median(r.Links.Select(l=>l.VP));
            var productive=routes.Where(l=>l.VP>median).OrderByDescending(l=>l.VP).Take(3).ToArray();
            if(productive.Length>=2)add("High-scoring routes",productive.Length+" routes contributed "+productive.Sum(l=>l.VP)+" VP, above the median "+median.ToString("0.#")+" VP among routes in this match.","This compares actual link scores in the same game, including own, opponent and merchant contributions. It does not establish a universal target.","Review which adjacent flipped tiles and merchants supported those routes before repeating their placement timing.",true,3,productive.Select(l=>l.Key));
            var sales=actions.Where(a=>a.Kind=="Sell").ToArray();int batches=sales.Count(a=>a.Goods>=2);
            if(batches>0)add("Efficient sale batching",batches+" of "+sales.Length+" Sell actions flipped at least two goods tiles.","Batching saved actions compared with selling each tile separately, without assigning the same VP to both Build and Sell.","Look for another connected group that could share a Sell action, while budgeting beer for each tile.",true,6,sales.Where(a=>a.Goods>=2).Select(a=>a.Key));
            else if(sales.Length>=3)add("Sales used separate actions",sales.Length+" Sell actions sold "+sales.Sum(a=>a.Goods)+" goods tiles; none sold two at once.","Separate sales may have been necessary for income, beer or competition. The report cannot prove a better legal batch existed.","When planning two goods builds, compare a shared merchant route and the timing of beer supply.",false,6,sales.Select(a=>a.Key));
            var wiped=assets.Where(a=>a.Removal=="Canal cleanup"&&!a.Flipped).ToArray();
            if(wiped.Length>0)add("Canal tiles left unfinished",wiped.Length+" unflipped tiles were removed at Canal cleanup.","Their construction did not produce industry VP. Suppliers may still have supported useful builds; printed VP is only a nominal opportunity.","In the last two Canal turns, compare finishing a tile with preparing a level-II carryover industry.",false,8,wiped.Select(a=>a.Key));
            if(!r.Introductory){
                var blocked=r.MatTiles.Where(m=>m.Player==p&&m.Boundary=="Rail entry"&&m.RailBlocked).ToArray();
                if(blocked.Length>0)add("Rail mat still had Canal-only tiles",String.Join(", ",blocked.Select(m=>StatisticsMetrics.IndustryName(m.Type)+" L"+m.Level))+" blocked construction at Rail entry.","A restriction on the actual next tile determines readiness; not every level-I tile is Rail-blocked.","Before the era changes, reserve Develop capacity for an industry you intend to build early in Rail.",false,8,new string[0]);
                var carried=assets.Where(a=>a.Carried&&a.CanalVP>0&&a.RailVP>0).ToArray();
                if(carried.Length>0)add("Carryover scored in both eras",carried.Length+" surviving tiles scored "+carried.Sum(a=>a.CanalVP)+" Canal VP and "+carried.Sum(a=>a.RailVP)+" Rail VP.","These tiles converted one construction into two scoring opportunities.","Compare their build and flip timing with the Rail-entry resource snapshot.",true,8,carried.Select(a=>a.Key));
            }
            int penalties=-StatisticsMetrics.Score(r,p,"Penalty");
            if(penalties>0)add("Debt cost victory points",penalties+" VP were deducted in recorded penalty events.","This is a realized score cost. The report does not treat ordinary loans as a tactical error.","Before spending the last cash in a round, check the next income payment and any required debt settlement.",false,4,r.Entries.Where(e=>e.Player==p&&e.Kind=="Balance"&&e.Category=="Penalty").Select(e=>"entry-"+e.Revision+"-"+e.Player));
            var purchases=r.Entries.Where(e=>e.Player==p&&e.Kind=="Market purchase").ToArray();
            if(purchases.Length>=3)add("Repeated market purchases",purchases.Length+" purchases supplied "+purchases.Sum(e=>e.Quantity)+" units for "+StatisticsFormat.Money(purchases.Sum(e=>e.TotalPrice))+" (already inside action spending).","Market buying may be the best way to finish a valuable action. The report does not assume a cheaper reachable supplier existed.","Compare one high-price purchase with the available supply timing in the resource and cash views.",false,5,purchases.Select(e=>"entry-"+e.Revision+"-"+e.Player));
            var unused=r.Entries.Where(e=>e.Player==p&&e.Kind=="Wildcard acquired"&&!r.Entries.Any(u=>u.Player==p&&u.Kind=="Card"&&u.AssetKey=="wild@"+e.Revision)).ToArray();
            if(unused.Length>0)add("Wild flexibility remained unused",unused.Length+" acquired wildcard episodes had no recorded payment before the finish.","Holding a wildcard may protect options. Its non-use does not prove that Scout was avoidable or that a strong placement existed.","Compare acquisition timing with the remaining own turns before choosing a similar late Scout action.",false,6,unused.Select(e=>"entry-"+e.Revision+"-"+e.Player));
            var slow=assets.Where(a=>a.Flipped&&a.FlipOwnTurn-a.OwnTurn>=4).ToArray();
            if(slow.Length>=2)add("Income arrived after several own turns",slow.Length+" tiles took at least four of your turns to flip.","Later flips leave fewer paydays for that income gain. Deliberate Rail positioning or scarce beer may justify the wait.","Compare one earlier sale with another build, using the income curve and actual payments as evidence.",false,2,slow.Select(a=>a.Key));
            var player=r.Players.First(x=>x.Color==p);
            if(player.FinalCash>=30)add("Review the final cash reserve",StatisticsFormat.Money(player.FinalCash)+" remained at the finish.",r.Introductory?"Introductory cash contributes to its additional scoring as well as tie-breaks. A reserve is not automatically waste.":"Cash is a tie-breaker and can reflect prudent planning. The report cannot identify an available legal VP purchase.","On the final two turns, compare your legal scoring options against the cash needed for them.",false,4,actions.Reverse().Take(4).Select(a=>a.Key));
            return result;
        }
    }
}
