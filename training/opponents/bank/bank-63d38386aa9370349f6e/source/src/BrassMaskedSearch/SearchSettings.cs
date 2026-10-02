using System;
using System.Linq;
namespace BrassLegacyAI
{
    // Search evidence is versioned independently of the rules and model weights.
    public sealed partial class Profile
    {
        public string search_contract="masked-ismcts/1.1.0",evaluator_kind="heuristic",
            q_transform="regularized_sibling",leaf_policy="settled_reply_prefix",
            candidate_policy="supported_progressive",selection_policy="puct";
        public int opponent_turn_limit=0,minimum_plan_visits=2,reopen_support=4;
        public double fpu_reduction=.02,q_min_span=.05,tail_admission=.05;
        public bool incumbent_only=false;
        public static void ValidateSearch(Profile p)
        {
            if(p.search_contract!="masked-ismcts/1.1.0" || p.evaluator_kind!="heuristic" ||
                !new[]{"regularized_sibling","none"}.Contains(p.q_transform) ||
                !new[]{"settled_reply_prefix","actual_next_root"}.Contains(p.leaf_policy) ||
                p.candidate_policy!="supported_progressive" || !new[]{"puct","unseen_first"}.Contains(p.selection_policy) ||
                p.opponent_turn_limit<0 || p.opponent_turn_limit>6 || p.minimum_plan_visits<2 || p.reopen_support<1 ||
                new[]{p.fpu_reduction,p.q_min_span,p.tail_admission}.Any(x=>double.IsNaN(x) || double.IsInfinity(x)) ||
                p.fpu_reduction<0 || p.q_min_span<=0 || p.q_min_span>1 || p.tail_admission<0 || p.tail_admission>0.5)
                throw new Exception("heuristic_search_contract");
        }
    }
    public static class SearchSelection
    {
        public static double Mean(SearchEdge e,int seat,double fallback)
        {return e!=null && e.visits>0?e.returns[seat]/e.visits:fallback;}
        public static double Transform(double q,double lower,double upper,double span)
        {return Math.Max(0,Math.Min(1,(q-lower)/Math.Max(span,upper-lower)));}
    }
}
