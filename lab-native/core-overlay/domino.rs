//! Opt-in lab objective. Pure NN values and stochastic continuation stay separate.
use super::{brass::{Brass,Key}, math::{self,Statistics}, trajectory::Pending, *};
use crate::{state::{dto::{Choice,Industry,Kind},observation::Observation},types::{Era,Phase}};
use serde::{Deserialize,Serialize};
use std::collections::BTreeSet;

pub const ALGORITHM: &str = "brass-gumbel-is-mcts/domino-count-v1";
pub const SESSION: &str = "brass-gumbel-is-mcts/domino-session-v1";
pub const COUNTS: [f64;4] = [7.,8.,12.,4.];

#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    pub owner: usize,
    pub strategy: usize,
    pub coefficient: f64,
    pub direction: Vec<f64>,
    pub bundle_id: String,
    pub slots: Vec<String>,
    pub regions: Vec<String>,
    pub prefix: f64,
}
impl Objective {
    pub fn validate(&self)->Result<()> {
        check(self.owner<4 && self.strategy<7 && self.coefficient.is_finite()
            && self.coefficient>0. && self.coefficient<=1. && self.prefix.is_finite()
            && self.direction.len()==448 && self.direction.iter().all(|x|x.is_finite())
            && self.bundle_id.len()==64 && 1+self.slots.len()+self.regions.len()==72,
            "invalid DOMiNO objective")?;
        check(self.strategy!=0 || (self.coefficient==1. && self.direction.iter().all(|x|*x==0.)),"neutral objective")
    }
    pub fn active(&self)->bool {self.strategy!=0 && self.coefficient!=1.}
    pub fn reward(&self,view:&Observation,choice:&Choice)->Result<f64> {
        if !self.active() || usize::from(u8::from(view.seat))!=self.owner || view.context.phase!=Phase::Action {return Ok(0.);}
        let (phase,x)=event(view,choice,&self.slots,&self.regions)?;
        Ok(x.iter().zip(&self.direction[phase*112..(phase+1)*112]).map(|(a,b)|a*b).sum::<f64>()/(4.*COUNTS[phase]))
    }
    pub fn initialize(&self,stats:&mut Statistics,tail:f64)->Result<()> {
        check(tail.is_finite(),"nonfinite intrinsic tail")?;
        let task=stats.value;
        stats.value[self.owner]=self.coefficient*task[self.owner]+(1.-self.coefficient)*tail;
        stats.diversity=Some(Channels{owner:self.owner,coefficient:self.coefficient,task_value:task,
            intrinsic_value:tail,task_returns:vec![0.;stats.count.len()],intrinsic_returns:vec![0.;stats.count.len()]});
        Ok(())
    }
}

#[derive(Debug,Clone,Serialize)]
pub struct Channels {
    pub owner:usize,
    pub coefficient:f64,
    pub task_value:[f64;4],
    pub intrinsic_value:f64,
    pub task_returns:Vec<f64>,
    pub intrinsic_returns:Vec<f64>,
}

/// Exact Python domino_features.event family order/scales, permitted predecision only.
pub fn event(v:&Observation,c:&Choice,slots:&[String],regions:&[String])->Result<(usize,Vec<f64>)> {
    check(v.context.phase==Phase::Action && (1..=8).contains(&v.context.round),"DOMiNO main action/round")?;
    let phase=match (v.context.era,v.context.round) {(Era::CanalEra,1..=4)=>0,(Era::CanalEra,_)=>1,(_,1..=6)=>2,_=>3};
    let families=[Kind::Build,Kind::Develop,Kind::Sell,Kind::Loan,Kind::Network,Kind::Scout,Kind::Pass];
    check(families.contains(&c.kind),"DOMiNO action family")?;
    let industries=[Industry::CottonMill,Industry::Manufacturer,Industry::Pottery,Industry::Brewery,Industry::IronWorks,Industry::CoalMine];
    let p=&v.players[usize::from(u8::from(v.seat))];
    check(p.seat==v.seat,"DOMiNO observer")?;
    let own:Vec<_>=v.slots.iter().filter(|s|s.tile.as_ref().is_some_and(|t|t.seat==v.seat)).collect();
    let tiles:Vec<_>=own.iter().map(|s|s.tile.as_ref().unwrap()).collect();
    let links:Vec<_>=v.links.iter().filter(|l|l.owner==Some(v.seat)).collect();
    let mut footprint:BTreeSet<&str>=own.iter().map(|s|s.id.split(':').next().unwrap()).collect();
    for l in &links {for r in &l.regions {footprint.insert(r);}}
    let mut groups=vec![families.iter().map(|k|f64::from(*k==c.kind)).collect::<Vec<_>>()];
    let mut investment=vec![];
    for i in industries {investment.push(tiles.iter().filter(|t|t.industry==i).count() as f64/8.);}
    for i in industries {investment.push(tiles.iter().filter(|t|t.industry==i).map(|t|f64::from(t.level)).sum::<f64>()/32.);}
    for i in industries {let m=p.mat.iter().find(|m|m.industry==i).ok_or_else(||invalid("DOMiNO mat"))?;
        investment.push(m.tiles.first().map_or(8.,|t|f64::from(t.level))/8.);}
    groups.push(investment);
    groups.push([true,false].into_iter().flat_map(|flipped|industries.map(|i|
        tiles.iter().filter(|t|t.industry==i && t.flipped==flipped).count() as f64/8.)).collect());
    groups.push(vec![f64::from(p.cash)/100.,f64::from(p.income)/30.,f64::from(p.spent)/100.]);
    let mut network=vec![links.len() as f64/16.];
    network.extend(regions.iter().map(|r|f64::from(footprint.contains(r.as_str()))));
    network.extend(slots.iter().map(|s|f64::from(own.iter().any(|o|o.id==*s))));
    groups.push(network);
    let mut x=vec![];
    for group in groups {let scale=(5.*group.len() as f64).sqrt();x.extend(group.into_iter().map(|a|a.clamp(-1.,1.)/scale));}
    check(x.len()==112,"DOMiNO descriptor width")?;
    Ok((phase,x))
}

pub struct Completion {pub objective:Objective,pub total:f64}

#[derive(Default,Debug,Clone,Serialize)]
pub struct Work {
    pub main_steps:u64,
    pub continuation_steps:u64,
    pub aborted_trajectories:u64,
    pub root_bootstraps:u64,
    pub terminal_bootstraps:u64,
    pub intrinsic_work_ns:u64,
}

impl Pending<Brass> {
    /// Consume only this sampled world. No tree mutation before every component exists.
    pub fn finish_diversity<E:LeafEvaluator<Key,String>>(&mut self,game:&Brass,evaluator:&mut E,
        draws:&mut Draws,max_depth:usize,evaluation:&mut Option<Statistics>,check_time:&mut impl FnMut()->Result<()>)->Result<()> {
        let Some(o)=game.objective().cloned() else {return Ok(());};
        let leaf_prefix=game.intrinsic_prefix(&self.world);
        let tail=if !o.active() || self.terminal.is_some() {0.} else {
            let mut steps=self.steps;
            loop {
                check_time()?;
                crate::work::checkpoint(crate::work::Stage::Traversal)?;
                match game.inspect(&mut self.world)? {
                    Position::Terminal(_)=>{self.intrinsic_terminal=true;break game.intrinsic_prefix(&self.world)-leaf_prefix;},
                    Position::Decision(info) if info.actor==o.owner => {
                        let value=evaluator.intrinsic(&info)?;
                        break game.intrinsic_prefix(&self.world)-leaf_prefix+value;
                    },
                    position=>{
                        check(steps<max_depth,"DOMiNO continuation depth limit")?;
                        let action=match position {
                            Position::Decision(info)=>{
                                let stats=evaluator.evaluate(&info)?;
                                draws.categorical(&info.actions.iter().cloned().zip(math::softmax(&stats.logits)?).collect::<Vec<_>>())?
                            },
                            Position::Chance(outcomes)=>draws.categorical(&outcomes)?,
                            Position::Terminal(_)=>unreachable!(),
                        };
                        game.step(&mut self.world,&action)?;steps+=1;self.intrinsic_steps+=1;
                    }
                }
            }
        };
        check_time()?;
        if let Some(s)=evaluation {o.initialize(s,tail)?;}
        self.diversity=Some(Completion{objective:o,total:leaf_prefix+tail});
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_root_only_mix_and_neutral() {
        let mut s=Statistics::new(vec![0.,1.],[0.2,0.4,0.6,0.8]).unwrap();
        let o=Objective{owner:2,strategy:3,coefficient:0.25,direction:vec![0.;448],bundle_id:"a".repeat(64),
            slots:vec![],regions:vec!["r".into();71],prefix:0.};
        o.validate().unwrap();o.initialize(&mut s,-2.).unwrap();
        assert_eq!(s.value,[0.2,0.4,-1.35,0.8]);
        assert_eq!(s.diversity.unwrap().task_value,[0.2,0.4,0.6,0.8]);
    }
    #[test]
    fn suffix_costs_and_atomic_three_channel_publication() {
        use super::super::tree::{Tree,PathEdge};
        let o=Objective{owner:2,strategy:1,coefficient:0.25,direction:vec![0.;448],bundle_id:"a".repeat(64),
            slots:vec![],regions:vec!["r".into();71],prefix:0.};
        let mut tree=Tree::<u64,u8>::new(10,10).unwrap();
        let mut handles=vec![];
        for key in 0..2 {
            let mut s=Statistics::new(vec![0.,0.],[0.1,0.2,0.8,0.4]).unwrap();o.initialize(&mut s,0.5).unwrap();
            handles.push(tree.insert(Info{key:std::sync::Arc::new(key),actor:2,actions:std::sync::Arc::new(vec![0,1])},s).unwrap());
        }
        let path=vec![PathEdge{node:handles[0],edge:0,spent:[0.,0.,0.01,0.]},
                      PathEdge{node:handles[1],edge:1,spent:[0.,0.,0.03,0.]}];
        let d=Completion{objective:o,total:0.6};
        let stage=||tree.prepare_diversity_backup(&path,[0.1,0.2,0.8,0.4],[0.,0.,0.05,0.],&[0.1,0.4],&d).unwrap();
        let staged=stage();let mut checks=0;
        assert!(tree.commit_checked(staged,None,&mut ||{checks+=1;check(checks<2,"cancel")}).is_err());
        for h in &handles {let s=&tree.get(*h).unwrap().stats;assert_eq!(s.count,vec![0,0]);assert_eq!(s.diversity.as_ref().unwrap().task_returns,vec![0.,0.]);}
        let staged=tree.prepare_diversity_backup(&path,[0.1,0.2,0.8,0.4],[0.,0.,0.05,0.],&[0.1,0.4],&d).unwrap();
        tree.commit_checked(staged,None,&mut ||Ok(())).unwrap();
        let a=&tree.get(handles[0]).unwrap().stats;let b=&tree.get(handles[1]).unwrap().stats;
        assert!((a.returns[0][2]-(0.25*0.76+0.75*0.5)).abs()<1e-12);
        assert!((b.returns[1][2]-(0.25*0.78+0.75*0.2)).abs()<1e-12);
        assert_eq!(a.returns[0][1],0.2);
        assert!((b.diversity.as_ref().unwrap().intrinsic_returns[1]-0.2).abs()<1e-12);
    }
}
