"""Checked, detached, opt-in hooks for the versioned teacher objective.

Historical archives and the installed core are never edited. Every replacement
is anchored to the sealed core; drift fails before a binary can be published.
"""
from pathlib import Path
from brass_ai.learning.contracts import SOURCE_ROOT, require


def apply(directory):
    root=Path(directory)/'brass-ai-core/src/search'
    def edit(name,before,after,count=1):
        p=root/name;s=p.read_text(encoding='utf-8')
        require(s.count(before)==count,'DOMiNO hook drift: '+name+' '+before[:65])
        p.write_bytes(s.replace(before,after).encode())
    (root/'domino.rs').write_bytes((SOURCE_ROOT/'lab-native/core-overlay/domino.rs').read_bytes().replace(b'\r\n',b'\n'))
    edit('mod.rs','pub mod brass;','pub mod domino;\npub mod brass;')
    edit('mod.rs','    fn spent(&self, world: &Self::World) -> [f64; 4];',
        '''    fn spent(&self, world: &Self::World) -> [f64; 4];
    fn objective(&self) -> Option<&domino::Objective> { None }
    fn intrinsic_prefix(&self, _world: &Self::World) -> f64 { 0. }''')
    edit('mod.rs','    fn evaluate(&mut self, info: &Info<K, A>) -> Result<Statistics>;',
        '''    fn evaluate(&mut self, info: &Info<K, A>) -> Result<Statistics>;
    fn intrinsic(&mut self, _info: &Info<K,A>) -> Result<f64> { Err(invalid("intrinsic evaluator unavailable")) }''')
    edit('math.rs','pub struct Statistics {','pub struct Statistics {\n    #[serde(skip_serializing_if="Option::is_none")]\n    pub diversity: Option<super::domino::Channels>,')
    edit('math.rs','            count: vec![0; logits.len()],','            diversity: None,\n            count: vec![0; logits.len()],')
    # Prefix participates in semantic key equality/hash but never in actor tensors.
    edit('brass.rs','pub struct Key {','pub struct Key {\n    #[serde(default)]\n    pub intrinsic_prefix: f64,')
    edit('brass.rs','        self.settled.hash(h);','        self.settled.hash(h);\n        bits(self.intrinsic_prefix).hash(h);')
    edit('brass.rs','pub struct Brass {','pub struct Brass {\n    objective: Option<super::domino::Objective>,')
    edit('brass.rs','pub struct World {','pub struct World {\n    intrinsic_prefix: f64,')
    edit('brass.rs','            decision: None,','            intrinsic_prefix: self.objective.as_ref().map_or(0.,|o|o.prefix),\n            decision: None,',2)
    edit('brass.rs','            prepared,','            objective: None,\n            prepared,')
    edit('brass.rs','                information,','                intrinsic_prefix: 0.,\n                information,')
    edit('brass.rs','    pub fn finish(&self) -> Result<()> {','''    pub fn set_objective(&mut self, o: super::domino::Objective) -> Result<()> {
        o.validate()?;
        check(o.owner==self.root_info.actor,"objective root owner")?;
        std::sync::Arc::make_mut(&mut self.root_info.key).intrinsic_prefix=o.prefix;
        self.objective=Some(o);Ok(())
    }
    pub fn finish(&self) -> Result<()> {''')
    edit('brass.rs','        let info = Self::information(&w.state, &w.memory, &w.ledger, &self.context)?;',
        '''        let mut info = Self::information(&w.state, &w.memory, &w.ledger, &self.context)?;
        std::sync::Arc::make_mut(&mut info.key).intrinsic_prefix=w.intrinsic_prefix;''')
    edit('brass.rs','            w.ledger.chosen(view, choice);','''            if let Some(o)=&self.objective { w.intrinsic_prefix+=o.reward(view,choice)?; }
            w.ledger.chosen(view, choice);''')
    edit('brass.rs','    fn spent(&self, w: &World) -> [f64; 4] {','''    fn objective(&self)->Option<&super::domino::Objective> { self.objective.as_ref() }
    fn intrinsic_prefix(&self,w:&World)->f64 {w.intrinsic_prefix}
    fn spent(&self, w: &World) -> [f64; 4] {''')
    # One pending path owns all stochastic work; no additional tree visits.
    edit('trajectory.rs','    world: G::World,','    pub(super) world: G::World,\n    pub(super) intrinsic_steps: u64,\n    pub(super) intrinsic_terminal: bool,\n    pub(super) diversity: Option<super::domino::Completion>,\n    prefixes: Vec<f64>,')
    edit('trajectory.rs','    terminal: Option<[f64; 4]>,','    pub(super) terminal: Option<[f64; 4]>,')
    edit('trajectory.rs','    steps: usize,','    pub(super) steps: usize,')
    edit('trajectory.rs','    let mut path = Vec::new();','    let mut prefixes=Vec::new();\n    let mut path = Vec::new();')
    edit('trajectory.rs','                    world,','                    intrinsic_steps: 0,\n                    intrinsic_terminal: false,\n                    diversity: None,\n                    prefixes,\n                    world,',2)
    edit('trajectory.rs','                path.push(PathEdge {','                if game.objective().is_some() {prefixes.push(game.intrinsic_prefix(&world));}\n                path.push(PathEdge {')
    edit('trajectory.rs','            (Some(s), None) => Ok(s.value),','            (Some(s), None) => Ok(s.diversity.as_ref().map_or(s.value,|d|d.task_value)),')
    edit('trajectory.rs','        Ok(std::array::from_fn(|s| {\n            value[s] - (self.spent[s] - first.spent[s])\n        }))',
        '''        let mut result=std::array::from_fn(|s|value[s]-(self.spent[s]-first.spent[s]));
        if let Some(d)=&self.diversity {let o=&d.objective;
            result[o.owner]=o.coefficient*result[o.owner]+(1.-o.coefficient)*(d.total-self.prefixes[0]);}
        Ok(result)''')
    edit('trajectory.rs','let staged = tree.prepare_backup(&self.path, value, self.spent)?;',
        '''let staged = if let Some(d)=&self.diversity {
            tree.prepare_diversity_backup(&self.path,value,self.spent,&self.prefixes,d)?
        } else {tree.prepare_backup(&self.path, value, self.spent)?};''',2)
    edit('trajectory.rs','old.logits == stats.logits && old.value == stats.value,',
        '''old.logits == stats.logits && (if self.diversity.is_some() {
                        old.diversity.as_ref().zip(stats.diversity.as_ref()).is_some_and(|(a,b)|a.task_value==b.task_value)
                    } else {old.value == stats.value}),''')
    # Raw channel staging accompanies the existing bounded publication transaction.
    edit('tree.rs','pub(super) struct Backup {','pub(super) struct Backup {\n    diversity: BTreeMap<(NodeKey,usize),(f64,f64)>,')
    edit('tree.rs','Ok(Backup { changes, values })','Ok(Backup { changes, values, diversity:BTreeMap::new() })')
    edit('tree.rs','    pub(super) fn publish_backup(','''    pub(super) fn prepare_diversity_backup(&self,path:&[PathEdge],value:[f64;4],spent:[f64;4],
        prefixes:&[f64],d:&super::domino::Completion)->Result<Backup> {
        check(path.len()==prefixes.len(),"intrinsic prefix path length")?;
        let mut staged=self.prepare_backup(path,value,spent)?;
        for ((edge,prefix),task) in path.iter().zip(prefixes).zip(staged.values.iter_mut()) {
            let node=self.get(edge.node)?;
            let channels=node.stats.diversity.as_ref().ok_or_else(||invalid("missing channel node"))?;
            let o=&d.objective;
            check(channels.owner==o.owner && channels.coefficient==o.coefficient,"channel objective changed")?;
            let intrinsic=d.total-prefix;
            let raw=staged.diversity.entry((edge.node.key,edge.edge))
                .or_insert((channels.task_returns[edge.edge],channels.intrinsic_returns[edge.edge]));
            raw.0+=task[o.owner];raw.1+=intrinsic;
            let combined=o.coefficient*task[o.owner]+(1.-o.coefficient)*intrinsic;
            staged.changes.get_mut(&(edge.node.key,edge.edge)).unwrap().1[o.owner]+=combined-task[o.owner];
            task[o.owner]=combined;
            check(raw.0.is_finite() && raw.1.is_finite() && combined.is_finite(),"nonfinite channel backup")?;
        }
        check(staged.changes.values().all(|(_,r)|r.iter().all(|v|v.is_finite())),"combined return overflow")?;
        Ok(staged)
    }
    pub(super) fn publish_backup(''')
    edit('tree.rs','        staged.values\n    }','''        for ((key,edge),(task,intrinsic)) in staged.diversity {
            let d=self.nodes[key].stats.diversity.as_mut().unwrap();
            d.task_returns[edge]=task;d.intrinsic_returns[edge]=intrinsic;
        }
        staged.values
    }''')
    edit('tree.rs','        let result = (|| {','''        let channel_undo:Vec<_>=staged.diversity.keys().map(|(key,edge)| {
            let d=self.nodes[*key].stats.diversity.as_ref().unwrap();(*key,*edge,d.task_returns[*edge],d.intrinsic_returns[*edge])
        }).collect();
        let result = (|| {''')
    edit('tree.rs','            crate::work::checkpoint(crate::work::Stage::Publication)?;\n            check_time()','''            for ((key,edge),(task,intrinsic)) in staged.diversity {
                let d=self.nodes[key].stats.diversity.as_mut().unwrap();d.task_returns[edge]=task;d.intrinsic_returns[edge]=intrinsic;
            }
            crate::work::checkpoint(crate::work::Stage::Publication)?;
            check_time()''')
    edit('tree.rs','        if result.is_err() {','''        if result.is_err() {
            for (key,edge,task,intrinsic) in channel_undo {
                let d=self.nodes[key].stats.diversity.as_mut().unwrap();d.task_returns[edge]=task;d.intrinsic_returns[edge]=intrinsic;
            }''')
    edit('session/mod.rs','pub struct Session {','pub struct Session {\n    objective: Option<super::domino::Objective>,')
    edit('session/mod.rs','pub struct Outcome {','pub struct Outcome {\n    #[serde(skip_serializing_if="Option::is_none")]\n    pub teacher_work: Option<super::domino::Work>,')
    edit('session/mod.rs','            status: "completed".into(),','            teacher_work: self.objective.as_ref().map(|_|super::domino::Work::default()),\n            status: "completed".into(),')
    edit('session/mod.rs','            teacher: None,','            objective: None,\n            teacher: None,')
    edit('session/mod.rs','    pub fn budget(&self) -> Budget {','''    pub fn enable_diversity(&mut self,o:super::domino::Objective)->Result<()> {
        check(self.request==0 && self.teacher.is_some() && !self.retain_descendants,"diversity teacher admission")?;
        o.validate()?;self.objective=Some(o);Ok(())
    }
    pub fn set_intrinsic_prefix(&mut self,prefix:f64)->Result<()> {
        check(prefix.is_finite(),"nonfinite live prefix")?;
        self.objective.as_mut().ok_or_else(||invalid("missing objective"))?.prefix=prefix;Ok(())
    }
    pub fn budget(&self) -> Budget {''')
    edit('session/mod.rs','        let mut game = Brass::new(self.canonical_input(input)?, false)?;',
        '''        let mut game = Brass::new(self.canonical_input(input)?, false)?;
        if let Some(o)=&self.objective {game.set_objective(o.clone())?;}''')
    edit('session/mod.rs','        let mut game = Brass::new(input, false)?;',
        '''        let mut game = Brass::new(input, false)?;
        if let Some(o)=&self.objective {game.set_objective(o.clone())?;}''')
    edit('session/mod.rs',"impl<E: LeafEvaluator<Key, String>, C: Clock> LeafEvaluator<Key, String> for Cached<'_, E, C> {",'''
impl<E: LeafEvaluator<Key, String>, C: Clock> LeafEvaluator<Key, String> for Cached<'_, E, C> {
    fn intrinsic(&mut self,info:&Info<Key,String>)->Result<f64> {
        check(self.budget.work_open(self.clock),"intrinsic deadline")?;
        let v=self.evaluator.intrinsic(info)?;
        check(self.budget.work_open(self.clock) && v.is_finite(),"late/nonfinite intrinsic reply")?;Ok(v)
    }''')
    edit('session/mod.rs','        let root_stats = if info.actions.len() == 1','        let mut root_stats = if info.actions.len() == 1')
    edit('session/mod.rs','        let mut first = info.actions[0].clone();','''        if let (Some(o),Some(s))=(&self.objective,&mut root_stats) {
            let tail=if o.active() {cached.intrinsic(&info)} else {Ok(0.)};
            match tail {Ok(v)=>o.initialize(s,v)?,Err(e)=>{
                if self.budget.work_open(clock) {return Err(e);}root_stats=None;
            }}
        }
        let mut continuation_draws=Draws::new(self.settings.search.seed.wrapping_add(self.epoch)^0xd6e8feb86659fd93,None,false);
        let mut first = info.actions[0].clone();''')
    edit('session/mod.rs','                        let pending = trajectory::traverse(','                        let mut pending = trajectory::traverse(')
    edit('session/mod.rs','                    let work = (|| {','                    let work: Result<()> = (|| {')
    edit('session/mod.rs','                        let evaluation = pending','                        let mut evaluation = pending')
    edit('session/mod.rs','                        let value = pending.root_return(evaluation.as_ref())?;',
        '''                        if self.objective.is_some() {
                            let intrinsic_start=clock.now_ns();
                            let finished=pending.finish_diversity(&game,&mut cached,&mut continuation_draws,self.settings.search.max_depth,
                                &mut evaluation,&mut ||check(self.budget.work_open(clock),"continuation deadline"));
                            let work=out.teacher_work.as_mut().unwrap();
                            work.main_steps+=pending.steps() as u64;work.continuation_steps+=pending.intrinsic_steps;
                            work.intrinsic_work_ns+=clock.now_ns().saturating_sub(intrinsic_start);
                            if finished.is_ok() {
                                if pending.intrinsic_terminal || pending.terminal.is_some() {work.terminal_bootstraps+=1;}
                                else if self.objective.as_ref().unwrap().active() {work.root_bootstraps+=1;}
                            }
                            finished?;
                        }
                        let value = pending.root_return(evaluation.as_ref())?;''')
    edit('session/mod.rs','                        pending.complete(&mut tree, evaluation, false)?;',
        '''                        if self.objective.is_some() {
                            pending.complete_checked(&mut tree,evaluation,&mut ||check(self.budget.work_open(clock),"channel publication deadline"))?;
                        } else {pending.complete(&mut tree, evaluation, false)?;}''')
    edit('session/mod.rs','                        if self.budget.work_open(clock) {\n                            return Err(e);\n                        }',
        '''                        if let Some(w)=&mut out.teacher_work {w.aborted_trajectories+=1;}
                        if self.objective.is_some() && e.to_string().contains("DOMiNO continuation depth limit") {
                            out.status="diversity_depth_limit".into();break;
                        }
                        if self.budget.work_open(clock) {
                            return Err(e);
                        }''')
