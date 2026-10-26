//! Exact finite HiddenOffer fixture; never a substitute for Brass qualification.
use super::*;
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub private: Vec<i32>,
    pub public: Vec<i32>,
    pub player: usize,
}
pub struct World {
    types: Vec<i32>,
    moves: Vec<i32>,
    shock: Option<i32>,
}
pub struct HiddenOffer {
    pub players: usize,
}
impl Game for HiddenOffer {
    type Key = Key;
    type Action = i32;
    type World = World;
    fn root(&mut self) -> Result<Info<Key, i32>> {
        check([2, 4].contains(&self.players), "toy player count")?;
        Ok(Info {
            key: std::sync::Arc::new(Key {
                private: vec![0],
                public: vec![1],
                player: 0,
            }),
            actor: 0,
            actions: (0..16).collect::<Vec<_>>().into(),
        })
    }
    fn sample(&mut self, d: &mut Draws) -> Result<World> {
        let mut types = vec![0];
        for i in 1..self.players {
            types.push(i32::from(d.uniform()? < if i == 1 { 0.8 } else { 0.5 }));
        }
        Ok(World {
            types,
            moves: vec![],
            shock: None,
        })
    }
    fn inspect(&self, w: &mut World) -> Result<Position<Key, i32>> {
        if let Some(shock) = w.shock {
            let price = (w.moves[0] + 1) as f64 / 17.;
            let mut v = [0.; 4];
            v[0] = w.moves[1..]
                .iter()
                .map(|a| {
                    if *a != 0 {
                        price + 0.02 * shock as f64
                    } else {
                        -0.05
                    }
                })
                .sum::<f64>()
                / (self.players - 1) as f64;
            for (i, a) in w.moves.iter().enumerate().skip(1) {
                v[i] = if *a != 0 {
                    0.4 + 0.5 * w.types[i] as f64 - price + 0.01 * shock as f64
                } else {
                    0.01
                };
            }
            return Ok(Position::Terminal(v));
        }
        let actor = w.moves.len();
        if actor == self.players {
            return Ok(Position::Chance(vec![(-1, 0.25), (1, 0.75)]));
        }
        let mut public = vec![1];
        public.extend(&w.moves);
        Ok(Position::Decision(Info {
            key: std::sync::Arc::new(Key {
                private: vec![w.types[actor]],
                public,
                player: actor,
            }),
            actor,
            actions: if actor == 0 {
                (0..16).collect::<Vec<_>>().into()
            } else {
                vec![0, 1].into()
            },
        }))
    }
    fn step(&self, w: &mut World, a: &i32) -> Result<()> {
        if w.moves.len() == self.players {
            w.shock = Some(*a);
        } else {
            w.moves.push(*a);
        }
        Ok(())
    }
    fn spent(&self, _: &World) -> [f64; 4] {
        [0.; 4]
    }
}
pub struct Evaluator;
impl LeafEvaluator<Key, i32> for Evaluator {
    fn evaluate(&mut self, i: &Info<Key, i32>) -> Result<Statistics> {
        let logits = if i.actor == 0 {
            i.actions
                .iter()
                .map(|a| -((*a - 7).abs() as f64) / 4.)
                .collect()
        } else {
            vec![
                0.,
                5. * (0.4 + 0.5 * i.key.private[0] as f64 - (i.key.public[1] + 1) as f64 / 17.),
            ]
        };
        Statistics::new(logits, [0.; 4])
    }
}
