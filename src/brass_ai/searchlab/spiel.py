"""Four-player OpenSpiel adapter over the isolated explicit-chance R1 extension.

States are trusted simulator objects, never policy inputs. Private callbacks are
not broadcast. Clones own detached reconstruction records, with one locked host
context restored on access; no game-object shallow copy is used.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import threading

import pyspiel

from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import LAB_ROOT, digest, read_json, require

TYPE = pyspiel.GameType(short_name='python_brass_four', long_name='Brass four-player Masked ISMCTS adapter',
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.CONSTANT_SUM, reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    min_num_players=4, max_num_players=4, provides_information_state_string=True,
    provides_information_state_tensor=False, provides_observation_string=True,
    provides_observation_tensor=False, parameter_specification={'root': str(LAB_ROOT), 'mono': ''})
INFO = pyspiel.GameInfo(num_distinct_actions=2**31-1, max_chance_outcomes=64,
    num_players=4, min_utility=0., max_utility=1., utility_sum=1., max_game_length=5000)


def action_id(semantic):
    return int.from_bytes(hashlib.sha256(semantic.encode()).digest()[:4], 'big') % (2**31-1)


class BrassSpielGame(pyspiel.Game):
    def __init__(self, params=None, *, root=None, mono=None, fast=False):
        params = params or {}
        super().__init__(TYPE, INFO, params)
        self.root = Path(root or params.get('root', LAB_ROOT)).resolve()
        self.mono = mono or params.get('mono') or None
        directory = self.root / 'artifacts/r4-host'
        require(read_json(directory/'runtime.json')['extension'] == 'brass-search-chance/1.0.0', 'Import the isolated search runtime first')
        if fast:
            from brass_ai.recovery.host_runtime import training_host
            directory=training_host(self.root,'search')
        self.host = RulesHost(directory, mono=self.mono)
        try:
            self.lock, self.owner = threading.RLock(), None
            self.composition = read_json(self.root/'configs/contracts/v0.2.0/card-composition.json')
            self.host.reset(1)
            if fast:self.host.enable_fastpath()
            template = self.host.call('checkpoint')['setup']
            # Only public board slots and the full merchant multiset, not a sampled deal.
            self.merchant_slots = [m['id'] for m in template['merchants']]
            self.merchant_pool = Counter(json.dumps(m.get('types') or []) for m in template['merchants'])
            require(len(self.merchant_slots)==9 and sum(self.merchant_pool.values())==9, 'Four-player merchant inventory changed')
        except BaseException:
            self.host.close()
            raise

    def close(self): self.host.close()

    @contextmanager
    def access(self, state):
        with self.lock:
            if self.owner is not state:
                state.context = self.host.call('restore', checkpoint=state.checkpoint)
                self.owner = state
            yield self.host

    def new_initial_state(self): return BrassSpielState(self)

    def from_observation(self, view, seed, recall=None):
        state = BrassSpielState(self)
        state.setup_stage = None
        with self.lock:
            state.context = self.host.reset(1, sample_world_setup(view, self.composition, seed))
            self.owner = state
            state.refresh(self.host)
        if recall is not None:
            state.recall[view['seat']] = copy.deepcopy(recall)
        return state

    def deserialize_state(self, data):
        value = json.loads(data)
        require(value.pop('format') == 'brass-spiel-state/1.0.0', 'Unknown state format')
        state = BrassSpielState(self)
        state.__dict__.update(value)
        state.setup_pools = {k: Counter(v) for k,v in state.setup_pools.items()}
        return state


class BrassSpielState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.game = game
        self.setup_stage = 'order'
        self.setup_pools = {'order': Counter(str(i) for i in range(4)),
            'merchants': game.merchant_pool.copy(), 'cards': Counter(game.composition)}
        self.setup_picked = {'order': [], 'merchants': [], 'cards': []}
        self.checkpoint = self.context = self.pending = None
        self.views, self.recall, self.actions = [None]*4, [[] for _ in range(4)], []
        self.legal = []
        self.final_returns = None
        self.final_result = None

    def current_player(self):
        if self.is_terminal(): return pyspiel.PlayerId.TERMINAL
        if self.setup_stage or self.pending: return pyspiel.PlayerId.CHANCE
        return self.context['actor']

    def is_terminal(self): return bool(self.context and self.context['terminal'])
    def is_chance_node(self): return self.current_player() == pyspiel.PlayerId.CHANCE
    def returns(self): return self.final_returns or [0.]*4
    def rewards(self): return self.returns() if self.is_terminal() else [0.]*4
    def history(self): return self.actions.copy()

    def semantic_map(self):
        if self.setup_stage:
            names = ['chance:'+self.setup_stage+':'+x for x,n in self.setup_pools[self.setup_stage].items() if n]
        elif self.pending:
            names = ['chance:card:'+x['card'] for x in self.pending['outcomes']]
        else:
            names = [c['id'] for c in self.legal]
        result = {action_id(n): n for n in names}
        require(len(result)==len(names), 'Semantic action hash collision; do not silently remap')
        return result

    def _legal_actions(self, player):
        return sorted(self.semantic_map()) if player==self.current_player() and not self.is_terminal() else []

    def chance_outcomes(self):
        require(self.is_chance_node(), 'Not chance')
        if self.setup_stage:
            pool = self.setup_pools[self.setup_stage]; total = sum(pool.values())
            return sorted((action_id('chance:'+self.setup_stage+':'+k), n/total) for k,n in pool.items() if n)
        return sorted((action_id('chance:card:'+x['card']), x['probability']) for x in self.pending['outcomes'])

    def _action_to_string(self, player, action): return self.semantic_map().get(action, str(action))

    def _apply_action(self, action):
        require(action in self.semantic_map(), 'Illegal semantic ID')
        name = self.semantic_map()[action]
        if self.setup_stage:
            stage = self.setup_stage; item = name.split(':',2)[2]
            self.setup_pools[stage][item] -= 1
            self.setup_picked[stage].append(item)
            if not sum(self.setup_pools[stage].values()):
                self.setup_stage = {'order':'merchants','merchants':'cards','cards':None}[stage]
                if self.setup_stage is None:
                    deck = self.setup_picked['cards']
                    setup = dict(order=list(map(int,self.setup_picked['order'])),
                        merchants=[dict(id=s,types=json.loads(t)) for s,t in zip(self.game.merchant_slots,self.setup_picked['merchants'])],
                        cards=dict(hands=[deck[i*8:(i+1)*8] for i in range(4)],
                            discards=[[deck[32+i]] for i in range(4)], discard_starts=[0]*4,
                            deck=deck[36:], wild_industry=['wild:industry']*4, wild_location=['wild:location']*4))
                    with self.game.lock:
                        self.context = self.game.host.reset(1, setup)
                        self.game.owner = self
                        self.refresh(self.game.host)
        else:
            with self.game.access(self) as host:
                if host.fast:
                    arguments=dict(card=name[len('chance:card:'):]) if self.pending else dict(choice=name)
                    snapshot=host.call('search_step',game_id=self.context['game_id'],revision=self.context['revision'],**arguments)
                    self.context=snapshot['context'];self.refresh(host,snapshot)
                elif self.pending:
                    self.context = host.call('chance_apply', game_id=self.context['game_id'],
                        revision=self.context['revision'], card=name[len('chance:card:'):])
                else:
                    self.context = host.apply(self.context, next(c for c in self.legal if c['id']==name))
                if not host.fast:self.refresh(host)
        self.actions.append(int(action))

    def refresh(self, host, snapshot=None):
        if host.fast and snapshot is None:snapshot=host.call('search_snapshot')
        self.pending = snapshot.get('chance') if snapshot is not None else host.call('chance')
        self.checkpoint = snapshot['checkpoint'] if snapshot is not None else host.call('checkpoint')
        if self.pending:
            self.legal = []  # Do not expose temporary draws or the temporary Rail deal.
            return
        self.views = snapshot['views'] if snapshot is not None else [host.observe(s) for s in range(4)]
        for s,view in enumerate(self.views):
            hand = view['own_hand']
            if not self.recall[s] or self.recall[s][-1]['hand'] != hand:
                self.recall[s].append(dict(revision=view['context']['revision'], hand=list(hand)))
        self.legal = snapshot['choices'] if snapshot is not None else ([] if self.is_terminal() else host.choices())
        if self.is_terminal():
            self.final_result=snapshot['result'] if snapshot is not None else host.call('result')
            self.final_returns = [p['utility'] for p in self.final_result['seats']]

    def observation_string(self, player=None):
        player = self.current_player() if player is None else player
        require(0<=player<4, 'Select a real observing seat')
        # Chance selections/deck/order are excluded until the allowed reveal boundary.
        return json.dumps(dict(view=self.views[player], setup=bool(self.setup_stage),
                               chance=self.pending['kind'] if self.pending else None), sort_keys=True)

    def information_state_string(self, player=None):
        player = self.current_player() if player is None else player
        require(0<=player<4, 'Select a real observing seat')
        return digest(dict(observation=self.observation_string(player), recall=self.recall[player]))

    def resample_from_infostate(self, player, sampler):
        require(not self.is_chance_node() and not self.is_terminal(), 'Resample at an observed decision')
        number = sampler() if callable(sampler) else sampler.uniform()
        return self.game.from_observation(self.views[player], int(float(number)*(2**53)), self.recall[player])

    def serialize(self):
        value = {k:copy.deepcopy(v) for k,v in self.__dict__.items() if k!='game'}
        return json.dumps(dict(format='brass-spiel-state/1.0.0', **value), sort_keys=True)

    def clone(self): return self.game.deserialize_state(self.serialize())
    def __str__(self): return self.observation_string(max(0,self.current_player()))


pyspiel.register_game(TYPE, BrassSpielGame)
