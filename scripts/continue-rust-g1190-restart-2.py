#!/usr/bin/env python3
"""Continue the second g1190 restart run (prepare-rust-g1190-restart-2.py), in one command.

Like continue-rust-g1190-restart.py, with 256 teacher rows per PPO step and the
insolvency charge kept (.0025, cap .025) in the reward:

1. With --skip-utility-qualification, start the one-iteration beta qualification
   immediately, without waiting for a utility fit. This choice persists across
   iterations and resumes. No teacher gradient is applied and the PPO LR holds.
   Without the flag, utility readiness is required before teacher collection.
2. Teaching switches on through the reviewed transition; the fresh search head
   starts Stage A (50 iterations), then Stage B (50) and Stage C (10).

The defaults are the CPU path with 64 workers and n=96 short searches (K=8);
long searches use n=640, K=16 until retuned. --iterations counts every iteration from
--expected-generation, calibration included. After an interruption, rerun with
--expected-generation set to the last committed generation: the command resumes
whichever step is unfinished.

--upgrade-utility-head applies a one-time same-run conversion at that committed
generation: utility hidden width 32 -> 256, Stage A budget 50 -> 100 (completed
iterations retained), then one fresh beta measurement without waiting for utility
qualification. Interrupted uncommitted work is archived before replaying it.
The learned utility/search weights, PPO/PPG moments, replay and RNG are retained.

--refresh-search-head replaces the 256-wide search head once with a fresh 32-wide
head at the committed generation. It restarts Stage A with w=0 and lambda=.8,
resets PPG beta to .25 and re-measures teacher beta with 96 teacher rows per PPO step while teaching stays on.
--adaptive-teacher-beta enables ongoing adjustment from the accepted beta.
Set --share-target at a committed iteration boundary. Adaptive beta carries
its current weight and running gradient norms into the new target.

--retune-teacher-search adopts long K=8 and C_scale=0.15 with short C_scale=0.5
at a committed Stage A boundary. It re-measures beta with the teacher on, retires
the old teacher labels and search values, and holds Stage A during buffer refill.
Each search-value fit runs for at most one epoch with the existing early stop.
Subsequent continuations use the committed search settings automatically.

--split-teacher-c-scale switches the mixed recipe to long C_scale=0.20 on
standard tables and 0.10 on cotton tables, and sets cotton targeting to 25%.
It retires both search buffers, keeps the committed search-head stage, and
re-measures beta with the teacher on.

--teacher-search-head-weight 0 selects a utility-only teacher override, including
held-outs. Pass 1 for full search-head weight. The override persists across
continuations. The retuned teacher recipe targets cotton in half of each split.

--reset-search-head replaces the current 32-wide head once and restarts Stage A.
Stage B uses its scheduled blend. The teacher-only override takes effect at
Stage C. Teacher cotton tables target one or two fixed random seats at 30%.
Ordinary tables use a 50/20/30 mix, with 15% cotton targeting on selected seats
in the cotton self-play games. Long and short C_scale are .15 and .5.
Ordinary PPG runs at most one epoch with a beta floor of .25 across all stages.

--update-opening-sampling continues the seated run with temperature 2 for each
seat's first three actions in ordinary and teacher games, 10% ordinary cotton
targeting, and a .5 PPG beta floor at every stage. Re-measure teacher beta on
the first iteration with teaching on. The new settings persist.

--update-cotton-sampling adopts the g2240 cotton recipe. Teacher cotton tables
make up 25% of training and held-outs, with normal teacher sampling disabled.
PPO uses 60/10/30 with 20% targeting on the cotton tables. Both use one fixed
random seat, target cotton builds from level two, and cap targeting at three
per era. The new settings persist.

--update-teacher-scales sets long C_scale=.10 and short C_scale=.70 for both
teacher splits. It re-measures beta with teaching on, refills the search
training buffers, and retains the dedicated held-out return games.

--update-cotton-network adds one targeted cotton-city network per era, separate
from the three cotton build/develop allowance, for both PPO and teacher cotton
seats. Teacher training tables use 15% cotton over time (4/5 of 32); each held-out
set stays three standard plus one cotton. Ordinary PPG caps at two epochs;
search-value PPG stays at one. Both keep the current .5 beta floor and buffers.

--update-g2600-recipe (at a committed g2400 cotton-network boundary) uses
temperature 1.3 for each seat's first three actions in PPO games and for the
cotton seat in teacher tables, long C_scale=.20 on standard and cotton teacher
tables (short stays .70), a .25 PPG beta floor in every search-head stage and
at most one ordinary PPG epoch. It retires the teacher label and search-value
buffers (held-out games included), keeps the search-head stage while they
refill, and re-measures beta with teaching on. The new settings persist.

--update-rail-cotton (at a committed g2600 boundary) stops PPO and teacher cotton
exploration from targeting Rail-era cotton builds below level three; level-one
builds stay excluded in both eras. Develop and network targets, allowances,
labels, buffers and the search-head stage are unchanged. The setting persists.

--update-search-lambda (at a committed g2654 boundary) sets the search-value return
lambda to .95 (was .8). Each fit recomputes the buffered games' returns at the new
lambda, so no buffer retires; held-out games still score the realized utility-to-go.
Labels, the search head, its stage and beta are unchanged. The setting persists.

--value-revision ITEMS (at a committed boundary) adopts g3140 value-phase items,
comma-separated; adopted items persist and only move forward
(ai-research/G3140_VALUE_REVISION.md):
  time        full-buffer train scans every tenth iteration; the search-value
              head keeps training.
  validation  the early stop validates on a quarter of the new games, which
              train in the next iteration (no permanent hold-out).
  heads       fresh era-separated local-scoring heads (weight .05).
  targets     win/utility heads train on half outcome, half lambda .95 return.
  weight=0.15, weight=0.5   local-scoring weight ladder (after heads).
  vp          total-VP training weight .1 -> .5.
  rail-end    Canal rows also predict the Rail-end map (after heads).
  ownership-stop  (g3318, after heads) the era heads' own early stop: every 16
              steps their held-out owner CE on the validation games decides
              whether they keep the steps (an increase keeps 0 and ends their
              fitting); after the utility stop they fit alone with the trunk
              frozen (ai-research/G3318_OWNERSHIP_CE_STOP.md).
  search-aux  (g3423, after ownership-stop and --teacher-revision) completed search-
              teacher tables train the era heads: each table sends its era-end
              scoring maps once with its finish message (CPU teacher rebuilt by the
              continuation), and once per iteration one batch of played roots from
              the 36 train tables plus the 4 leaving the search-value held-out set
              joins the first PPG step with ownership and local-VP losses only; the
              4 current search validation tables join the ownership stop
              (ai-research/G3423_SEARCH_AUXILIARY.md).
The held-out check, PPG cap .003, beta floor .25 and the strict early stop stay.

--rating-every 20 (at a rated generation, a multiple of ten) runs the PL strength
rating every 20 committed iterations from that generation on (was every 10):
from g3170, ratings run at g3190, g3210, ... Committed slots stay; set once.

--teacher-revision (at a committed boundary) adopts build steps 2 and 3
(ai-research/G3270_TEACHER_STEPS_2_3.md); set once, then kept:
  tables       36 train + 4 validation tables every iteration (no held-out tables
               every fifth); validation labels train from the next iteration.
  labels       8 iterations, age weights 8..1, long-search labels only; each use
               rebuilds the target from the current policy at label scale .20.
  batches      128 rows per PPO step, no row repeated inside the PPO update,
               grouped by legal width; logit parity on 8 rows per table.
  search value 16-iteration buffer, at most 32 steps, early stop on the latest 32
               held-out games (4 new per iteration; leaving games train the head).
Teacher beta is re-measured once with teaching on. Code-only in the same change:
teacher workers stream to a memory file (never wait), collection freezes the GC heap,
and tables are admitted while ordinary games finish.

--soft-policy (at a committed boundary, after --teacher-revision) adopts soft policy
step S1 (ai-research/G3423_SOFT_POLICY_S1.md); set once, then kept: a trainer-owned
Linear(192->1) without bias on the policy scorer's hidden layer predicts the rebuilt
label target softened at T=4 on the teacher term's 128 long-label rows per PPO step,
weight 8 x beta, with its own gradient call and Adam. Two head-only iterations, then
the trunk multiplier .2, .4, .6, .8, 1. Not exported; PPO rows join only at step S4.
"""
import importlib.util
from pathlib import Path

RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-2'


def main():
    path = Path(__file__).with_name('continue-rust-g1491.py')
    spec = importlib.util.spec_from_file_location('rust_g1491_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(run=RUN, teacher_fast=96, description=__doc__, share_target_required=True)


if __name__ == '__main__':
    main()
