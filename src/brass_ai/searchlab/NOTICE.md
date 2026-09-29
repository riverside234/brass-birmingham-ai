# R4/R5 code origins

The simulator adapter and S3 root action/observation tree are original project
code. No commercial game implementation or Daiyu source is included. The R4 host
build adds guarded pause/replay hooks to this project's original R1 host in a
separate generated file; it does not change the R1 sources or installed game.

`ismcts.py` adapts OpenSpiel's ISMCTS traversal and uses its runtime node, prior,
PUCT and resampling interfaces. Copyright 2019 DeepMind Technologies Limited.
These adapted mechanics are provided under Apache-2.0; the complete upstream
license accompanies this notice in `LICENSE-OpenSpiel`.

The R0 conceptual source pin remains
`google-deepmind/open_spiel@48401890ee9857e611678302371378175a8e4c6b`.
The separately pinned executable Python/C++ dependency is **open_spiel 2.0.2**.
The R4 changes use a masked fixed policy at opponent nodes, commit statistics
only after a complete simulation, enforce a turn deadline and bounded horizon,
and expose only permitted inputs at the real decision boundary. They are not
the unmodified upstream multi-player ISMCTS algorithm.

S3 follows root sampling and action/observation histories from Silver and Veness,
[Monte-Carlo Planning in Large POMDPs (2010)](https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html),
with complete-turn actions, four-seat neural leaves and separately motivated
masked opponents. No upstream POMCP code is copied, and its theoretical results
are not claimed for this approximation in Brass.
