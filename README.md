# Theory of Mind in a Simplified Team-Based Card Game

Code for the DMAS course project (University of Groningen).
Group 12: Thom Boerboom, Jesper de Voogd, Diego Regnerus.

**Mini-Spades:**
- 4 players in 2 teams; teammates sit opposite each other (seats 0+2 vs 1+3).
- 16 cards: J, Q, K, A in four suits.
- 4 tricks per game; spades are trump; no bidding.

**Agents:** GRUs trained with policy gradient.
- **ToM0:** decides from the game state only.
- **ToM1:** also gets a prediction model of the next card another player will play.
- **ToM2:** also gets a two-stage prediction. A public model predicts from public information only, then a belief model adds the agent's own hand.

## Layout
- `legacy/Spades_game.ipynb`: original game engine and training notebook (individual setting, including the ToM2 experiment).
- `models/`: trained weights (`.pth`), see the table below.
- `evaluate.py`: evaluation script. It re-implements the game, loads the trained weights and plays them against each other without training. It needs only numpy; torch is optional.
- `results/`:
  - `suite.txt` / `suite.json`: ToM0 and ToM1 results.
  - `suite_tom2.txt` / `suite_tom2.json`: ToM2 results.

## Models
Every model is a one-layer GRU (hidden size 64) followed by a linear layer with 16 outputs (one per card).

| File | Inputs | Role |
|---|---|---|
| `spades_model.pth` | 52 | ToM0, individual setting. Input: hand 16 + table 16 + history 16 + trick position 4. Also used as the ToM2 backup. |
| `spades_model_ToM0_team.pth` | 52 | ToM0, team setting. |
| `opponent_prediction_model.pth` / `_revised.pth` | 52 | ToM1 prediction model: next card of another player. |
| `spades_model_ToM1.pth` / `_revised.pth` | 100 | ToM1 decision model: 52 + predictions for the 3 other players (zeros for players who already played). |
| `spades_model_ToM1_team_coop.pth` | 68 | ToM1 decision model: 52 + prediction for the teammate. |
| `spades_model_ToM1_team_comp.pth` | 68 | ToM1, opponent-directed. Input layout not documented; **not evaluated**. |
| `spades_model_ToM2_2.pth` / `_2revised.pth` | 36 | ToM2 public model: table + history + position. |
| `spades_model_ToM2_1.pth` / `_1revised.pth` | 68 | ToM2 belief model: hand + table + history + public prediction + position. |
| `spades_model_ToM2_base.pth` / `_revised.pth` | 68 | ToM2 decision model: 52 + belief prediction. |
| `spades_model_ToM2_team_coop_base.pth` / `_comp_base.pth` | 68 | ToM2 decision models, team setting (teammate- / opponent-directed). |

The ToM2 agent only uses its ToM2 models when it plays 1st or 2nd in a trick. Otherwise the ToM0 backup decides, as in the notebook.

## Reproducing the results
```
pip install -r requirements.txt
python evaluate.py --suite --games 5000 > results/suite.txt           # ToM0 / ToM1 (~1-2 min)
python evaluate.py --suite tom2 --games 5000 > results/suite_tom2.txt # ToM2 (~1.5 min)
```

**Team mode:**
- Every deal is played twice with the teams swapped, so 5,000 deals give 10,000 games.
- Reports the team score (win 1, draw 0.5, loss 0), mean team tricks and trick margin, each with a 95% confidence interval.

**Notebook mode:** reproduces the single-agent setting of the notebook (one agent against three copies of an opponent).

Single matchups, for example:
```
python evaluate.py --mode team --a tom0:models/spades_model_ToM0_team.pth --b heuristic --games 5000
python evaluate.py --mode notebook --a tom0:models/spades_model.pth --b random --games 5000
python evaluate.py --mode notebook --a "tom2:models/spades_model_ToM2_base.pth:models/spades_model_ToM2_1.pth:models/spades_model_ToM2_2.pth:models/spades_model.pth" --b random --games 5000
```

**Agent specs:**
- `random`
- `heuristic` (rule-based player)
- `tom0:<w>`
- `tom1:<w>:<pred>`
- `tom1coop:<w>:<pred>`
- `tom1zero:<w>:<pred>`
- `tom2:<main>:<belief>:<public>:<backup>`
- `tom2zero:...`

The `*zero` variants are the ablation: the same agent with its prediction inputs set to zero.

## Notes
The training code for the team models (`*_team*`) is not in this repository. Their evaluation therefore assumes:
- the same input layout as the individual models;
- teammates in seats 0+2 vs 1+3;
- the revised prediction models for the team ToM1/ToM2 agents.
