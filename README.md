# Theory of Mind in a Simplified Team-Based Card Game

Code for the DMAS course project (University of Groningen).
Group: Thom Boerboom, Jesper de Voogd, Diego Regnerus.

Mini-Spades: 4 players, 2 teams, 16 cards (A K Q J x 4 suits), 4 tricks, spades are trump, no bidding.
Agents are GRUs trained with policy gradient (ToM0), extended with a prediction model of the other
players' next card (ToM1) and a two-stage variant (ToM2).

## Layout
- `legacy/Spades_game.ipynb` - original engine and training notebook (individual-reward, ToM2 experiment)
- `models/` - saved weights (`.pth`), see table below
- `results/` - evaluation output (to be added)
- `evaluate.py` - evaluation script (to be added)

## Models
Input size read from the weight files. Meaning of the extra inputs still to be confirmed.

| File | GRU input size | Notes |
|---|---|---|
| spades_model.pth | 52 | ToM0, individual reward (hand 16 + table 16 + history 16 + seat 4) |
| spades_model_ToM0_team.pth | 52 | ToM0, team reward |
| opponent_prediction_model.pth / _revised.pth | 52 | predicts the next card played by another player |
| spades_model_ToM1.pth / _revised.pth | 100 | 52 + predictions for the 3 other players (3 x 16) |
| spades_model_ToM1_team_coop.pth | 68 | 52 + 1 prediction (teammate?) - to confirm |
| spades_model_ToM1_team_comp.pth | 68 | to confirm (2 opponents would be 84) |
| spades_model_ToM2_*.pth | 68 / 36 | two-stage ToM2 models - to confirm |

## Evaluation
`python evaluate.py --suite --games 5000` runs the standard matchups (numpy only, about 1-2 minutes) and writes
`results/suite.json` and the printed table (`results/suite.txt`). Single matchups:

```
python evaluate.py --mode team --a tom0:models/spades_model_ToM0_team.pth --b heuristic --games 5000
python evaluate.py --mode notebook --a tom0:models/spades_model.pth --b random --games 5000
```
Team mode plays every deal twice with the teams swapped and reports team tricks, trick margin and win rate
with 95% confidence intervals. Notebook mode reproduces the single-agent setting of the notebook.
`tom1zero:` is an ablation (a ToM1 model whose prediction inputs are replaced by zeros).

Open points (to confirm with the author of the models): which team is which seat, the exact input layout of the
`_team_` and ToM2 models, and which predictor belongs to which agent. The pairings used in the suite are assumptions.
