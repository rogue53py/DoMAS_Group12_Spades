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

## Status
Evaluation (duplicate deals, team tricks, win rate with confidence intervals) is still to be written.
