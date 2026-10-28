#!/usr/bin/env python3
"""Evaluation of the Mini-Spades agents (ToM0 / ToM1 GRU models).

Re-implements the game and the way the notebook (legacy/Spades_game.ipynb) feeds the models,
but WITHOUT training: weights are loaded from the .pth files and played against each other.

Game: 4 players (seats 0-3, clockwise), teams {0,2} vs {1,3}, 16 cards (J Q K A x H D C S),
4 cards each, 4 tricks, spades are trump, follow suit, winner of a trick leads the next one.
Card index = rank_idx * 4 + suit_idx, as in the notebook (suit 3 = spades).

Two evaluation modes
  notebook : one agent (seat 0) against three identical opponents; individual win rate
             (most tricks, ties split). Reproduces the numbers printed by the notebook.
  team     : team A (seats 0,2) vs team B (seats 1,3). Every deal is played twice with the
             teams swapped (duplicate dealing). Reports team tricks, trick margin and win rate
             with 95% confidence intervals.

Agent specs (strings)
  random
  heuristic                       simple rule-based player (see HeuristicAgent)
  tom0:<weights.pth>              GRU, input 52 = hand(16) + table(16) + history(16) + seat(4)
  tom1:<weights.pth>:<pred.pth>   GRU, input 100 = 52 + predictions for the 3 other players
  tom1coop:<weights.pth>:<pred.pth>   GRU, input 68 = 52 + prediction for the teammate only (UNVERIFIED layout)
  tom1zero:<weights.pth>:<pred.pth>    ablation: a tom1 model whose prediction inputs are replaced by zeros

Examples
  python evaluate.py --mode notebook --a tom0:models/spades_model.pth --b random --games 5000
  python evaluate.py --mode team --a tom0:models/spades_model_ToM0_team.pth --b heuristic --games 5000
  python evaluate.py --suite        # runs the standard set of matchups and writes results/

Only numpy is required. If PyTorch is installed it is used to read the .pth files, otherwise a
small built-in reader is used.
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time
import zipfile

import numpy as np

SPADE = 3


def suit(c):
    return c % 4


def rank(c):
    return c // 4


def beats(c, win_card, led):
    if suit(c) == SPADE and suit(win_card) != SPADE:
        return True
    return suit(c) == suit(win_card) and rank(c) > rank(win_card)


def trick_winner(table, led):
    """table: list of (seat, card). Spades win; otherwise the highest card of the led suit."""
    sp = [p for p in table if suit(p[1]) == SPADE]
    pool = sp if sp else [p for p in table if suit(p[1]) == led]
    return max(pool, key=lambda p: rank(p[1]))[0]


def onehot(cards, n=16):
    v = np.zeros(n)
    for c in cards:
        v[c] += 1
    return v


# ----------------------------------------------------------------------------- weights
def _read_pth_without_torch(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return lambda storage, off, size, stride, *a: ("T", storage, off, tuple(size))
            if name == "OrderedDict":
                return collections.OrderedDict
            if name.endswith("Storage"):
                return name
            raise pickle.UnpicklingError(f"unexpected object {mod}.{name} in {path}")

        def persistent_load(self, pid):
            return ("S", pid[1], pid[2])

    sd = U(z.open(pkl)).load()
    dt = {"FloatStorage": "<f4", "DoubleStorage": "<f8"}
    out = {}
    for k, (_, (_, stype, key), off, size) in sd.items():
        raw = np.frombuffer(z.read(f"{prefix}data/{key}"), dtype=dt[stype])
        n = int(np.prod(size))
        out[k] = raw[off:off + n].reshape(size).astype(np.float64)
    return out


def load_weights(path):
    try:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.detach().numpy().astype(np.float64) for k, v in sd.items()}
    except ImportError:
        return _read_pth_without_torch(path)


_NETS = {}


class GRUNet:
    """One-layer GRU followed by a linear layer, evaluated one step at a time (batch size 1)."""

    def __init__(self, path):
        w = load_weights(path)
        self.wih, self.whh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"]
        self.bih, self.bhh = w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
        self.wfc, self.bfc = w["fc.weight"], w["fc.bias"]
        self.H, self.n_in = self.whh.shape[1], self.wih.shape[1]
        self.path = path

    @staticmethod
    def get(path):
        if path not in _NETS:
            _NETS[path] = GRUNet(path)
        return _NETS[path]

    def step(self, x, h):
        if len(x) != self.n_in:
            raise ValueError(f"{os.path.basename(self.path)} expects {self.n_in} inputs, got {len(x)}")
        if h is None:
            h = np.zeros(self.H)
        H = self.H
        gi, gh = self.wih @ x + self.bih, self.whh @ h + self.bhh
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        z = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
        return self.wfc @ h + self.bfc, h


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


# ----------------------------------------------------------------------------- state
class State:
    """What is on the table when `seat` is about to play."""

    def __init__(self, order, idx, table, hist):
        self.order, self.idx, self.table, self.hist = order, idx, table, hist
        self.tablehot = onehot([c for _, c in table])
        self.histhot = onehot(hist)

    @property
    def led(self):
        return suit(self.table[0][1]) if self.table else None


# ----------------------------------------------------------------------------- agents
class Agent:
    kind = "agent"

    def start_game(self, seat, hand):
        self.seat, self.hand = seat, list(hand)

    def new_trick(self):
        pass

    def watch(self, actor, st):
        pass

    def saw_play(self, seat, card):
        pass

    def predictor_stats(self):
        return None


class RandomAgent(Agent):
    kind = "random"

    def __init__(self, rng):
        self.rng = rng

    def act(self, st, legal):
        return legal[self.rng.integers(len(legal))]


class HeuristicAgent(Agent):
    """Rule-based reference player (no learning, no beliefs).
    Lead: highest non-spade (highest spade if only spades). Follow: if the teammate is currently
    winning, play the lowest legal card; otherwise win the trick with the lowest winning card if
    possible, else discard the lowest card (non-spades first)."""
    kind = "heuristic"

    def __init__(self, rng=None):
        pass

    def act(self, st, legal):
        low = lambda cs: min(cs, key=lambda c: (suit(c) == SPADE, rank(c)))
        if not st.table:
            non = [c for c in legal if suit(c) != SPADE]
            return max(non or legal, key=rank)
        win_seat = trick_winner(st.table, st.led)
        win_card = dict(st.table)[win_seat]
        if win_seat == (self.seat + 2) % 4:
            return low(legal)
        winning = [c for c in legal if beats(c, win_card, st.led)]
        return low(winning) if winning else low(legal)


class GRUAgent(Agent):
    """GRU policy as in the notebook. Hidden state is carried over the agent's own turns in a game.
    kind: tom0 | tom1 | tom1coop. seat_input: 'own' (position of this agent in the trick) or
    'player0' (position of seat 0, which is what frozen opponents received in the notebook)."""

    def __init__(self, kind, net_path, pred_path, rng, greedy=False, seat_input="own"):
        self.kind, self.rng, self.greedy, self.seat_input = kind, rng, greedy, seat_input
        self.net = GRUNet.get(net_path)
        self.pred = GRUNet.get(pred_path) if pred_path else None
        self.stats = dict(hits=0, n=0, chance=0.0)

    def start_game(self, seat, hand):
        super().start_game(seat, hand)
        self.h = None
        self.pred_h = {1: None, 2: None, 3: None}   # predictor hidden state per other player (relative id)
        self.new_trick()

    def new_trick(self):
        self.played = False
        self.slots = []          # predictions in play order (zeros for players that acted before us)
        self.slot_by_rel = {}
        self.pending = {}        # seat -> (probs, #candidate cards), scored in saw_play

    # --- predictor of the other players' next card (input as in the notebook)
    def _predict(self, actor, st):
        rel = (actor - self.seat) % 4
        seen = self.histhot_plus(st)
        x = np.concatenate([onehot(self.hand), st.tablehot, st.histhot, np.eye(4)[rel]])
        logits, self.pred_h[rel] = self.pred.step(x, self.pred_h[rel])
        allowed = seen == 0
        if not allowed.any():
            allowed[:] = True
        logits = np.where(allowed, logits, -np.inf)
        probs = softmax(logits)
        self.pending[actor] = (probs, int(allowed.sum()))
        return rel, probs

    def histhot_plus(self, st):
        return st.histhot + st.tablehot + onehot(self.hand)

    def watch(self, actor, st):
        if self.pred is None or self.played or actor == self.seat:
            return
        rel, probs = self._predict(actor, st)    # hidden state advances; output is not used as input
        self.slots.append(np.zeros(16))
        self.slot_by_rel[rel] = np.zeros(16)

    def saw_play(self, seat, card):
        if seat in self.pending:
            probs, n = self.pending.pop(seat)
            self.stats["n"] += 1
            self.stats["hits"] += int(np.argmax(probs) == card)
            self.stats["chance"] += 1.0 / n

    def predictor_stats(self):
        return self.stats if self.pred is not None else None

    def act(self, st, legal):
        pos_seat = self.seat if self.seat_input == "own" else 0
        pos = st.order.index(pos_seat)
        x = [onehot(self.hand), st.tablehot, st.histhot, np.eye(4)[pos]]
        if self.pred is not None:
            for f in st.order[st.idx + 1:]:           # players that still have to play after us
                rel, probs = self._predict(f, st)
                self.slots.append(probs)
                self.slot_by_rel[rel] = probs
            if self.kind == "tom1":
                x += self.slots
            elif self.kind == "tom1zero":      # ablation: same model, predictions replaced by zeros
                x += [np.zeros(16)] * 3
            elif self.kind == "tom1coop":
                x.append(self.slot_by_rel.get(2, np.zeros(16)))
        logits, self.h = self.net.step(np.concatenate(x), self.h)
        self.played = True
        lg = np.array([logits[c] for c in legal])
        if self.greedy:
            return legal[int(np.argmax(lg))]
        return legal[self.rng.choice(len(legal), p=softmax(lg))]


def make_agent(spec, rng, greedy=False, seat_input="own"):
    parts = spec.split(":")
    kind = parts[0]
    if kind == "random":
        return RandomAgent(rng)
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "tom0":
        return GRUAgent("tom0", parts[1], None, rng, greedy, seat_input)
    if kind in ("tom1", "tom1coop", "tom1zero"):
        return GRUAgent(kind, parts[1], parts[2], rng, greedy, seat_input)
    raise ValueError(f"unknown agent spec {spec!r}")


# ----------------------------------------------------------------------------- engine
def play_deal(hands, leader, agents):
    """Plays one game of 4 tricks. Returns the number of tricks won per seat."""
    for s in range(4):
        agents[s].start_game(s, hands[s])
    hist, tricks = [], [0] * 4
    for _ in range(4):
        order = [(leader + i) % 4 for i in range(4)]
        table = []
        for a in agents:
            a.new_trick()
        for idx, seat in enumerate(order):
            st = State(order, idx, table, hist)
            for s in range(4):
                if s != seat:
                    agents[s].watch(seat, st)
            a = agents[seat]
            hand = a.hand
            if table:
                f = [c for c in hand if suit(c) == suit(table[0][1])]
                legal = sorted(f or hand)
            else:
                legal = sorted(hand)
            card = a.act(st, legal)
            assert card in legal
            hand.remove(card)
            table.append((seat, card))
            for s in range(4):
                agents[s].saw_play(seat, card)
        winner = trick_winner(table, suit(table[0][1]))
        tricks[winner] += 1
        leader = winner
        hist += [c for _, c in table]
    return tricks


def new_deal(rng):
    p = rng.permutation(16)
    return [[int(c) for c in p[4 * k:4 * k + 4]] for k in range(4)], int(rng.integers(4))


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def pred_summary(agents):
    h = n = c = 0
    for a in agents:
        s = a.predictor_stats()
        if s:
            h, n, c = h + s["hits"], n + s["n"], c + s["chance"]
    return None if n == 0 else dict(accuracy=h / n, chance=c / n, n=n)


# ----------------------------------------------------------------------------- modes
def run_notebook(spec_a, spec_b, games, seed, greedy=False, seat_input_b="own"):
    """Agent A at seat 0 against three copies of B (the setting of the notebook)."""
    rng = np.random.default_rng(seed)
    a = make_agent(spec_a, rng, greedy)
    opp = [make_agent(spec_b, rng, greedy, seat_input_b) for _ in range(3)]
    agents = [a] + opp
    win, tr = [], []
    for _ in range(games):
        hands, leader = new_deal(rng)
        t = play_deal(hands, leader, agents)
        best = max(t)
        winners = [i for i in range(4) if t[i] == best]
        win.append(1 / len(winners) if 0 in winners else 0.0)
        tr.append(t[0])
    return dict(mode="notebook", a=spec_a, b=spec_b, games=games,
                win_rate=ci(win), tricks_A=ci(tr), predictor=pred_summary([a]))


def run_team(spec_a, spec_b, games, seed, greedy=False, seat_input="own"):
    """Team A vs team B with duplicate deals: every deal is played twice with the teams swapped."""
    rng = np.random.default_rng(seed)
    A = [make_agent(spec_a, rng, greedy, seat_input) for _ in range(2)]
    B = [make_agent(spec_b, rng, greedy, seat_input) for _ in range(2)]
    tricks_a, margin, win, tie = [], [], [], []
    for _ in range(games):
        hands, leader = new_deal(rng)
        ta, mg, w, ti = [], [], [], []
        for swap in (0, 1):
            sa, sb = ((0, 2), (1, 3)) if swap == 0 else ((1, 3), (0, 2))
            agents = [None] * 4
            agents[sa[0]], agents[sa[1]] = A
            agents[sb[0]], agents[sb[1]] = B
            t = play_deal(hands, leader, agents)
            a_t = t[sa[0]] + t[sa[1]]
            ta.append(a_t)
            mg.append(a_t - (4 - a_t))
            w.append(1.0 if a_t > 2 else 0.5 if a_t == 2 else 0.0)
            ti.append(1.0 if a_t == 2 else 0.0)
        tricks_a.append(np.mean(ta)); margin.append(np.mean(mg)); win.append(np.mean(w)); tie.append(np.mean(ti))
    return dict(mode="team", a=spec_a, b=spec_b, games=games * 2, deals=games,
                tricks_A=ci(tricks_a), margin=ci(margin), win_rate=ci(win), tie_rate=float(np.mean(tie)),
                predictor=pred_summary(A))


def fmt(r):
    m = lambda t, p=1: f"{t[0] * (100 if p else 1):.1f}{'%' if p else ''} ±{t[1] * (100 if p else 1):.1f}"
    s = f"{r['a']}  vs  {r['b']}  [{r['mode']}, {r['games']} games]\n"
    s += f"  win rate (ties split): {m(r['win_rate'])}\n"
    s += f"  tricks won by A: {r['tricks_A'][0]:.3f} ±{r['tricks_A'][1]:.3f}"
    s += " (of 4, per player)" if r["mode"] == "notebook" else " (of 4, per team)"
    if r["mode"] == "team":
        s += f"\n  trick margin A-B: {r['margin'][0]:+.3f} ±{r['margin'][1]:.3f}   tie rate: {r['tie_rate']*100:.1f}%"
    if r["predictor"]:
        p = r["predictor"]
        s += f"\n  predictor: next-card accuracy {p['accuracy']*100:.1f}% vs uniform-guess level {p['chance']*100:.1f}% (n={p['n']})"
    return s


def standard_suite(models, games, seed):
    M = lambda f: os.path.join(models, f)
    P, P2 = M("opponent_prediction_model.pth"), M("opponent_prediction_model_revised.pth")
    t0, t0t = M("spades_model.pth"), M("spades_model_ToM0_team.pth")
    jobs = [
        ("notebook", f"tom0:{t0}", "random", dict(seat_input_b="own")),
        ("notebook", f"tom0:{t0}", f"tom0:{t0}", dict(seat_input_b="player0")),
        ("notebook", f"tom1:{M('spades_model_ToM1.pth')}:{P}", f"tom0:{t0}", dict(seat_input_b="player0")),
        ("notebook", f"tom1:{M('spades_model_ToM1_revised.pth')}:{P2}", f"tom0:{t0}", dict(seat_input_b="player0")),
        ("notebook", f"tom1zero:{M('spades_model_ToM1_revised.pth')}:{P2}", f"tom0:{t0}", dict(seat_input_b="player0")),
        ("team", f"tom0:{t0t}", "random", {}),
        ("team", "heuristic", "random", {}),
        ("team", f"tom0:{t0t}", "heuristic", {}),
        ("team", f"tom1coop:{M('spades_model_ToM1_team_coop.pth')}:{P2}", f"tom0:{t0t}", {}),
        ("team", f"tom1:{M('spades_model_ToM1_revised.pth')}:{P2}", f"tom0:{t0t}", {}),
        ("team", f"tom1zero:{M('spades_model_ToM1_revised.pth')}:{P2}", f"tom0:{t0t}", {}),
        ("team", f"tom0:{t0t}", f"tom0:{t0t}", {}),
    ]
    out = []
    for mode, a, b, kw in jobs:
        t = time.time()
        try:
            r = (run_notebook if mode == "notebook" else run_team)(a, b, games, seed, **kw)
            print(fmt(r), f"\n  ({time.time() - t:.0f}s)\n", flush=True)
        except Exception as e:  # keep going, report the problem
            r = dict(mode=mode, a=a, b=b, error=str(e))
            print(f"{a} vs {b}: ERROR {e}\n", flush=True)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["notebook", "team"], default="team")
    ap.add_argument("--a"), ap.add_argument("--b")
    ap.add_argument("--games", type=int, default=2000, help="games (notebook) or deals, each played twice (team)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="play the arg-max card instead of sampling")
    ap.add_argument("--seat-input", choices=["own", "player0"], default="own",
                    help="seat one-hot given to opponents: own seat, or seat 0's position (notebook quirk)")
    ap.add_argument("--suite", action="store_true", help="run the standard set of matchups")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.suite:
        res = standard_suite(args.models, args.games, args.seed)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "suite.json"), "w") as f:
            json.dump(dict(games=args.games, seed=args.seed, results=res), f, indent=1, default=float)
        print(f"saved {args.out}/suite.json")
        return
    if not (args.a and args.b):
        ap.error("--a and --b are required (or use --suite)")
    if args.mode == "notebook":
        r = run_notebook(args.a, args.b, args.games, args.seed, args.greedy, args.seat_input)
    else:
        r = run_team(args.a, args.b, args.games, args.seed, args.greedy, args.seat_input)
    print(fmt(r))


if __name__ == "__main__":
    main()
