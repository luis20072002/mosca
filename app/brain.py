"""Reservorio de conectoma: red LIF con los pesos sinapticos reales del MaleCNS.

Idea para la clase: el conectoma es un reservorio recurrente FIJO (no se entrena
ni una sinapsis). Solo se entrena una lectura lineal sobre las tasas de disparo
de las poblaciones de aprendizaje/decision (KC, MBON, CX, DN).  Es reservoir
computing sobre el cerebro real de una mosca.
"""
import os, json, pickle
import numpy as np, pandas as pd, scipy.sparse as sp
import cube

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

DT, TAU, VTH, T_STEPS = 1.0, 20.0, 1.0, 40      # ms
READOUT_POPS = ["vpn", "kc", "dan", "mbon", "cx", "dn"]


class FlyBrain:
    def __init__(self, seed=0):
        self.W = sp.load_npz(f"{DATA}/W.npz").T.tocsr().astype(np.float32)  # entrada a post
        self.neurons = pd.read_parquet(f"{DATA}/neurons.parquet")
        self.N = self.W.shape[0]
        self.pop = self.neurons["pop"].to_numpy()
        self.idx = {p: np.flatnonzero(self.pop == p) for p in np.unique(self.pop)}

        # normaliza por la entrada sinaptica media (mantiene la red en rango estable)
        scale = np.abs(self.W).sum(axis=1).A.ravel().mean()
        self.W.data /= np.float32(scale)

        rng = np.random.default_rng(seed)
        ret = self.idx["vpn"]
        # proyeccion fija estimulo -> neuronas de proyeccion visual.
        # El lobulo optico intrinseco (89k neuronas) no esta en el modelo reducido,
        # asi que la imagen del cubo se inyecta donde vive la representacion visual.
        self.P = rng.normal(0, 1, (324, ret.size)).astype(np.float32) * 0.6
        self.ret = ret
        self.read_idx = np.concatenate([self.idx[p] for p in READOUT_POPS])
        self.gain_rec, self.gain_in, self.noise = 4.0, 2.2, 0.10
        self.readout = None
        self.moves = None
        self.rng = rng

    # ------------------------------------------------------------ dinamica LIF
    def run(self, stim, record=False):
        """stim: (B, 324). Devuelve tasas (B, n_read) y opcionalmente el raster."""
        B = stim.shape[0]
        rng = np.random.default_rng(12345)
        iext = np.zeros((self.N, B), np.float32)
        iext[self.ret] = (stim @ self.P).T * self.gain_in
        v = np.zeros((self.N, B), np.float32)
        s = np.zeros((self.N, B), np.float32)
        counts = np.zeros((self.N, B), np.float32)
        raster = [] if record else None
        a = np.float32(DT / TAU)
        for t in range(T_STEPS):
            i_rec = self.W @ s
            drive = self.gain_rec * i_rec + iext
            drive += rng.normal(0, self.noise, (self.N, B)).astype(np.float32)
            v += a * (-v + drive)
            np.maximum(v, -2.0, out=v)
            s = (v > VTH).astype(np.float32)
            v *= (1.0 - s)                      # reset
            counts += s
            if record:
                fired = np.flatnonzero(s[:, 0] > 0)
                raster.append(fired)
        rates = (counts[self.read_idx].T / (T_STEPS * DT / 1000.0))   # Hz
        return rates, raster, counts

    # ------------------------------------------------------- lectura entrenada
    def fit(self, states, labels, ridge=1.0):
        X = self._features(states)
        Y = np.eye(len(self.moves), dtype=np.float32)[labels]
        self.readout = dual_ridge(X, Y, ridge)
        return self

    def _features(self, states, chunk=192):
        out = []
        for i in range(0, len(states), chunk):
            stim = np.stack([cube.onehot(s) for s in states[i:i + chunk]])
            r, _, _ = self.run(stim)
            out.append(r)
        X = np.vstack(out)
        return np.log1p(X)

    def decide(self, state):
        """Devuelve (movimiento, distribucion sobre movimientos, raster, tasas)."""
        stim = cube.onehot(state)[None, :]
        rates, raster, counts = self.run(stim, record=True)
        x = np.hstack([np.log1p(rates), np.ones((1, 1), np.float32)])
        logits = (x @ self.readout).ravel()
        p = np.exp(logits - logits.max()); p /= p.sum()
        return self.moves[int(p.argmax())], p, raster, counts


def dual_ridge(X, Y, lam):
    """Ridge en forma dual: coste O(n_muestras^3), no O(n_features^3)."""
    Xb = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    K = Xb @ Xb.T
    A = np.linalg.solve(K + lam * np.eye(len(K), dtype=K.dtype), Y)
    return (Xb.T @ A).astype(np.float32)


# ------------------------------------------------------------------ BFS: verdad
def build_table(moves, depth):
    """Distancia exacta al estado resuelto para todo estado a <= `depth` giros."""
    start = cube.SOLVED.tobytes()
    dist = {start: 0}
    frontier = [cube.SOLVED]
    for d in range(1, depth + 1):
        nxt = []
        for st in frontier:
            for m in moves:
                s2 = cube.apply(st, m)
                k = s2.tobytes()
                if k not in dist:
                    dist[k] = d
                    nxt.append(s2)
        frontier = nxt
    return dist


def optimal_move(state, table, moves):
    d = table.get(state.tobytes())
    if d is None or d == 0:
        return None
    for m in moves:
        k = cube.apply(state, m).tobytes()
        if table.get(k, 99) == d - 1:
            return m
    return None
