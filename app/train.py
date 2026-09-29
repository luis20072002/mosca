import time, pickle, os, numpy as np, brain, cube

MOVES = ["U", "U'", "R", "R'", "F", "F'"]
DEPTH = 5
HERE = os.path.dirname(__file__)

t0 = time.time()
table = brain.build_table(MOVES, DEPTH)
keys = [k for k, d in table.items() if d > 0]
print(f"BFS: {len(table)} estados alcanzables en <= {DEPTH} giros")

rng = np.random.default_rng(1)
sel = rng.choice(len(keys), size=len(keys), replace=False)
states, labels = [], []
for i in sel:
    s = np.frombuffer(keys[i], dtype=np.int8).copy()
    m = brain.optimal_move(s, table, MOVES)
    if m is not None:
        states.append(s); labels.append(MOVES.index(m))
labels = np.array(labels)
print("muestras:", len(states), "| balance:", np.bincount(labels))

n_test = 400
b = brain.FlyBrain(seed=0)
b.moves = MOVES
print("extrayendo tasas de disparo del conectoma...")
X = b._features(states)
print(f"  features {X.shape} en {time.time()-t0:.0f}s")

Xtr, Ytr = X[:-n_test], labels[:-n_test]
Xte, Yte = X[-n_test:], labels[-n_test:]
best = None
Ytr1 = np.eye(len(MOVES), dtype=np.float32)[Ytr]
for lam in [1, 10, 100, 1e3, 1e4]:
    Wout = brain.dual_ridge(Xtr, Ytr1, lam)
    acc = ((np.hstack([Xte, np.ones((len(Xte), 1), np.float32)]) @ Wout).argmax(1) == Yte).mean()
    print(f"  ridge={lam:>7}: precision test {acc:.3f}")
    if best is None or acc > best[0]:
        best = (acc, Wout)
# referencia honesta: lectura lineal sobre el estado crudo, sin cerebro
Rtr = np.stack([cube.onehot(s) for s in states[:-n_test]])
Rte = np.stack([cube.onehot(s) for s in states[-n_test:]])
bl = max(((np.hstack([Rte, np.ones((len(Rte), 1), np.float32)]) @
           brain.dual_ridge(Rtr, Ytr1, l)).argmax(1) == Yte).mean() for l in [1, 10, 100])
print(f"  referencia sin conectoma (lineal sobre el cubo crudo): {bl:.3f}")
acc, b.readout = best
print(f"precision en test: {acc:.3f} (azar = {1/len(MOVES):.3f})")

# --- prueba real: la red resuelve cubos que nunca vio?
solved = 0; trials = 25; lens = []
for k in range(trials):
    s = cube.SOLVED.copy()
    for m in cube.scramble(DEPTH, MOVES, np.random.default_rng(1000 + k)):
        s = cube.apply(s, m)
    for step in range(3 * DEPTH):
        if cube.is_solved(s): break
        mv, _, _, _ = b.decide(s)
        s = cube.apply(s, mv)
    if cube.is_solved(s):
        solved += 1; lens.append(step)
t1 = time.time(); b.decide(cube.apply(cube.SOLVED.copy(), "R")); 
print(f"tiempo por decision: {time.time()-t1:.2f}s")
print(f"cubos resueltos por la red: {solved}/{trials}" +
      (f" | giros medios {np.mean(lens):.1f} (optimo {DEPTH})" if lens else ""))

pickle.dump({"readout": b.readout, "moves": MOVES, "depth": DEPTH,
             "acc": float(acc), "solve_rate": solved / trials, "baseline": float(bl)},
            open(f"{HERE}/readout.pkl", "wb"))
print(f"guardado readout.pkl | total {time.time()-t0:.0f}s")
