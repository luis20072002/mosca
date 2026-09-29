"""Servidor Flask: el navegador muestra el cubo, Python corre la red del conectoma."""
import os, json, pickle, numpy as np
from flask import Flask, jsonify, request, send_from_directory
import brain, cube

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"))

print("cargando conectoma MaleCNS...")
B = brain.FlyBrain(seed=0)
ck = pickle.load(open(os.path.join(HERE, "readout.pkl"), "rb"))
B.readout, B.moves = ck["readout"], ck["moves"]
DEPTH = ck["depth"]
META = json.load(open(os.path.join(HERE, "..", "data", "meta.json")))

# neuronas que se dibujan en el raster (muestra estratificada por poblacion)
POP_ORDER = ["vpn", "kc", "dan", "mbon", "cx", "dn"]
POP_ES = {"vpn": "Proyeccion visual (entrada)",
          "kc": "Celulas de Kenyon", "dan": "Dopaminergicas (recompensa)",
          "mbon": "MBON (valor)", "cx": "Complejo central", "dn": "Descendentes (motor)"}
rng = np.random.default_rng(7)
rows, row_pop, pop_rows = [], [], {}
for p in POP_ORDER:
    idx = B.idx[p]
    take = idx if idx.size <= 220 else rng.choice(idx, 220, replace=False)
    pop_rows[p] = (len(rows), len(rows) + len(take))
    rows.extend(take.tolist()); row_pop.extend([p] * len(take))
ROWS = np.array(rows)
ROW_OF = {g: i for i, g in enumerate(ROWS)}
POP_IDX = {p: B.idx[p] for p in POP_ORDER}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    return jsonify(dict(
        neurons=int(B.N), synapses=int(B.W.nnz), depth=DEPTH,
        moves=B.moves, accuracy=ck["acc"], solve_rate=ck["solve_rate"],
        baseline=ck.get("baseline"),
        inhibitory=int((B.neurons["sign"] < 0).sum()),
        pops=[dict(key=p, name=POP_ES[p], n=int(B.idx[p].size),
                   r0=pop_rows[p][0], r1=pop_rows[p][1]) for p in POP_ORDER],
        n_rows=len(ROWS), steps=brain.T_STEPS, source=META))


@app.post("/api/scramble")
def scramble():
    d = int(request.json.get("depth", DEPTH))
    seed = int(request.json.get("seed", np.random.randint(1 << 30)))
    seq = cube.scramble(d, B.moves, np.random.default_rng(seed))
    st = cube.apply_seq(cube.SOLVED.copy(), seq)
    return jsonify(state=st.tolist(), scramble=seq, solved=cube.is_solved(st))


@app.post("/api/step")
def step():
    st = np.array(request.json["state"], np.int8)
    if cube.is_solved(st):
        return jsonify(done=True)
    mv, p, raster, counts = B.decide(st)
    nxt = cube.apply(st, mv)

    spikes = []                      # (t, fila) para el raster
    psth = {p_: [] for p_ in POP_ORDER}
    for t, fired in enumerate(raster):
        for g in fired:
            r = ROW_OF.get(int(g))
            if r is not None:
                spikes.append([t, r])
    c0 = counts[:, 0]
    for p_ in POP_ORDER:
        psth[p_] = float(c0[POP_IDX[p_]].sum() / POP_IDX[p_].size /
                         (brain.T_STEPS * brain.DT / 1000.0))
    return jsonify(done=False, move=mv, probs=p.tolist(), state=nxt.tolist(),
                   solved=cube.is_solved(nxt), spikes=spikes, rates=psth,
                   total_spikes=int(c0.sum()))


if __name__ == "__main__":
    print("listo -> http://127.0.0.1:5000")
    app.run(port=5000, debug=False, threaded=False)
