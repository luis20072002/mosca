"""Extrae una subred del conectoma MaleCNS (Janelia) para la demo.

Lee los .feather descargados y guarda una matriz dispersa de pesos sinapticos
reales + metadatos, en data/subnet.npz
"""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.feather as pf
import scipy.sparse as sp, json, os

DATA = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------- 1. neuronas
ann = pf.read_table(f"{DATA}/body-annotations.feather",
                    columns=["bodyId", "superclass", "class", "type", "somaSide"]).to_pandas()
ann["superclass"] = ann["superclass"].fillna("")
ann["class"] = ann["class"].fillna("")

# Cada poblacion = una etapa del "pipeline" visual-motor de la mosca
def pick(mask, name, cap=None, seed=0):
    d = ann[mask].copy()
    if cap is not None and len(d) > cap:
        d = d.sample(cap, random_state=seed)
    d["pop"] = name
    return d

pops = [
    pick(ann.superclass == "ol_sensory",         "retina"),      # fotorreceptores
    pick(ann.superclass == "visual_projection",  "vpn"),         # proyeccion visual
    pick(ann["class"] == "Kenyon_Cell",          "kc"),          # codigo disperso
    pick(ann["class"] == "MBON",                 "mbon"),        # lectura de valor
    pick(ann["class"] == "DAN",                  "dan"),         # recompensa / RPE
    pick(ann["class"] == "CX",                   "cx"),          # seleccion de accion
    pick(ann.superclass == "descending_neuron",  "dn"),          # salida motora
]
neurons = pd.concat(pops, ignore_index=True).drop_duplicates("bodyId", keep="first")
neurons = neurons.sort_values("bodyId").reset_index(drop=True)
bodies = neurons.bodyId.to_numpy(np.int64)
index = {b: i for i, b in enumerate(bodies)}
N = len(bodies)
print(f"neuronas seleccionadas: {N}")
print(neurons["pop"].value_counts())

# ------------------------------------------------- 2. neurotransmisor -> signo
nt = pf.read_table(f"{DATA}/body-neurotransmitters.feather",
                   columns=["body", "consensus_nt"]).to_pandas()
nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
# Convencion estandar en modelos del conectoma de Drosophila (Shiu et al. 2024):
# acetilcolina excitatoria; GABA y glutamato (via GluCl) inhibitorios.
INHIB = {"gaba", "glutamate"}
sign = np.ones(N, np.float32)
nt_named = nt.reindex(bodies)
sign[nt_named.isin(INHIB).to_numpy()] = -1.0
neurons["nt"] = nt_named.to_numpy()
neurons["sign"] = sign
print("inhibitorias:", int((sign < 0).sum()), "/", N)

# ------------------------------------------------------- 3. aristas reales
src = pa.memory_map(f"{DATA}/connectome-weights.feather")
tbl = pf.read_table(src, columns=["body_pre", "body_post", "weight"])
print("aristas totales en el conectoma:", tbl.num_rows)

rows, cols, vals = [], [], []
MIN_W = 3                      # descarta contactos debiles/ruido de segmentacion
for batch in tbl.to_batches(max_chunksize=4_000_000):
    pre = batch.column("body_pre").to_numpy(zero_copy_only=False)
    post = batch.column("body_post").to_numpy(zero_copy_only=False)
    w = batch.column("weight").to_numpy(zero_copy_only=False)
    keep = w >= MIN_W
    pre, post, w = pre[keep], post[keep], w[keep]
    ip = np.searchsorted(bodies, pre); ip[ip >= len(bodies)] = 0
    iq = np.searchsorted(bodies, post); iq[iq >= len(bodies)] = 0
    ok = (bodies[ip] == pre) & (bodies[iq] == post)
    if ok.any():
        rows.append(ip[ok]); cols.append(iq[ok]); vals.append(w[ok])

rows = np.concatenate(rows); cols = np.concatenate(cols)
vals = np.concatenate(vals).astype(np.float32)
print("sinapsis dentro de la subred:", len(vals))

# peso con signo del neurotransmisor de la neurona presinaptica
vals = vals * sign[rows]
W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
W.sum_duplicates()

sp.save_npz(f"{DATA}/W.npz", W)
neurons.to_parquet(f"{DATA}/neurons.parquet")
meta = {"N": N, "edges": int(W.nnz), "min_weight": MIN_W,
        "pops": neurons["pop"].value_counts().to_dict()}
json.dump(meta, open(f"{DATA}/meta.json", "w"), indent=2)
print("listo ->", meta)
