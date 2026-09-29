import numpy as np, cube

rng = np.random.default_rng(0)
ok = True

# 1) cada giro cuatro veces = identidad
for m in cube.FACES:
    s = cube.SOLVED.copy()
    for _ in range(4):
        s = cube.apply(s, m)
    assert cube.is_solved(s), f"{m}^4 != id"

# 2) (R U R' U')^6 = identidad  (prueba la interaccion entre caras)
s = cube.SOLVED.copy()
for _ in range(6):
    s = cube.apply_seq(s, ["R", "U", "R'", "U'"])
assert cube.is_solved(s), "(R U R' U')^6 != id"

# 3) las piezas fisicas se conservan (ninguna esquina/arista inventa colores)
base = cube._piece_sets(cube.SOLVED)
for trial in range(200):
    seq = cube.scramble(25, cube.ALL_MOVES, rng)
    s = cube.apply_seq(cube.SOLVED.copy(), seq)
    assert cube._piece_sets(s) == base, f"piezas inconsistentes: {seq}"
    # 4) revolver + inverso = resuelto
    assert cube.is_solved(cube.apply_seq(s, cube.inverse_seq(seq)))

# 5) cada color aparece exactamente 9 veces
for trial in range(50):
    s = cube.apply_seq(cube.SOLVED.copy(), cube.scramble(30, cube.ALL_MOVES, rng))
    assert np.all(np.bincount(s, minlength=6) == 9)

print("cube.py: todas las pruebas pasan")
