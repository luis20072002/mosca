"""Cubo de Rubik 3x3 en notacion de facelets (orden Kociemba: U R F D L B)."""
import numpy as np

FACES = "URFDLB"
COLORS = list(FACES)                      # color = cara de origen
SOLVED = np.repeat(np.arange(6), 9).astype(np.int8)

# Cada facelet se identifica por (cubie 3D, normal de su cara). Un giro horario
# es una rotacion de -90 grados alrededor del eje saliente de esa cara, aplicada
# tanto a la posicion como a la normal. Derivarlo asi evita errores de tablas.
_AXIS = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0),
         "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}


def _facelet_coords(i):
    f, r, c = FACES[i // 9], (i % 9) // 3, i % 3
    if f == "U":   p = (c - 1, 1, r - 1)
    elif f == "F": p = (c - 1, 1 - r, 1)
    elif f == "R": p = (1, 1 - r, 1 - c)
    elif f == "D": p = (c - 1, -1, 1 - r)
    elif f == "L": p = (-1, 1 - r, c - 1)
    else:          p = (1 - c, 1 - r, -1)
    return p, _AXIS[f]


def _rot(v, axis):
    """Rotacion de -90 grados (horaria vista desde fuera) alrededor de `axis`."""
    x, y, z = v
    ax, ay, az = axis
    if ax:  return (x, ax * z, -ax * y)
    if ay:  return (-ay * z, y, ay * x)
    return (az * y, -az * x, z)


_COORD = {i: _facelet_coords(i) for i in range(54)}
_LOOKUP = {pn: i for i, pn in _COORD.items()}


def _perm(face):
    """Permutacion p tal que nuevo[i] = viejo[p[i]] para un giro horario."""
    axis = _AXIS[face]
    fixed = [k for k in range(3) if axis[k]][0]
    p = np.arange(54)
    for i in range(54):
        pos, nrm = _COORD[i]
        if pos[fixed] != axis[fixed]:
            continue                       # facelet fuera de la capa que gira
        dest = _LOOKUP[(_rot(pos, axis), _rot(nrm, axis))]
        p[dest] = i
    return p



PERMS = {}
for _f in FACES:
    _p = _perm(_f)
    PERMS[_f] = _p                                     # horario
    PERMS[_f + "'"] = np.argsort(_p)                   # antihorario
    PERMS[_f + "2"] = _p[_p]                           # doble

ALL_MOVES = [f + s for f in FACES for s in ("", "'", "2")]


def apply(state, move):
    return state[PERMS[move]]


def apply_seq(state, moves):
    for m in moves:
        state = apply(state, m)
    return state


def inverse(move):
    if move.endswith("2"):
        return move
    return move[0] if move.endswith("'") else move + "'"


def inverse_seq(moves):
    return [inverse(m) for m in reversed(moves)]


def is_solved(state):
    return bool(np.array_equal(state, SOLVED))


def scramble(depth, moves, rng):
    """Secuencia aleatoria sin deshacer el giro inmediatamente anterior."""
    seq = []
    for _ in range(depth):
        cand = [m for m in moves if not seq or m[0] != seq[-1][0]]
        seq.append(str(rng.choice(cand)))
    return seq


def onehot(state):
    v = np.zeros((54, 6), np.float32)
    v[np.arange(54), state] = 1.0
    return v.ravel()


# ------------------------------------------------------------------ validacion
def _cubies():
    """Agrupa los 54 facelets en piezas fisicas usando coordenadas 3D."""
    g = {}
    for f_i, f in enumerate(FACES):
        for r in range(3):
            for c in range(3):
                i = f_i * 9 + r * 3 + c
                if f == "U":   p = (c - 1, 1, r - 1)
                elif f == "F": p = (c - 1, 1 - r, 1)
                elif f == "R": p = (1, 1 - r, 1 - c)
                elif f == "D": p = (c - 1, -1, 1 - r)
                elif f == "L": p = (-1, 1 - r, c - 1)
                else:          p = (1 - c, 1 - r, -1)
                g.setdefault(p, []).append(i)
    return [v for v in g.values() if len(v) > 1]


CUBIES = _cubies()


def _piece_sets(state):
    """Colores de cada pieza; deben conservarse ante cualquier giro."""
    return sorted(tuple(sorted(state[list(p)])) for p in CUBIES)
