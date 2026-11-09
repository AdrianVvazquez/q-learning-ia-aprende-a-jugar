"""
Q-learning tabular en un tablero de 4x4.

El agente empieza abajo a la izquierda y debe llegar a la meta (arriba a la
derecha) sin caer en las trampas. Aprende solo, por ensayo y error, con la
regla de actualización de Q-learning:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Uso:
    python q_learning.py                 # 500 episodios, semilla 5
    python q_learning.py --episodios 1000 --semilla 42
"""
import argparse

import numpy as np

# ---------------- Entorno ----------------
FILAS, COLUMNAS = 4, 4
INICIO = (3, 0)
META = (0, 3)
TRAMPAS = {(1, 1), (2, 3)}

ACCIONES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # arriba, abajo, izquierda, derecha
FLECHAS = ['↑', '↓', '←', '→']

RECOMPENSA_META = 10
RECOMPENSA_TRAMPA = -10
RECOMPENSA_PASO = -1


def paso(estado, accion):
    """Aplica una acción. Devuelve (estado_siguiente, recompensa, terminado)."""
    f, c = estado
    df, dc = ACCIONES[accion]
    nf, nc = f + df, c + dc
    if not (0 <= nf < FILAS and 0 <= nc < COLUMNAS):  # chocar con el borde: se queda quieto
        nf, nc = f, c
    siguiente = (nf, nc)
    if siguiente == META:
        return siguiente, RECOMPENSA_META, True
    if siguiente in TRAMPAS:
        return siguiente, RECOMPENSA_TRAMPA, True
    return siguiente, RECOMPENSA_PASO, False


# ---------------- Agente ----------------
def epsilon(episodio, eps_min=0.05, decaimiento=0.99):
    """Exploración: empieza en 1 (todo al azar) y baja hasta eps_min."""
    return max(eps_min, decaimiento ** (episodio - 1))


def elegir_accion(Q, estado, eps, rng):
    """Política épsilon-voraz: explora con probabilidad eps, si no aprovecha."""
    if rng.random() < eps:
        return int(rng.integers(len(ACCIONES)))
    q = Q[estado]
    mejores = np.flatnonzero(q == q.max())  # desempate al azar
    return int(rng.choice(mejores))


def entrenar(episodios=500, alpha=0.5, gamma=0.9, max_pasos=60, semilla=5, verbose=True):
    rng = np.random.default_rng(semilla)
    Q = np.zeros((FILAS, COLUMNAS, len(ACCIONES)))
    resultados = []
    for ep in range(1, episodios + 1):
        eps = epsilon(ep)
        estado, total, final = INICIO, 0, 'límite de pasos'
        for _ in range(max_pasos):
            a = elegir_accion(Q, estado, eps, rng)
            siguiente, r, terminado = paso(estado, a)

            # --- regla de Q-learning ---
            futuro = 0.0 if terminado else Q[siguiente].max()
            objetivo = r + gamma * futuro            # lo que obtuvo
            sorpresa = objetivo - Q[estado][a]        # lo que obtuvo - lo que esperaba
            Q[estado][a] += alpha * sorpresa

            total += r
            estado = siguiente
            if terminado:
                final = 'meta' if r > 0 else 'trampa'
                break
        resultados.append((final, total))
        if verbose and (ep <= 5 or ep % 100 == 0):
            print(f'Episodio {ep:4d} | ε = {eps:.2f} | {final:<6} | recompensa total {total:+d}')
    return Q, resultados


def camino_voraz(Q, max_pasos=20):
    estado, camino = INICIO, [INICIO]
    for _ in range(max_pasos):
        estado, _, terminado = paso(estado, int(np.argmax(Q[estado])))
        camino.append(estado)
        if terminado:
            break
    return camino


def imprimir_politica(Q):
    for f in range(FILAS):
        fila = []
        for c in range(COLUMNAS):
            if (f, c) == META:
                fila.append('★')
            elif (f, c) in TRAMPAS:
                fila.append('X')
            else:
                fila.append(FLECHAS[int(np.argmax(Q[f, c]))])
        print('  ' + ' '.join(fila))


def imprimir_tabla(Q):
    print('Casilla     ' + '  '.join(f'{fl:>6}' for fl in FLECHAS))
    for f in range(FILAS):
        for c in range(COLUMNAS):
            if (f, c) == META or (f, c) in TRAMPAS:
                continue
            valores = '  '.join(f'{round(v, 1) + 0.0:6.1f}' for v in Q[f, c])
            print(f'{f * COLUMNAS + c + 1:>7}     {valores}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Q-learning tabular en un tablero de 4x4')
    p.add_argument('--episodios', type=int, default=500)
    p.add_argument('--alpha', type=float, default=0.5, help='tasa de aprendizaje')
    p.add_argument('--gamma', type=float, default=0.9, help='factor de descuento')
    p.add_argument('--semilla', type=int, default=5)
    args = p.parse_args()

    Q, _ = entrenar(args.episodios, args.alpha, args.gamma, semilla=args.semilla)
    print('\nTabla Q aprendida:')
    imprimir_tabla(Q)
    print('\nPolítica (mejor acción por casilla):')
    imprimir_politica(Q)
    camino = camino_voraz(Q)
    print(f'\nCamino aprendido ({len(camino) - 1} pasos): ' + ' → '.join(str(s) for s in camino))
