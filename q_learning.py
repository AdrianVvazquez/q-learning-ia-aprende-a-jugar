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
import numbers
import re
import sys

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

MIN_EPISODIOS, MAX_EPISODIOS = 1, 100_000


def paso(estado, accion):
    """Aplica una acción. Devuelve (estado_siguiente, recompensa, terminado)."""
    if not 0 <= accion < len(ACCIONES):
        raise ValueError(f'acción no válida: {accion}')
    if estado == META or estado in TRAMPAS:
        raise ValueError(f'el episodio ya terminó en {estado}')
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
    return min(1.0, max(eps_min, decaimiento ** (episodio - 1)))  # no puede pasar de 1


def elegir_accion(Q, estado, eps, rng):
    """Política épsilon-voraz: explora con probabilidad eps, si no aprovecha."""
    if rng.random() < eps:
        return int(rng.integers(len(ACCIONES)))
    q = Q[estado]
    mejores = np.flatnonzero(q == q.max())  # desempate al azar
    return int(rng.choice(mejores))


def entrenar(episodios=500, alpha=0.5, gamma=0.9, max_pasos=60, semilla=5, verbose=True):
    for nombre, valor in (('episodios', episodios), ('max_pasos', max_pasos)):
        if isinstance(valor, bool) or not isinstance(valor, numbers.Integral):
            raise TypeError(f'{nombre} debe ser un número entero, no {type(valor).__name__}')
    if not MIN_EPISODIOS <= episodios <= MAX_EPISODIOS:
        raise ValueError(f'episodios debe estar entre {MIN_EPISODIOS} y {MAX_EPISODIOS} '
                         f'(no {episodios!r})')
    if max_pasos < 1:
        raise ValueError(f'max_pasos debe ser al menos 1 (no {max_pasos!r})')
    for nombre, valor in (('alpha', alpha), ('gamma', gamma)):
        if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
            raise TypeError(f'{nombre} debe ser un número, no {type(valor).__name__}')
    # Las comparaciones con nan son siempre falsas, así que nan también se rechaza.
    if not 0 < alpha <= 1:
        raise ValueError(f'alpha debe ser mayor que 0 y como mucho 1 (no {alpha!r})')
    if not 0 <= gamma <= 1:
        raise ValueError(f'gamma debe estar entre 0 y 1 (no {gamma!r})')
    # None es válido: NumPy elige entonces una semilla aleatoria.
    if semilla is not None:
        if isinstance(semilla, bool) or not isinstance(semilla, numbers.Integral):
            raise TypeError(f'semilla debe ser un número entero, no {type(semilla).__name__}')
        if semilla < 0:
            raise ValueError(f'semilla debe ser 0 o mayor (no {semilla!r})')

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


def numero_entero(valor):
    if not isinstance(valor, str):
        raise TypeError(f'se esperaba texto, no {type(valor).__name__}')
    # Solo dígitos 0-9: int() también aceptaría '1_000', '٥٠٠' o '５００'.
    if not re.fullmatch(r'\s*[+-]?[0-9]+\s*', valor, flags=re.ASCII):
        raise argparse.ArgumentTypeError(f'debe ser un número entero, por ejemplo 500 (no {valor!r})')
    return int(valor)


def episodios_validos(valor):
    episodios = numero_entero(valor)
    if episodios < MIN_EPISODIOS:
        raise argparse.ArgumentTypeError(f'el mínimo es {MIN_EPISODIOS}')
    if episodios > MAX_EPISODIOS:
        raise argparse.ArgumentTypeError(f'el máximo es {MAX_EPISODIOS}')
    return episodios


def numero_decimal(valor):
    if not isinstance(valor, str):
        raise TypeError(f'se esperaba texto, no {type(valor).__name__}')
    # Solo dígitos 0-9 y punto decimal: float() también aceptaría 'nan', 'inf' o '٠.٥'.
    if not re.fullmatch(r'\s*[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?\s*', valor,
                        flags=re.ASCII):
        raise argparse.ArgumentTypeError(f'debe ser un número, por ejemplo 0.5 (no {valor!r})')
    return float(valor)


def alpha_valido(valor):
    alpha = numero_decimal(valor)
    if not 0 < alpha <= 1:
        raise argparse.ArgumentTypeError('debe ser mayor que 0 y como mucho 1')
    return alpha


def gamma_valido(valor):
    gamma = numero_decimal(valor)
    if not 0 <= gamma <= 1:
        raise argparse.ArgumentTypeError('debe estar entre 0 y 1')
    return gamma


def semilla_valida(valor):
    semilla = numero_entero(valor)
    if semilla < 0:
        raise argparse.ArgumentTypeError('debe ser 0 o mayor')
    return semilla


if __name__ == '__main__':
    # La consola de Windows usa cp1252 al redirigir la salida y no admite ε ni las flechas.
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='Q-learning tabular en un tablero de 4x4')
    p.add_argument('--episodios', type=episodios_validos, default=500)
    p.add_argument('--alpha', type=alpha_valido, default=0.5, help='tasa de aprendizaje (0, 1]')
    p.add_argument('--gamma', type=gamma_valido, default=0.9, help='factor de descuento [0, 1]')
    p.add_argument('--semilla', type=semilla_valida, default=5, help='entero, 0 o mayor')
    args = p.parse_args()

    Q, _ = entrenar(args.episodios, args.alpha, args.gamma, semilla=args.semilla)
    print('\nTabla Q aprendida:')
    imprimir_tabla(Q)
    print('\nPolítica (mejor acción por casilla):')
    imprimir_politica(Q)
    camino = camino_voraz(Q)
    print(f'\nCamino aprendido ({len(camino) - 1} pasos): ' + ' → '.join(str(s) for s in camino))
