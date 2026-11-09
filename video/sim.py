import os
"""Q-learning en un tablero 4x4: genera los datos reales que usa el vídeo."""
import numpy as np
import pickle

R, C = 4, 4
START = (3, 0)
GOAL = (0, 3)
TRAPS = {(1, 1), (2, 3)}
ACTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # arriba, abajo, izquierda, derecha
ALPHA, GAMMA = 0.5, 0.9
N_EP = 500


def step(s, a):
    r, c = s
    dr, dc = ACTS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < R and 0 <= nc < C):
        nr, nc = r, c
    ns = (nr, nc)
    if ns == GOAL:
        return ns, 10, True
    if ns in TRAPS:
        return ns, -10, True
    return ns, -1, False


def eps_at(ep):
    return max(0.05, 0.99 ** (ep - 1))


def run(seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((R, C, 4))
    snaps = {0: Q.copy()}
    trajs, qsteps, acts = {}, {}, {}
    for ep in range(1, N_EP + 1):
        eps = eps_at(ep)
        s = START
        traj, qs, aa = [s], [], []
        out = None
        for _ in range(60):
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                q = Q[s]
                a = int(rng.choice(np.flatnonzero(q == q.max())))
            ns, rw, done = step(s, a)
            target = rw + (0 if done else GAMMA * Q[ns].max())
            Q[s][a] += ALPHA * (target - Q[s][a])
            traj.append(ns); qs.append(Q.copy()); aa.append(a)
            s = ns
            if done:
                out = 'goal' if rw > 0 else 'trap'
                break
        trajs[ep] = (traj, out)
        qsteps[ep] = qs
        acts[ep] = aa
        snaps[ep] = Q.copy()
    return Q, snaps, trajs, qsteps, acts


def greedy_path(Q):
    s, path = START, [START]
    for _ in range(20):
        a = int(np.argmax(Q[s]))
        s, _, done = step(s, a)
        path.append(s)
        if done:
            break
    return path


def ok(seed):
    Q, snaps, trajs, qsteps, acts = run(seed)
    for ep in (1, 2, 3):
        traj, out = trajs[ep]
        if out != 'trap' or not (4 <= len(traj) - 1 <= 11):
            return None
    fg = next(e for e in trajs if trajs[e][1] == 'goal')
    if not (5 <= fg <= 15) or len(trajs[fg][0]) - 1 > 30:
        return None
    p = greedy_path(Q)
    if p[-1] != GOAL or len(p) - 1 != 6:
        return None
    return dict(Q=Q, snaps=snaps, trajs=trajs, qsteps=qsteps, acts=acts, fg=fg, path=p, seed=seed)


if __name__ == '__main__':
    for seed in range(5000):
        res = ok(seed)
        if res:
            break
    print('seed', res['seed'], 'first goal ep', res['fg'], 'path', res['path'])
    for ep in (1, 2, 3, res['fg']):
        print(ep, res['trajs'][ep])
    print(np.round(res['Q'], 1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.pkl'), 'wb') as f:
        pickle.dump(res, f)
