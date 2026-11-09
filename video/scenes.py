from common import *
from sim import eps_at, N_EP

# ======================= ESCENA 1 · Gancho (42 s) =======================
S1_CAPS = [
    (0.3, 4.6, 'Este robot no sabe nada del juego: ni dónde está la meta, ni qué casillas son peligrosas.'),
    (4.6, 11.5, 'Intento 1: se mueve al azar… y cae en una trampa.'),
    (11.5, 19.0, 'Intento 2, intento 3… sigue fallando. Pero cada error le deja una pista.'),
    (19.0, 26.2, 'Tras cientos de intentos, algo cambia.'),
    (26.2, 33.0, 'Intento 500: va directo a la meta y esquiva las trampas.'),
    (33.0, 41.6, '¿Cómo aprendió? Con un algoritmo de 1989 llamado Q-learning.'),
]


def s1_segments():
    segs = []
    for a, b, ep in [(4.8, 11.5, 1), (11.5, 15.3, 2), (15.3, 19.0, 3)]:
        traj = TR[ep][0]
        n = len(traj) - 1
        sd = min(0.55, (b - a - 1.3) / n)
        segs.append(dict(a=a, b=b, ep=ep, traj=traj, sd=sd, end=a + n * sd))
    return segs


S1_SEGS = s1_segments()
S1_WIN_T0, S1_WIN_SD = 26.6, 0.55
S1_WIN_END = S1_WIN_T0 + (len(PATH) - 1) * S1_WIN_SD


def s1(t):
    img, d = new()
    cs = 170
    x0, y0 = (W - 4 * cs) // 2, 140
    R = cs * 0.27
    flash = None
    ep_label, result = 1, None

    # ¿qué episodio está en pantalla?
    seg = None
    for s in S1_SEGS:
        if s['a'] <= t < s['b']:
            seg = s
    if t < S1_SEGS[0]['a']:
        grid(d, x0, y0, cs)
        bob = math.sin(t * 3) * 4
        cx, cy = cell_center(x0, y0, cs, START)
        agent(d, cx, cy + bob, R, look=(math.sin(t * 1.3), 0))
        for i, (r, c) in enumerate([(0, 3), (1, 1), (2, 3)]):
            qa = ramp(t, 1.0 + i * 0.4) * (1 - ramp(t, 4.2))
            qx, qy = cell_center(x0, y0, cs, (r, c))
            text(d, (qx + cs * 0.3, qy - cs * 0.32), '?', 44, fade(YEL, qa * 0.9), bold=True)
        ep_label = 1
    elif seg is not None:
        u = (t - seg['a']) / seg['sd']
        n = len(seg['traj']) - 1
        dt = t - seg['end']
        fl = clamp(1 - abs(dt - 0.25) / 0.6) if dt > 0 else 0
        flash = (seg['traj'][-1], fl)
        grid(d, x0, y0, cs, flash=flash)
        trail(d, x0, y0, cs, seg['traj'], u, (120, 70, 80), 6)
        pos, look = traj_pos(seg['traj'], u)
        cx, cy = cell_center(x0, y0, cs, pos)
        sc = 1 - ease(dt / 0.6) if dt > 0 else 1
        agent(d, cx, cy, R, look=look, scale=sc)
        ep_label = seg['ep']
        if dt > 0:
            result = ('✗  Cayó en una trampa', RED, ramp(dt, 0, 0.3))
    elif t < S1_WIN_T0:
        # avance rápido
        p = ease((t - 19.0) / 7.0)
        ep = int(4 + p * (N_EP - 5))
        traj = TR[ep][0]
        u = ((t - 19.0) * 14) % max(1, len(traj) - 1)
        grid(d, x0, y0, cs)
        pos, look = traj_pos(traj, u)
        cx, cy = cell_center(x0, y0, cs, pos)
        agent(d, cx, cy, R, look=look)
        ep_label = ep
        # icono de avance rápido
        fx, fy = 310, 640
        for k in range(2):
            ox = fx - 30 + k * 34
            d.polygon([(ox, fy - 22), (ox, fy + 22), (ox + 34, fy)], fill=MUTED)
        text(d, (310, 700), '× 100', 30, MUTED, bold=True)
    else:
        u = (t - S1_WIN_T0) / S1_WIN_SD
        dt = t - S1_WIN_END
        fl = clamp(1 - abs(dt - 0.3) / 0.8) if dt > 0 else 0
        grid(d, x0, y0, cs, flash=(GOAL, fl))
        trail(d, x0, y0, cs, PATH, u, YEL, 8)
        pos, look = traj_pos(PATH, u)
        cx, cy = cell_center(x0, y0, cs, pos)
        jump = -abs(math.sin(dt * 9)) * 18 * (1 - ease(dt / 1.5)) if dt > 0 else 0
        agent(d, cx, cy + jump, R, look=look)
        ep_label = N_EP
        if dt > 0:
            result = ('✓  ¡Meta en 6 pasos!', GREEN, ramp(dt, 0, 0.3))

    # paneles laterales
    text(d, (310, 430), 'INTENTO', 30, MUTED, bold=True)
    text(d, (310, 510), str(ep_label), 110, TXT, bold=True)
    if result:
        s, col, a = result
        text(d, (1610, 480), s, 38, fade(col, a), bold=True)

    # título
    ta = ramp(t, 33.0, 0.8)
    if ta > 0:
        dark = Image.new('RGB', (W, H), BG)
        img = Image.blend(img, dark, 0.86 * ta)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 400), 'Q-learning', 130, fade(TXT, ta), bold=True)
        text(d, (W // 2, 520), 'Cómo una IA aprende de sus errores', 50, fade(BLUE, ramp(t, 33.6, 0.8)))
        text(d, (W // 2, 620), 'probar  ·  equivocarse  ·  anotar qué funciona', 34, fade(MUTED, ramp(t, 34.4, 0.8)))
    captions(d, t, S1_CAPS)
    return img


# ======================= ESCENA 2 · Reglas (40 s) =======================
S2_CAPS = [
    (0.3, 8.0, 'Pongamos nombres. El tablero tiene 16 casillas: cada una es un ESTADO.'),
    (8.0, 16.0, 'En cada estado, el robot puede elegir entre 4 ACCIONES: arriba, abajo, izquierda o derecha.'),
    (16.0, 26.0, 'Tras cada acción recibe una RECOMPENSA: +10 en la meta, −10 en una trampa y −1 por cada paso.'),
    (26.0, 34.0, 'Ese −1 por paso le enseña a no dar vueltas: un camino corto suma más que uno largo.'),
    (34.0, 39.7, 'Su objetivo: conseguir la mayor recompensa total posible.'),
]
LONG_PATH = [(3, 0), (3, 1), (3, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3)]


def s2(t):
    img, d = new()
    header(d, 'Las reglas del juego', t)
    x0, y0, cs = 170, 175, 160
    R = cs * 0.27
    grid(d, x0, y0, cs, nums=ramp(t, 1.0), labels=ramp(t, 17.0, 0.6),
         hl=(START, win(t, 1.0, 8.0) * (0.6 + 0.4 * math.sin(t * 5))))
    # caminos de comparación
    pa = win(t, 26.5, 34.0, 0.6)
    if pa > 0:
        u = clamp((t - 26.5) / 3.0) * (len(LONG_PATH) - 1)
        trail(d, x0, y0, cs, LONG_PATH, u, fade((150, 150, 175), pa), 7)
        u2 = clamp((t - 29.5) / 2.0) * (len(PATH) - 1)
        trail(d, x0, y0, cs, PATH, u2, fade(GREEN, pa), 9)
    # robot (con un paso de demostración)
    if 20.0 <= t < 24.5:
        u = ramp(t, 20.0, 0.5) - ramp(t, 23.6, 0.5)
        pos = (3 - u, 0)
        look = (0, -1) if t < 23 else (0, 1)
    else:
        pos, look = START, (0, 0)
    cx, cy = cell_center(x0, y0, cs, pos)
    agent(d, cx, cy + math.sin(t * 3) * 3, R, look=look)
    if 20.4 <= t < 23.4:
        k = (t - 20.4) / 3.0
        text(d, (cx + 70, cy - 40 - k * 60), '−1', 48, fade(RED, 1 - ease((k - 0.6) / 0.4)), bold=True)
    # flechas de acciones
    aa = win(t, 8.3, 16.0)
    if aa > 0:
        p = 0.75 + 0.25 * math.sin(t * 6)
        sx, sy = cell_center(x0, y0, cs, START)
        for k in range(4):
            dx, dy = DIRS[k]
            if (dx, dy) in [(0, 1), (-1, 0)]:
                col = fade(MUTED, aa * 0.5)  # chocaría contra el borde
            else:
                col = fade(YEL, aa * p)
            arrow(d, sx + dx * cs * 0.62, sy + dy * cs * 0.62, dx, dy, cs * 0.42, 7, col)

    # tarjetas
    X, Wc = 980, 800
    card(d, X, 175, Wc, 150, 'ESTADO', 'La casilla donde está el robot. Hay 16.', YEL, ramp(t, 0.6, 0.5))
    card(d, X, 345, Wc, 150, 'ACCIÓN', 'Arriba, abajo, izquierda o derecha.', BLUE, ramp(t, 8.3, 0.5))
    card(d, X, 515, Wc, 150, 'RECOMPENSA', '+10 meta  ·  −10 trampa  ·  −1 por paso', GREEN, ramp(t, 16.3, 0.5))
    ca = ramp(t, 26.6, 0.5)
    if ca > 0:
        d.rounded_rectangle((X, 685, X + Wc, 835), 16, fill=fade(PANEL, ca))
        text(d, (X + 36, 725), 'Camino largo (10 pasos):  9 × (−1) + 10 =', 29, fade(MUTED, ca, PANEL), anchor='lm')
        text(d, (X + Wc - 36, 725), '+1', 34, fade((180, 180, 200), ca, PANEL), bold=True, anchor='rm')
        cb = ramp(t, 29.6, 0.5)
        text(d, (X + 36, 790), 'Camino corto (6 pasos):  5 × (−1) + 10 =', 29, fade(TXT, cb, PANEL), anchor='lm')
        text(d, (X + Wc - 36, 790), '+5', 34, fade(GREEN, cb, PANEL), bold=True, anchor='rm')
    oa = ramp(t, 34.2, 0.6)
    if oa > 0:
        text(d, (W // 2, 905), 'OBJETIVO:  la mayor recompensa total', 40, fade(YEL, oa), bold=True)
    captions(d, t, S2_CAPS)
    return img


# ======================= ESCENA 3 · La tabla Q (68 s) =======================
S3_CAPS = [
    (0.3, 9.0, 'La idea clave: el robot lleva una libreta con una puntuación para cada pareja (casilla, acción).'),
    (9.0, 18.0, 'Esa puntuación es el valor Q: cuánta recompensa espera conseguir si hace esa acción ahí y después juega bien.'),
    (18.0, 24.0, 'Al principio todos los valores valen 0: el robot no sabe nada.'),
    (24.0, 36.0, 'Tras unos cuantos tropiezos, por fin llega a la meta… y la acción que lo llevó ahí gana puntos.'),
    (36.0, 47.0, 'Ese aprendizaje se contagia hacia atrás: la casilla anterior descubre que lleva a una casilla buena, y así sucesivamente.'),
    (47.0, 58.0, 'Las trampas hacen lo contrario: las acciones que llevan a ellas se tiñen de rojo.'),
    (58.0, 67.7, 'Al final basta con leer la tabla: en cada casilla, elegir la acción con el valor más alto.'),
]
NONTERM = [(r, c) for r in range(4) for c in range(4) if (r, c) != GOAL and (r, c) not in TRAPS]

# tiempos del episodio en que llega a la meta por primera vez
_fg_traj = TR[FG][0]
_fg_n = len(_fg_traj) - 1
_fg_m = min(6, _fg_n)
FG_T0, FG_FAST_END, FG_SLOW_SD = 27.0, 29.5, 1.0
FG_GOAL_T = FG_FAST_END + _fg_m * FG_SLOW_SD


def fg_u(t):
    if t < FG_FAST_END:
        return (t - FG_T0) / (FG_FAST_END - FG_T0) * (_fg_n - _fg_m)
    return (_fg_n - _fg_m) + (t - FG_FAST_END) / FG_SLOW_SD


def table(d, x0, y0, Q, hl=None, best=0.0, a=1.0):
    lw, cw, rh, hh = 230, 135, 48, 54
    text(d, (x0 + 20, y0 + hh / 2), 'Casilla', 28, fade(MUTED, a), bold=True, anchor='lm')
    for k in range(4):
        text(d, (x0 + lw + k * cw + cw / 2, y0 + hh / 2), ARROWS[k], 34, fade(MUTED, a), bold=True)
    for i, (r, c) in enumerate(NONTERM):
        y = y0 + hh + i * rh
        text(d, (x0 + 20, y + rh / 2), f'{r * 4 + c + 1}', 26, fade(TXT, a), anchor='lm')
        q = Q[r][c]
        kb = int(np.argmax(q))
        for k in range(4):
            x = x0 + lw + k * cw
            d.rectangle((x + 3, y + 3, x + cw - 3, y + rh - 3), fill=fade(qcol(q[k], PANEL), a))
            text(d, (x + cw / 2, y + rh / 2), fmt(q[k]), 25, fade(TXT, a), mono=True)
            if best > 0 and k == kb:
                d.rectangle((x + 2, y + 2, x + cw - 2, y + rh - 2), outline=fade(YEL, best * a), width=4)
        if hl and hl[0] == (r, c):
            x = x0 + lw + hl[1] * cw
            d.rectangle((x + 1, y + 1, x + cw - 1, y + rh - 1), outline=fade(YEL, hl[2] * a), width=5)


def s3_cell_view(t):
    img, d = new()
    header(d, 'La tabla Q', t)
    cx, cy, S = 560, 500, 420
    x, y = cx - S / 2, cy - S / 2
    q = [1.8, -3.2, -1.0, 6.5]
    qa = ramp(t, 9.3, 0.8)
    g = 4
    tris = [
        [(x + g, y + g), (x + S - g, y + g), (cx, cy)],
        [(x + g, y + S - g), (x + S - g, y + S - g), (cx, cy)],
        [(x + g, y + g), (x + g, y + S - g), (cx, cy)],
        [(x + S - g, y + g), (x + S - g, y + S - g), (cx, cy)],
    ]
    for k in range(4):
        d.polygon(tris[k], fill=mix(CELL, qcol(q[k]), qa))
    d.line((x, y, x + S, y + S), fill=BG, width=4)
    d.line((x + S, y, x, y + S), fill=BG, width=4)
    pos = [(cx, y + S * 0.16), (cx, y + S * 0.84), (x + S * 0.18, cy), (x + S * 0.82, cy)]
    for k in range(4):
        dx, dy = DIRS[k]
        la = ramp(t, 1.5 + k * 0.4)
        if qa < 1:
            arrow(d, pos[k][0], pos[k][1], dx, dy, 70, 8, fade(MUTED, la * (1 - qa)))
        text(d, pos[k], fmt(q[k]), 46, fade(TXT, qa, CELL), bold=True)
    agent(d, cx, cy, 34, look=(math.sin(t), 0))
    ba = ramp(t, 13.5, 0.5)
    if ba > 0:
        d.rectangle((x + S * 0.66, cy - 45, x + S - 10, cy + 45), outline=fade(YEL, ba), width=5)
        text(d, (x + S + 30, cy), '← la mejor', 34, fade(YEL, ba), bold=True, anchor='lm')
    # texto derecho
    X = 1080
    text(d, (X, 300), 'Q(casilla, acción)', 64, fade(BLUE, ramp(t, 9.5, 0.6)), bold=True, anchor='lm')
    for i, line in enumerate(['= la recompensa que espera conseguir',
                              'si hace esa acción en esa casilla',
                              'y después sigue jugando bien.']):
        text(d, (X, 400 + i * 54), line, 38, fade(TXT, ramp(t, 10.2 + i * 0.5, 0.6)), anchor='lm')
    text(d, (X, 620), 'Una casilla = 4 valores, uno por acción.', 32, fade(MUTED, ramp(t, 12.0, 0.6)), anchor='lm')
    text(d, (X, 680), '(valores de ejemplo)', 28, fade(MUTED, ramp(t, 12.0, 0.6) * 0.7), anchor='lm')
    la = win(t, 1.0, 9.6)
    if la > 0:
        text(d, (X, 420), 'Una libreta con una nota', 46, fade(TXT, la), bold=True, anchor='lm')
        text(d, (X, 480), 'para cada (casilla, acción)', 46, fade(TXT, la), bold=True, anchor='lm')
    captions(d, t, S3_CAPS)
    return img


def s3_grid_view(t):
    img, d = new()
    header(d, 'La tabla Q', t)
    x0, y0, cs = 120, 190, 160
    R = cs * 0.25
    hl_t, hl_cell, flash = None, None, None
    trail_info = None
    if t < 24.0:
        Q, ep = SNAPS[0], 0
        pos, look = START, (0, 0)
    elif t < FG_T0:
        e = 1 + (FG - 1) * clamp((t - 24.0) / (FG_T0 - 24.0)) * 0.999
        ep = int(e)
        Q = SNAPS[ep]
        traj = TR[ep][0]
        u = (e - ep) * (len(traj) - 1)
        pos, look = traj_pos(traj, u)
    elif t < 36.0:
        ep = FG
        u = clamp(fg_u(t), 0, _fg_n)
        k = int(u)
        Q = (SIM['qsteps'][FG][k - 1] if k >= 1 else SNAPS[FG - 1])
        pos, look = traj_pos(_fg_traj, u)
        trail_info = (_fg_traj, u)
        if k >= 1 and t >= FG_FAST_END:
            s_prev = _fg_traj[k - 1]
            act = SIM['acts'][FG][k - 1]
            hl_t = (s_prev, act, clamp(1 - (u - k) * 1.2))
            hl_cell = (s_prev, clamp(1 - (u - k) * 1.2))
        dt = t - FG_GOAL_T
        if dt > 0:
            flash = (GOAL, clamp(1 - abs(dt - 0.3) / 0.9))
    elif t < 58.0:
        p = clamp((t - 36.0) / 21.0) ** 2.0
        e = FG + (N_EP - FG) * p
        ep = int(e)
        e2 = min(N_EP, ep + 1)
        f = e - ep
        Q = SNAPS[ep] * (1 - f) + SNAPS[e2] * f
        traj = TR[min(N_EP, ep + 1)][0]
        u = ((t - 36.0) * 12) % max(1, len(traj) - 1)
        pos, look = traj_pos(traj, u)
    else:
        Q, ep = QFINAL, N_EP
        u = clamp((t - 60.0) / 3.5) * (len(PATH) - 1)
        pos, look = traj_pos(PATH, u)
    pa = ramp(t, 58.5, 0.8)
    grid(d, x0, y0, cs, Q=Q, qa=1 - 0.45 * pa, vals=pa < 0.5, policy=QFINAL if pa > 0 else None, pa=pa,
         nums=1.0, flash=flash, hl=hl_cell)
    if trail_info:
        trail(d, x0, y0, cs, trail_info[0], trail_info[1], YEL if t >= 58 else (200, 170, 90), 6)
    cx, cy = cell_center(x0, y0, cs, pos)
    agent(d, cx, cy, R, look=look)
    text(d, (x0, 150), f'Episodio {ep}', 34, TXT, bold=True, anchor='lm')
    table(d, 1000, 172, Q, hl=hl_t, best=pa)
    if t >= 36.0 and t < 58.0:
        sx, sy = x0 + 4 * cs, 150
        text(d, (sx, sy), '▶▶ avance rápido', 26, MUTED, anchor='rm')
    return img, d


def s3(t):
    if t < 17.2:
        return s3_cell_view(t)
    g, d = s3_grid_view(t)
    captions(d, t, S3_CAPS)
    if t < 18.4:
        return blend(s3_cell_view(t), g, ease((t - 17.2) / 1.2))
    return g


# ======================= ESCENA 4 · Explorar o aprovechar (30 s) =======================
S4_CAPS = [
    (0.3, 9.0, 'Un problema: si siempre elige lo que ya conoce, puede quedarse con el primer camino que encontró.'),
    (9.0, 18.0, 'Por eso, a veces explora: con probabilidad ε (épsilon) elige una acción al azar.'),
    (18.0, 29.7, 'Al principio explora mucho; con el tiempo ε baja y aprovecha cada vez más lo que ya sabe.'),
]
# Decisiones de ejemplo con ε = 0.3: (tiempo, posición del marcador 0..1)
S4_DECISIONS = [(9.6, 0.62), (12.4, 0.18), (15.2, 0.81)]


def s4(t):
    img, d = new()
    header(d, 'Explorar o aprovechar', t)
    x0, y0, cs = 120, 200, 150
    R = cs * 0.26
    X = 820
    if t < 9.3:
        # siempre el camino largo
        cyc = (t % 4.6) / 3.8
        u = clamp(cyc) * (len(LONG_PATH) - 1)
        grid(d, x0, y0, cs)
        trail(d, x0, y0, cs, LONG_PATH, u, (150, 150, 175), 7)
        pos, look = traj_pos(LONG_PATH, u)
        cx, cy = cell_center(x0, y0, cs, pos)
        agent(d, cx, cy, R, look=look)
        a = ramp(t, 0.8)
        text(d, (X + 60, 380), 'Siempre el mismo camino…', 50, fade(TXT, a), bold=True, anchor='lm')
        text(d, (X + 60, 460), '10 pasos, cuando existe uno de 6.', 38, fade(MUTED, ramp(t, 2.0)), anchor='lm')
        text(d, (X + 60, 560), 'Si nunca prueba nada nuevo,', 38, fade(TXT, ramp(t, 3.5)), anchor='lm')
        text(d, (X + 60, 615), 'nunca descubrirá el atajo.', 38, fade(YEL, ramp(t, 4.0)), bold=True, anchor='lm')
    else:
        grid(d, x0, y0, cs, Q=QFINAL, qa=0.35)
        cx, cy = cell_center(x0, y0, cs, (2, 0))
        agent(d, cx, cy, R, look=(0, -1))
        if t < 18.4:
            # barra ε = 0.3 y marcador
            bx, by, bw, bh = X + 60, 330, 880, 70
            eps = 0.3
            d.rounded_rectangle((bx, by, bx + bw * eps, by + bh), 10, fill=(74, 52, 120))
            d.rounded_rectangle((bx + bw * eps, by, bx + bw, by + bh), 10, fill=(32, 78, 70))
            text(d, (bx + bw * eps / 2, by + bh / 2), 'ε = 0.3', 30, TXT, bold=True)
            text(d, (bx + bw * (1 + eps) / 2, by + bh / 2), '1 − ε = 0.7', 30, TXT, bold=True)
            text(d, (bx + bw * eps / 2, by + bh + 40), 'EXPLORAR', 28, PURP, bold=True)
            text(d, (bx + bw * (1 + eps) / 2, by + bh + 40), 'APROVECHAR', 28, GREEN, bold=True)
            cur = None
            for td, m in S4_DECISIONS:
                if t >= td:
                    cur = (td, m)
            if cur:
                td, m = cur
                sp = ease((t - td) / 1.1)
                mpos = (0.05 + sp * (m - 0.05 + 2)) % 1 if sp < 1 else m
                mpos = m if sp >= 1 else ((td * 7 + sp * 3.2) % 1)
                mx = bx + bw * mpos
                d.polygon([(mx, by - 6), (mx - 18, by - 36), (mx + 18, by - 36)], fill=YEL)
                if sp >= 1:
                    ra = ramp(t, td + 1.1, 0.3)
                    explore = m < 0.3
                    if explore:
                        k = [1, 3, 2][S4_DECISIONS.index(cur) % 3]
                        msg, col = 'Explora → acción al azar', PURP
                    else:
                        k = int(np.argmax(QFINAL[2][0]))
                        msg, col = 'Aprovecha → la mejor acción conocida', GREEN
                    text(d, (bx, 560), msg, 42, fade(col, ra), bold=True, anchor='lm')
                    dx, dy = DIRS[k]
                    arrow(d, cx + dx * cs * 0.62, cy + dy * cs * 0.62, dx, dy, cs * 0.45, 8, fade(col, ra))
            text(d, (bx, 660), 'Cada vez que decide, “tira un dado”.', 32, fade(MUTED, ramp(t, 10.0)), anchor='lm')
        else:
            a = ramp(t, 18.4, 0.6)
            px, py, pw, ph = X + 120, 230, 820, 520
            d.line((px, py, px, py + ph), fill=fade(LINE, a), width=3)
            d.line((px, py + ph, px + pw, py + ph), fill=fade(LINE, a), width=3)
            for v in (0, 0.5, 1.0):
                yy = py + ph - v * ph
                text(d, (px - 20, yy), f'{v:.1f}', 26, fade(MUTED, a), anchor='rm')
            for e in (1, 100, 200, 300, 400, 500):
                xx = px + (e - 1) / (N_EP - 1) * pw
                text(d, (xx, py + ph + 32), str(e), 24, fade(MUTED, a))
            text(d, (px + pw / 2, py + ph + 80), 'episodio', 28, fade(MUTED, a))
            text(d, (px - 70, py - 30), 'ε', 40, fade(PURP, a), serif_it=True)
            prog = clamp((t - 19.0) / 9.0)
            emax = 1 + prog * (N_EP - 1)
            pts = []
            for e in range(1, int(emax) + 1):
                pts.append((px + (e - 1) / (N_EP - 1) * pw, py + ph - eps_at(e) * ph))
            if len(pts) > 1:
                d.line(pts, fill=PURP, width=6, joint='curve')
            ecur = max(1, int(emax))
            mx, my = pts[-1] if pts else (px, py)
            d.ellipse((mx - 12, my - 12, mx + 12, my + 12), fill=YEL)
            text(d, (px + pw, py - 30), f'episodio {ecur}:  ε = {eps_at(ecur):.2f}', 32, fade(TXT, a), bold=True, anchor='rm')
            text(d, (px + 40, py + 40), '← casi todo al azar', 28, fade(MUTED, ramp(t, 19.5)), anchor='lm')
            if prog > 0.6:
                text(d, (px + pw, py + ph - 70), 'casi siempre aprovecha →', 28, fade(MUTED, ramp(t, 25.0)), anchor='rm')
    captions(d, t, S4_CAPS)
    return img


# ======================= ESCENA 5 · La ecuación (70 s) =======================
S5_CAPS = [
    (0.3, 10.0, 'Ahora sí, la ecuación. En palabras: valor nuevo = valor viejo + un poco × la sorpresa.'),
    (10.0, 20.0, 'La sorpresa es la diferencia entre lo que obtuvo y lo que esperaba.'),
    (20.0, 30.0, 'Lo que obtuvo: la recompensa r, más el mejor valor de la casilla siguiente, rebajado por γ (gamma).'),
    (30.0, 38.0, 'α (alfa) decide cuánto aprende de cada sorpresa; γ hace que el futuro valga un poco menos que el presente.'),
    (38.0, 52.0, 'Ejemplo con α = 0.5 y γ = 0.9. Junto a la meta: Q = 0 + 0.5 × (10 + 0.9 × 0 − 0) = 5.'),
    (52.0, 64.0, 'Un paso antes: Q = 0 + 0.5 × (−1 + 0.9 × 5 − 0) = 1.75. Así viaja el valor hacia atrás.'),
    (64.0, 69.7, 'Repetido miles de veces, los valores se asientan: 6.2 → 8 → 10.'),
]

EQ = [('Q(s,a)', TXT), (' ← ', MUTED), ('Q(s,a)', TXT), (' + ', MUTED), ('α', BLUE), (' · [ ', MUTED),
      ('r', GREEN), (' + ', GREEN), ('γ', PURP), (' · máx Q(s′,a′)', GREEN), (' − ', MUTED), ('Q(s,a)', RED),
      (' ]', MUTED)]


def toprow(d, x0, y0, cs, qvals, hl=None, ag=None, a=1.0):
    """Fila superior del tablero: (0,0) (0,1) (0,2) y la meta. qvals: dict col->valor de la acción →."""
    for c in range(4):
        x = x0 + c * cs
        box = (x + 3, y0 + 3, x + cs - 3, y0 + cs - 3)
        if c == 3:
            d.rounded_rectangle(box, 12, fill=fade((26, 92, 70), a))
            star(d, x + cs / 2, y0 + cs / 2, cs * 0.27, fade(GREEN, a))
            continue
        d.rounded_rectangle(box, 12, fill=fade(CELL, a))
        v = qvals.get(c, 0.0)
        cx, cy = x + cs / 2, y0 + cs / 2
        tri = [(x + cs - 3, y0 + 3), (x + cs - 3, y0 + cs - 3), (cx, cy)]
        d.polygon(tri, fill=fade(qcol(v), a))
        text(d, (x + cs * 0.79, cy), f'{v:.2f}'.rstrip('0').rstrip('.') if v else '0', int(cs * 0.13),
             fade(TXT, a), bold=True)
        text(d, (x + 12, y0 + 10), str(c + 1), int(cs * 0.11), fade(MUTED, a), anchor='lt')
        if hl == c:
            d.rounded_rectangle(box, 12, outline=fade(YEL, a), width=5)
    if ag is not None:
        cx, cy = x0 + ag * cs + cs * 0.38, y0 + cs / 2
        agent(d, cx, cy, cs * 0.17, look=(1, 0), a=a)


def s5(t):
    img, d = new()
    header(d, 'La ecuación', t)
    # Etapa 1: en palabras
    wa = win(t, 0.4, 20.6, 0.6)
    if wa > 0:
        rich(d, W // 2, 250, [('valor nuevo', TXT), ('  =  ', MUTED), ('valor viejo', TXT), ('  +  ', MUTED),
                              ('un poco', BLUE), ('  ×  ', MUTED), ('sorpresa', YEL)], 60, wa, bold=True)
        sa = win(t, 10.3, 20.6, 0.6)
        if sa > 0:
            rich(d, W // 2, 380, [('sorpresa', YEL), ('  =  ', MUTED), ('lo que obtuvo', GREEN), ('  −  ', MUTED),
                                  ('lo que esperaba', RED)], 52, sa, bold=True)
    # Etapa 2: símbolos
    ea = ramp(t, 20.6, 0.8)
    if ea > 0:
        spans = rich(d, W // 2, 250, EQ, 58, ea, bold=True)
        la = ramp(t, 22.5, 0.6)
        if la > 0:
            def under(i0, i1, label, col, yy=320):
                xa, xb = spans[i0][0] + 6, spans[i1][1] - 6
                d.line((xa, yy, xb, yy), fill=fade(col, la), width=4)
                d.line((xa, yy - 12, xa, yy), fill=fade(col, la), width=4)
                d.line((xb, yy - 12, xb, yy), fill=fade(col, la), width=4)
                text(d, ((xa + xb) / 2, yy + 34), label, 30, fade(col, la), bold=True)
            under(6, 9, 'lo que obtuvo', GREEN)
            under(11, 11, 'lo que esperaba', RED)
            under(4, 4, 'un poco', BLUE)
            under(2, 2, 'valor viejo', MUTED)
    # α y γ
    ca = win(t, 30.3, 38.4, 0.5)
    if ca > 0:
        d.rounded_rectangle((260, 470, 940, 640), 16, fill=fade(PANEL, ca))
        text(d, (300, 520), 'α  tasa de aprendizaje', 36, fade(BLUE, ca, PANEL), bold=True, anchor='lm')
        text(d, (300, 585), '0.5 → aprende la mitad de cada sorpresa', 29, fade(TXT, ca, PANEL), anchor='lm')
        d.rounded_rectangle((980, 470, 1660, 640), 16, fill=fade(PANEL, ramp(t, 31.5) * ca))
        text(d, (1020, 520), 'γ  factor de descuento', 36, fade(PURP, ramp(t, 31.5) * ca, PANEL), bold=True, anchor='lm')
        text(d, (1020, 585), '0.9 → el futuro vale el 90 %', 29, fade(TXT, ramp(t, 31.5) * ca, PANEL), anchor='lm')
    # ejemplos
    x0, y0, cs = 150, 520, 160
    if t >= 38.4:
        a = ramp(t, 38.4, 0.6)
        q = {}
        if t >= 44.0:
            q[2] = 5.0 * ramp(t, 44.0, 0.8)
        if t >= 58.0:
            q[1] = 1.75 * ramp(t, 58.0, 0.8)
        if t >= 64.0:
            k = ramp(t, 64.0, 1.5)
            q = {0: 6.2 * k, 1: 1.75 + (8.0 - 1.75) * k, 2: 5.0 + 5.0 * k}
        hl = 2 if t < 52 else (1 if t < 64 else None)
        ag = 2 if t < 52 else (1 if t < 64 else None)
        toprow(d, x0, y0, cs, q, hl=hl, ag=ag, a=a)
        text(d, (x0, y0 - 40), 'Fila de arriba del tablero  ·  acción →', 28, fade(MUTED, a), anchor='lm')
        X = 880
        e1 = win(t, 38.6, 52.0, 0.5)
        if e1 > 0:
            text(d, (X, 540), 'Casilla 3, junto a la meta:', 34, fade(MUTED, e1), anchor='lm')
            rich(d, X, 610, [('Q = 0 + 0.5 × (', TXT), ('10', GREEN), (' + 0.9 × ', TXT), ('0', GREEN),
                             (' − ', TXT), ('0', RED), (')', TXT)], 42, e1 * ramp(t, 40.0), anchor_center=False, bold=True)
            text(d, (X, 690), 'Q = 5', 56, fade(GREEN, e1 * ramp(t, 43.5)), bold=True, anchor='lm')
            text(d, (X, 760), 'La meta termina la partida: no hay futuro, máx Q = 0.', 28,
                 fade(MUTED, e1 * ramp(t, 46.0)), anchor='lm')
        e2 = win(t, 52.3, 64.0, 0.5)
        if e2 > 0:
            text(d, (X, 540), 'Casilla 2, un paso antes:', 34, fade(MUTED, e2), anchor='lm')
            rich(d, X, 610, [('Q = 0 + 0.5 × (', TXT), ('−1', GREEN), (' + 0.9 × ', TXT), ('5', GREEN),
                             (' − ', TXT), ('0', RED), (')', TXT)], 42, e2 * ramp(t, 53.5), anchor_center=False, bold=True)
            text(d, (X, 690), 'Q = 0.5 × 3.5 = 1.75', 56, fade(GREEN, e2 * ramp(t, 57.5)), bold=True, anchor='lm')
            text(d, (X, 760), 'El 5 de la casilla siguiente “viaja” hacia atrás.', 28,
                 fade(MUTED, e2 * ramp(t, 59.0)), anchor='lm')
        e3 = ramp(t, 64.3, 0.6)
        if e3 > 0:
            text(d, (X, 560), 'Tras 500 episodios:', 34, fade(MUTED, e3), anchor='lm')
            rich(d, X, 650, [('6.2', GREEN), ('  →  ', MUTED), ('8.0', GREEN), ('  →  ', MUTED), ('10', GREEN)],
                 64, e3, anchor_center=False, bold=True)
            text(d, (X, 740), 'Cuanto más cerca de la meta, más vale.', 30, fade(TXT, ramp(t, 65.5)), anchor='lm')
    captions(d, t, S5_CAPS)
    return img


# ======================= ESCENA 6 · Deep Q-Learning (40 s) =======================
S6_CAPS = [
    (0.3, 8.0, 'Este tablero cabe en una tabla de 64 números. ¿Y un videojuego de verdad?'),
    (8.0, 16.0, 'Una pantalla de Atari tiene 210 × 160 píxeles: el número de pantallas posibles tiene más de 70 000 cifras.'),
    (16.0, 26.0, 'La solución: cambiar la tabla por una red neuronal que mira la pantalla y estima un valor Q por cada acción.'),
    (26.0, 34.0, 'Eso es Deep Q-Learning. En 2015, DeepMind usó una sola red así para aprender 49 juegos de Atari a partir de los píxeles.'),
    (34.0, 39.7, 'La ecuación de antes sigue siendo el corazón: ahora la red aprende a acercarse a ese objetivo.'),
]
DIGITS = int(33600 * math.log10(128)) + 1
_rng = np.random.default_rng(7)
PIX = np.zeros((42, 32, 3), dtype=np.uint8)
PIX[:] = (12, 14, 40)
for yy in range(30, 42):
    PIX[yy, :] = (40, 90, 60) if yy % 2 else (36, 80, 54)
for _ in range(9):
    x, y = _rng.integers(0, 30), _rng.integers(4, 26)
    PIX[y:y + 2, x:x + 3] = _rng.choice([(230, 90, 90), (90, 160, 240), (240, 200, 80), (180, 120, 230)])
PIX[27:30, 14:17] = (255, 209, 102)
import sys as _sys
_sys.set_int_max_str_digits(0)
DIGIT_STR = str(2 ** 235200)[:400]  # 128^33600 = 2^235200, primeras cifras reales


def pixel_screen(img, x, y, w, h, t, a=1.0):
    arr = PIX.copy()
    sh = int(t * 6) % 32
    arr[4:26] = np.roll(arr[4:26], sh, axis=1)
    im = Image.fromarray(arr).resize((int(w), int(h)), Image.NEAREST)
    if a < 1:
        im = Image.blend(Image.new('RGB', im.size, BG), im, a)
    img.paste(im, (int(x), int(y)))


def s6(t):
    img, d = new()
    header(d, 'Deep Q-Learning', t)
    # parte 1: 64 números
    p1 = win(t, 0.2, 8.4, 0.5)
    if p1 > 0:
        grid(d, 260, 250, 110, Q=QFINAL, qa=0.8, a=p1)
        text(d, (1200, 400), '16 casillas × 4 acciones', 52, fade(TXT, p1), bold=True)
        text(d, (1200, 500), '= 64 números', 72, fade(GREEN, p1 * ramp(t, 2.0)), bold=True)
        text(d, (1200, 600), 'Caben en una hoja de cuaderno.', 34, fade(MUTED, p1 * ramp(t, 3.5)))
    # parte 2: pantalla de Atari
    p2 = win(t, 8.4, 16.4, 0.5)
    if p2 > 0:
        pixel_screen(img, 220, 200, 384, 504, t, p2)
        text(d, (412, 740), '210 × 160 píxeles · 128 colores', 28, fade(MUTED, p2))
        X = 780
        text(d, (X, 260), 'Pantallas posibles:', 40, fade(TXT, p2), anchor='lm')
        text(d, (X, 340), '128 ^ (210 × 160)', 60, fade(YEL, p2 * ramp(t, 9.3)), bold=True, anchor='lm')
        da = p2 * ramp(t, 10.5)
        if da > 0:
            n = int(clamp((t - 10.5) / 4.0) * 360)
            s = DIGIT_STR[:n]
            for i in range(0, len(s), 40):
                text(d, (X, 430 + (i // 40) * 38), s[i:i + 40], 30, fade(MUTED, da), mono=True, anchor='lm')
            text(d, (X, 820), f'… un número de {DIGITS:,} cifras'.replace(',', ' '), 40,
                 fade(RED, p2 * ramp(t, 13.0)), bold=True, anchor='lm')
    # parte 3: red neuronal
    p3 = ramp(t, 16.4, 0.6)
    if p3 > 0:
        pixel_screen(img, 110, 260, 240, 315, t, p3)
        text(d, (230, 610), 'estado (píxeles)', 28, fade(MUTED, p3))
        layers = [(520, 7), (740, 7), (960, 5)]
        outs_x = 1200
        ys = lambda n: [418 + (i - (n - 1) / 2) * 68 for i in range(n)]
        out_y = [418 + (i - 1.5) * 110 for i in range(4)]
        cols = [(360, [418 + (i - 2) * 70 for i in range(5)])] + [(x, ys(n)) for x, n in layers] + [(outs_x, out_y)]
        for li in range(len(cols) - 1):
            xa, ya = cols[li]
            xb, yb = cols[li + 1]
            for i, y1 in enumerate(ya):
                for j, y2 in enumerate(yb):
                    d.line((xa, y1, xb, y2), fill=fade(LINE, p3 * 0.8), width=2)
        # pulsos
        for li in range(len(cols) - 1):
            xa, ya = cols[li]
            xb, yb = cols[li + 1]
            ph = (t * 0.9 - li * 0.25) % 1
            for i, y1 in enumerate(ya):
                j = (i * 3 + int(t * 2)) % len(yb)
                px, py = xa + (xb - xa) * ph, y1 + (yb[j] - y1) * ph
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=fade(BLUE, p3 * 0.9))
        for x, yy in cols[1:-1]:
            for y in yy:
                d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=fade(PANEL, p3), outline=fade(BLUE, p3), width=3)
        text(d, (740, 160), 'red neuronal', 30, fade(BLUE, p3), bold=True)
        # salidas
        vals = [2.1 + math.sin(t * 1.3) * 0.8, -0.6 + math.sin(t * 0.9 + 1) * 0.6,
                1.0 + math.sin(t * 1.1 + 2) * 0.7, 4.2 + math.sin(t * 0.7 + 3) * 0.5]
        kb = int(np.argmax(vals))
        for k, y in enumerate(out_y):
            col = YEL if k == kb else TXT
            d.ellipse((outs_x - 20, y - 20, outs_x + 20, y + 20), fill=fade(PANEL, p3), outline=fade(col, p3), width=3)
            text(d, (outs_x + 40, y), f'Q( {ARROWS[k]} )', 32, fade(col, p3), bold=True, anchor='lm')
            bw = max(0, vals[k]) * 70
            d.rounded_rectangle((outs_x + 190, y - 16, outs_x + 190 + max(6, bw), y + 16), 6,
                                fill=fade(GREEN if vals[k] > 0 else RED, p3))
            text(d, (outs_x + 210 + bw, y), f'{vals[k]:.1f}'.replace('-', '−'), 28, fade(TXT, p3), anchor='lm')
        text(d, (outs_x + 200, 160), 'un valor Q por acción', 30, fade(MUTED, p3), bold=True)
        ba = win(t, 26.3, 34.2, 0.5)
        if ba > 0:
            d.rounded_rectangle((460, 780, 1460, 860), 40, fill=fade(PANEL, ba))
            text(d, (960, 820), 'DQN · DeepMind · 2015 · 49 juegos de Atari', 34, fade(YEL, ba, PANEL), bold=True)
        ea = ramp(t, 34.3, 0.6)
        if ea > 0:
            rich(d, 960, 800, [('objetivo = ', TXT), ('r', GREEN), (' + ', TXT), ('γ', PURP),
                               (' · máx Q(s′, a′)', GREEN)], 42, ea, bold=True)
            text(d, (960, 875), 'la red se ajusta para que Q(s, a) se acerque al objetivo', 30, fade(MUTED, ea))
    captions(d, t, S6_CAPS)
    return img


# ======================= ESCENA 7 · Cierre (10 s) =======================
S7_CAPS = [
    (0.3, 9.8, 'Q-learning en una frase: probar, equivocarse y anotar qué funciona. ¿Te animas a programar este tablero?'),
]


def s7(t):
    img, d = new()
    header(d, 'Resumen', t)
    items = [
        (YEL, 'Valor Q', 'cuánto vale hacer una acción en una casilla'),
        (GREEN, 'Actualización', 'Q ← Q + α · (lo que obtuvo − lo que esperaba)'),
        (BLUE, 'Deep Q-Learning', 'una red neuronal en lugar de la tabla'),
    ]
    for i, (col, a_, b_) in enumerate(items):
        a = ramp(t, 0.5 + i * 0.8, 0.5)
        y = 260 + i * 150
        d.ellipse((200, y - 30, 260, y + 30), fill=fade(col, a))
        text(d, (230, y), str(i + 1), 34, fade(BG, a, col), bold=True)
        text(d, (300, y - 22), a_, 40, fade(col, a), bold=True, anchor='lm')
        text(d, (300, y + 28), b_, 34, fade(TXT, a), anchor='lm')
    grid(d, 1340, 230, 100, Q=QFINAL, qa=0.5, policy=QFINAL, pa=ramp(t, 3.0, 0.8))
    text(d, (W // 2, 810), 'Reto: programa este tablero de 4 × 4. Son solo 64 números.', 40,
         fade(YEL, ramp(t, 4.0, 0.6)), bold=True)
    captions(d, t, S7_CAPS)
    return img


SCENES = [(s1, 42), (s2, 40), (s3, 68), (s4, 30), (s5, 70), (s6, 40), (s7, 10)]
