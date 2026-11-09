import os
import math
import pickle
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim import START, GOAL, TRAPS

W, H, FPS = 1920, 1080, 30
BG = (14, 19, 32)
PANEL = (24, 32, 51)
CELL = (31, 41, 66)
LINE = (48, 60, 92)
BAND = (9, 12, 21)
TXT = (236, 240, 247)
MUTED = (140, 150, 173)
GREEN = (61, 220, 151)
RED = (255, 92, 98)
YEL = (255, 209, 102)
BLUE = (86, 204, 242)
PURP = (184, 148, 255)

FD = '/usr/share/fonts/truetype/dejavu/'

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.pkl'), 'rb') as _f:
    SIM = pickle.load(_f)
TR = SIM['trajs']
SNAPS = SIM['snaps']
QFINAL = SIM['Q']
FG = SIM['fg']
PATH = SIM['path']
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (dx,dy) por acción ↑ ↓ ← →
ARROWS = ['↑', '↓', '←', '→']


@lru_cache(None)
def F(size, bold=False, mono=False, serif_it=False):
    if serif_it:
        n = 'DejaVuSerif-Italic.ttf'
    elif mono:
        n = 'DejaVuSansMono-Bold.ttf' if bold else 'DejaVuSansMono.ttf'
    else:
        n = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(FD + n, int(size))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def mix(a, b, t):
    t = clamp(t)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def fade(c, a, bg=BG):
    return mix(bg, c, a)


def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ramp(t, t0, d=0.4):
    return ease((t - t0) / d)


def win(t, a, b, d=0.4):
    """1 dentro de [a,b] con entradas y salidas suaves."""
    return min(ramp(t, a, d), 1 - ramp(t, b - d, d))


def new():
    img = Image.new('RGB', (W, H), BG)
    return img, ImageDraw.Draw(img)


def text(d, xy, s, size, fill, bold=False, anchor='mm', mono=False, serif_it=False):
    d.text(xy, s, font=F(size, bold, mono, serif_it), fill=fill, anchor=anchor)


def tw(d, s, size, bold=False, mono=False):
    return d.textlength(s, font=F(size, bold, mono))


def wrap(d, s, size, maxw, bold=False):
    words, lines, cur = s.split(' '), [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        if tw(d, test, size, bold) <= maxw:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def header(d, s, t):
    a = ramp(t, 0.2, 0.5)
    text(d, (80, 72), s.upper(), 26, fade(MUTED, a), bold=True, anchor='lm')
    d.line((80, 96, 80 + tw(d, s.upper(), 26, True), 96), fill=fade(BLUE, a), width=3)


def captions(d, t, caps):
    d.rectangle((0, 950, W, H), fill=BAND)
    for a, b, s in caps:
        if a <= t < b:
            al = win(t, a, b, 0.3)
            lines = wrap(d, s, 36, 1720)
            y0 = 1015 - (len(lines) - 1) * 23
            for i, line in enumerate(lines):
                text(d, (W // 2, y0 + i * 46), line, 36, fade(TXT, al, BAND))


def blend(a, b, t):
    if t <= 0:
        return a
    if t >= 1:
        return b
    return Image.blend(a, b, t)


# ---------- dibujo del tablero ----------

def fmt(v, nd=1):
    s = f'{v:.{nd}f}'
    if float(s) == 0:
        s = s.lstrip('-')
    return s.replace('-', '−')


def qcol(v, base=CELL):
    if abs(v) < 0.05:
        return base
    t = clamp(abs(v) / 10) ** 0.6
    return mix(base, GREEN if v > 0 else RED, 0.15 + 0.75 * t)


def star(d, cx, cy, R, fill):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = R if i % 2 == 0 else R * 0.45
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    d.polygon(pts, fill=fill)


def arrow(d, x, y, dx, dy, L, wdt, col):
    x1, y1 = x - dx * L * 0.45, y - dy * L * 0.45
    x2, y2 = x + dx * L * 0.5, y + dy * L * 0.5
    hx, hy = x2 - dx * L * 0.32, y2 - dy * L * 0.32
    d.line((x1, y1, hx, hy), fill=col, width=wdt)
    px, py = -dy, dx
    hw = L * 0.22
    d.polygon([(x2, y2), (hx + px * hw, hy + py * hw), (hx - px * hw, hy - py * hw)], fill=col)


def grid(d, x0, y0, cs, Q=None, qa=1.0, vals=False, policy=None, pa=0.0, nums=0.0,
         labels=0.0, a=1.0, flash=None, hl=None):
    g = 3
    for r in range(4):
        for c in range(4):
            x, y = x0 + c * cs, y0 + r * cs
            box = (x + g, y + g, x + cs - g, y + cs - g)
            cx, cy = x + cs / 2, y + cs / 2
            fl = flash[1] if flash and flash[0] == (r, c) else 0
            if (r, c) == GOAL:
                col = mix((26, 92, 70), GREEN, fl * 0.7)
                d.rounded_rectangle(box, 12, fill=fade(col, a))
                star(d, cx, cy - (cs * 0.08 if labels > 0 else 0) * labels, cs * 0.27, fade(mix(GREEN, (255, 255, 255), fl), a))
                if labels > 0:
                    text(d, (cx, cy + cs * 0.3), '+10', int(cs * 0.17), fade(TXT, a * labels, col), bold=True)
            elif (r, c) in TRAPS:
                col = mix((84, 28, 38), RED, fl * 0.7)
                d.rounded_rectangle(box, 12, fill=fade(col, a))
                rr = cs * 0.28
                oy = -cs * 0.07 * labels
                d.ellipse((cx - rr, cy - rr + oy, cx + rr, cy + rr + oy), fill=fade((44, 12, 20), a))
                rr2 = rr * 0.55
                d.ellipse((cx - rr2, cy - rr2 + oy, cx + rr2, cy + rr2 + oy), fill=fade((22, 6, 10), a))
                if labels > 0:
                    text(d, (cx, cy + cs * 0.33), '−10', int(cs * 0.17), fade(TXT, a * labels, col), bold=True)
            else:
                if Q is not None and qa > 0:
                    q = Q[r][c]
                    tris = [
                        [(x + g, y + g), (x + cs - g, y + g), (cx, cy)],
                        [(x + g, y + cs - g), (x + cs - g, y + cs - g), (cx, cy)],
                        [(x + g, y + g), (x + g, y + cs - g), (cx, cy)],
                        [(x + cs - g, y + g), (x + cs - g, y + cs - g), (cx, cy)],
                    ]
                    for k in range(4):
                        d.polygon(tris[k], fill=fade(mix(CELL, qcol(q[k]), qa), a))
                    d.line((x + g, y + g, x + cs - g, y + cs - g), fill=BG, width=2)
                    d.line((x + cs - g, y + g, x + g, y + cs - g), fill=BG, width=2)
                    if vals:
                        pos = [(cx, y + cs * 0.17), (cx, y + cs * 0.83), (x + cs * 0.2, cy), (x + cs * 0.8, cy)]
                        for k in range(4):
                            text(d, pos[k], fmt(q[k]), int(cs * 0.115),
                                 fade(TXT, a * qa * 0.95), bold=True)
                else:
                    d.rounded_rectangle(box, 12, fill=fade(CELL, a))
                if nums > 0:
                    text(d, (x + 12, y + 10), str(r * 4 + c + 1), int(cs * 0.11), fade(MUTED, nums * a), anchor='lt')
                if policy is not None and pa > 0:
                    k = int(np.argmax(policy[r][c]))
                    dx, dy = DIRS[k]
                    arrow(d, cx, cy, dx, dy, cs * 0.55, max(4, int(cs * 0.06)), fade(YEL, pa * a))
            if hl and hl[0] == (r, c) and hl[1] > 0:
                d.rounded_rectangle(box, 12, outline=fade(YEL, hl[1] * a), width=5)


def agent(d, x, y, R, look=(0, 0), a=1.0, scale=1.0, col=YEL):
    R *= scale
    if R < 1:
        return
    c = fade(col, a)
    d.line((x, y - R, x, y - R * 1.45), fill=c, width=max(2, int(R * 0.1)))
    rr = R * 0.14
    d.ellipse((x - rr, y - R * 1.45 - rr, x + rr, y - R * 1.45 + rr), fill=c)
    d.ellipse((x - R, y - R, x + R, y + R), fill=c)
    for ex in (-0.36, 0.36):
        ecx = x + ex * R + look[0] * R * 0.12
        ecy = y - 0.08 * R + look[1] * R * 0.12
        er = R * 0.22
        d.ellipse((ecx - er, ecy - er, ecx + er, ecy + er), fill=fade((255, 255, 255), a))
        pr = R * 0.11
        pcx, pcy = ecx + look[0] * R * 0.08, ecy + look[1] * R * 0.08
        d.ellipse((pcx - pr, pcy - pr, pcx + pr, pcy + pr), fill=fade((20, 24, 38), a))


def traj_pos(traj, u):
    """Posición continua (fila, col) y dirección de mirada a lo largo de una trayectoria."""
    if len(traj) == 1:
        return traj[0], (0, 0)
    u = clamp(u, 0, len(traj) - 1)
    k = int(min(u, len(traj) - 2))
    f = ease(u - k)
    a, b = traj[k], traj[k + 1]
    look = (b[1] - a[1], b[0] - a[0])
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), look


def cell_center(x0, y0, cs, rc):
    return x0 + rc[1] * cs + cs / 2, y0 + rc[0] * cs + cs / 2


def trail(d, x0, y0, cs, traj, u, col, wdt=6):
    if len(traj) < 2 or u <= 0:
        return
    k = int(min(u, len(traj) - 1))
    pts = [cell_center(x0, y0, cs, p) for p in traj[:k + 1]]
    pos, _ = traj_pos(traj, u)
    pts.append(cell_center(x0, y0, cs, pos))
    d.line(pts, fill=col, width=wdt, joint='curve')


def card(d, x, y, w, h, title, body, col, a, body_size=30):
    if a <= 0:
        return
    d.rounded_rectangle((x, y, x + w, y + h), 16, fill=fade(PANEL, a))
    d.rounded_rectangle((x, y, x + 8, y + h), 4, fill=fade(col, a))
    text(d, (x + 36, y + 40), title, 30, fade(col, a, PANEL), bold=True, anchor='lm')
    lines = wrap(d, body, body_size, w - 70) if isinstance(body, str) else body
    for i, line in enumerate(lines):
        text(d, (x + 36, y + 88 + i * (body_size + 12)), line, body_size, fade(TXT, a, PANEL), anchor='lm')


def rich(d, x, y, pieces, size, a=1.0, anchor_center=True, bold=False):
    """Dibuja una línea con trozos de colores. Devuelve las x de inicio/fin de cada trozo."""
    widths = [tw(d, s, size, bold) for s, _ in pieces]
    total = sum(widths)
    cx = x - total / 2 if anchor_center else x
    spans = []
    for (s, col), wdt in zip(pieces, widths):
        text(d, (cx, y), s, size, fade(col, a), bold=bold, anchor='lm')
        spans.append((cx, cx + wdt))
        cx += wdt
    return spans
