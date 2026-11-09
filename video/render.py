import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np

from common import W, H, FPS
import scenes as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
FADE = 0.35


def render_scene(i):
    fn, dur = S.SCENES[i]
    final = f'{OUT}/scene{i + 1}.mp4'
    tmp = f'{OUT}/scene{i + 1}.part.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-r', str(FPS), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(dur * FPS))
    for f in range(n):
        t = f / FPS
        arr = np.asarray(fn(t), dtype=np.uint8)
        k = min(1.0, t / FADE, (dur - t) / FADE)
        if k < 1:
            arr = (arr.astype(np.float32) * max(0.0, k)).astype(np.uint8)
        p.stdin.write(arr.tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, final)
    return final


def audio():
    sr = 44100
    total = sum(d for _, d in S.SCENES)
    n = int(total * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    # colchón suave: La m – Fa – Do – Sol, 4 s por acorde
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [130.81, 164.81, 196.0], [196.0, 246.94, 293.66]]
    seg = 4.0
    for ci in range(int(total / seg) + 1):
        ch = chords[ci % 4]
        a, b = int(ci * seg * sr), int(min(n, (ci + 1) * seg * sr + sr))
        if a >= n:
            break
        tt = t[a:b] - ci * seg
        env = np.minimum(1, tt / 1.2) * np.clip((seg + 1 - tt) / 1.2, 0, 1)
        sig = sum(np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt + 0.3) for f in ch)
        out[a:b] += 0.035 * env * sig
    # efectos
    def tone(t0, freqs, dur, vol, decay):
        a = int(t0 * sr)
        m = int(dur * sr)
        if a + m > n:
            m = n - a
        tt = np.arange(m) / sr
        env = np.exp(-tt * decay) * np.minimum(1, tt / 0.01)
        sig = sum(np.sin(2 * np.pi * f * tt) for f in freqs)
        out[a:a + m] += vol * env * sig

    offs = np.cumsum([0] + [d for _, d in S.SCENES])
    for sg in S.S1_SEGS:
        tone(sg['end'] + 0.1, [110, 116], 0.6, 0.12, 5)        # fallo
    tone(S.S1_WIN_END + 0.05, [523.25, 659.25, 783.99], 1.2, 0.08, 3)  # éxito
    tone(S.S1_WIN_END + 0.20, [1046.5], 1.0, 0.05, 4)
    tone(offs[2] + S.FG_GOAL_T + 0.05, [523.25, 659.25, 783.99], 1.2, 0.08, 3)
    # fundido general
    out *= np.minimum(1, t / 1.5) * np.minimum(1, (total - t) / 2.0)
    out = np.clip(out, -1, 1)
    pcm = (out * 32767 * 0.9).astype(np.int16)
    path = f'{OUT}/audio.wav'
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == '__main__':
    order = [4, 2, 0, 5, 1, 3, 6]  # las largas primero
    with Pool(2) as pool:
        files = pool.map(render_scene, order, chunksize=1)
    with open(f'{OUT}/list.txt', 'w') as f:
        for i in range(len(S.SCENES)):
            f.write(f"file 'scene{i + 1}.mp4'\n")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', f'{OUT}/list.txt',
                    '-c', 'copy', f'{OUT}/video_mudo.mp4'], check=True)
    a = audio()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f'{OUT}/video_mudo.mp4', '-i', a,
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest',
                    os.path.join(OUT, 'q_learning_ia_aprende_a_jugar.mp4')], check=True)
    print('done')
