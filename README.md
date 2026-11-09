# Q-learning: cómo una IA aprende a jugar

Implementación sencilla de **Q-learning tabular** en Python, junto con el código que genera un vídeo explicativo de 5 minutos sobre el algoritmo.

Un robot empieza en la esquina inferior izquierda de un tablero de 4 × 4 y tiene que llegar a la meta (★) sin caer en las trampas (X). Nadie le dice las reglas: aprende solo, por ensayo y error.

```
 . . . ★
 . X . .
 . . . X
 R . . .
```

## Ejecutarlo

Requiere Python 3.9+ y NumPy.

```bash
pip install -r requirements.txt
python q_learning.py
```

Opciones:

```bash
python q_learning.py --episodios 1000 --alpha 0.3 --gamma 0.95 --semilla 42
```

El script muestra el progreso del entrenamiento, la tabla Q aprendida, la mejor acción en cada casilla y el camino final:

```
Política (mejor acción por casilla):
  → → → ★
  ↑ X ↑ ↑
  ↑ ← ↑ X
  ↑ ← ↑ →

Camino aprendido (6 pasos): (3, 0) → (2, 0) → (1, 0) → (0, 0) → (0, 1) → (0, 2) → (0, 3)
```

## Cómo funciona

**Estados, acciones y recompensas**

| Elemento | En este tablero |
|---|---|
| Estado | La casilla donde está el robot (16 posibles) |
| Acción | Arriba, abajo, izquierda o derecha |
| Recompensa | +10 al llegar a la meta, −10 al caer en una trampa, −1 por cada paso |

El −1 por paso hace que un camino corto valga más que uno largo.

**La tabla Q**

Para cada pareja (casilla, acción), el agente guarda un número `Q(s, a)`: cuánta recompensa espera conseguir si hace esa acción en esa casilla y después sigue jugando bien. Son 16 × 4 = 64 números que empiezan en 0.

**La regla de actualización**

Después de cada paso, el valor se corrige según la "sorpresa":

```
Q(s,a) ← Q(s,a) + α · [ r + γ · máx Q(s′,a′) − Q(s,a) ]
                        └──── lo que obtuvo ────┘   └ lo que esperaba
```

- `α` (alfa, tasa de aprendizaje): cuánto aprende de cada sorpresa. Por defecto 0.5.
- `γ` (gamma, factor de descuento): cuánto vale el futuro frente al presente. Por defecto 0.9.

Ejemplo con α = 0.5 y γ = 0.9, la primera vez que el robot llega a la meta desde la casilla 3:

```
Q = 0 + 0.5 × (10 + 0.9 × 0 − 0) = 5
```

Y en un episodio posterior, desde la casilla 2:

```
Q = 0 + 0.5 × (−1 + 0.9 × 5 − 0) = 1.75
```

Así el valor de la meta "viaja" hacia atrás. Tras 500 episodios, la fila superior converge a 6.2 → 8.0 → 10.

**Explorar o aprovechar (ε-voraz)**

Con probabilidad ε el robot elige una acción al azar (explora); si no, elige la de mayor valor Q (aprovecha). ε empieza en 1.0 y baja un 1 % por episodio hasta 0.05.

## Del Q-learning al Deep Q-Learning

Una tabla funciona cuando hay pocos estados. En un videojuego real, como Atari (210 × 160 píxeles, 128 colores), el número de pantallas posibles tiene más de 70 000 cifras. Deep Q-Learning (DQN) reemplaza la tabla por una red neuronal que recibe el estado y estima un valor Q por cada acción. En 2015, DeepMind usó una misma red de este tipo para aprender 49 juegos de Atari a partir de los píxeles.

## El vídeo

La carpeta [`video/`](video/) contiene el código que genera el vídeo explicativo (1920 × 1080, 30 fps, 5 minutos). Todo lo que aparece en él sale de un entrenamiento real del mismo algoritmo con la semilla 5.

Requiere `numpy`, `Pillow` y `ffmpeg` instalado en el sistema.

```bash
cd video
python sim.py      # entrena y guarda los datos en sim.pkl
python render.py   # renderiza las escenas y crea out/q_learning_ia_aprende_a_jugar.mp4
```

| Archivo | Contenido |
|---|---|
| `sim.py` | Entorno y entrenamiento; elige una semilla con un arranque didáctico |
| `common.py` | Colores, tipografías y funciones de dibujo (tablero, robot, subtítulos) |
| `scenes.py` | Las 7 escenas y sus subtítulos |
| `render.py` | Renderizado con ffmpeg, unión de escenas y audio |

## Para seguir practicando

- Cambia la posición de las trampas o agrega más casillas.
- Prueba otros valores de `α` y `γ` y observa cuántos episodios tarda en converger.
- Sustituye la tabla por una pequeña red neuronal: ese es el primer paso hacia DQN.
