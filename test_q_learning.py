"""
Pruebas de los límites de episodios (mínimo 1, máximo 100000) y de las funciones
epsilon, paso, elegir_accion, entrenar, camino_voraz,
imprimir_politica e imprimir_tabla.

Uso:
    python -m unittest test_q_learning -v
"""
import argparse
import io
import math
import os
import subprocess
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from q_learning import (
    COLUMNAS, FILAS, INICIO, META, TRAMPAS, alpha_valido, camino_voraz, elegir_accion, entrenar,
    episodios_validos, epsilon, gamma_valido, imprimir_politica, imprimir_tabla, numero_decimal,
    paso, semilla_valida,
)

SCRIPT = Path(__file__).with_name('q_learning.py')
MINIMO = 1
MAXIMO = 100_000


def ejecutar(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding='utf-8',
        env={'PYTHONIOENCODING': 'utf-8', **os.environ},
    )


class ParserDePrueba(argparse.ArgumentParser):
    """Parser que lanza una excepción en vez de cerrar el programa."""

    def error(self, mensaje):
        raise ValueError(mensaje)


def parsear_episodios(*args):
    parser = ParserDePrueba()
    parser.add_argument('--episodios', type=episodios_validos, default=500)
    return parser.parse_args(list(args)).episodios


class TestEpisodiosValidos(unittest.TestCase):
    # --- límites ---
    def test_acepta_el_maximo(self):
        self.assertEqual(episodios_validos(str(MAXIMO)), MAXIMO)

    def test_acepta_el_minimo(self):
        self.assertEqual(episodios_validos(str(MINIMO)), MINIMO)

    def test_acepta_todo_el_rango_valido(self):
        for episodios in range(MINIMO, MAXIMO + 1):
            if episodios_validos(str(episodios)) != episodios:
                self.fail(f'{episodios} no se devolvió igual')

    def test_acepta_valores_por_debajo_del_maximo(self):
        self.assertEqual(episodios_validos('500'), 500)
        self.assertEqual(episodios_validos(str(MAXIMO - 1)), MAXIMO - 1)

    def test_rechaza_por_debajo_del_minimo(self):
        for valor in ('0', '-0', '-1', '-5', '-100000', '-' + '9' * 40):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                episodios_validos(valor)
            self.assertEqual(str(error.exception), 'el mínimo es 1')

    def test_rechaza_por_encima_del_maximo(self):
        for valor in (MAXIMO + 1, MAXIMO + 2, 1_000_000, 10 ** 40):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                episodios_validos(str(valor))
            self.assertEqual(str(error.exception), 'el máximo es 100000')

    # --- formato del texto ---
    def test_devuelve_un_entero(self):
        for valor in ('1', '500', str(MAXIMO)):
            with self.subTest(valor=valor):
                self.assertIs(type(episodios_validos(valor)), int)

    def test_acepta_digitos_con_signo_ceros_o_espacios(self):
        casos = {' 500 ': 500, '\t500\n': 500, '+500': 500, '00500': 500, '0100000': MAXIMO}
        for valor, esperado in casos.items():
            with self.subTest(valor=valor):
                self.assertEqual(episodios_validos(valor), esperado)

    def assertRechazaFormato(self, valor):
        with self.assertRaises(argparse.ArgumentTypeError) as error:
            episodios_validos(valor)
        self.assertIn('debe ser un número entero', str(error.exception))
        self.assertIn(repr(valor), str(error.exception))

    def test_rechaza_texto_que_no_es_un_entero(self):
        for valor in ('mil', 'quinientos', 'five', '', '  ', '+', '-', '500.0', '500,0', '1e3',
                      '0x10', '0b101', '5 00', '500a', 'ñ', '½', '--5', '+-5'):
            with self.subTest(valor=valor):
                self.assertRechazaFormato(valor)

    def test_rechaza_separadores_de_miles(self):
        for valor in ('1_000', '100_000', '1.000', '1,000', '1 000'):
            with self.subTest(valor=valor):
                self.assertRechazaFormato(valor)

    def test_rechaza_digitos_de_otros_alfabetos(self):
        casos = {
            '٥٠٠': 'árabe', '۵۰۰': 'persa', '५००': 'devanagari', '５００': 'ancho completo',
            '๕๐๐': 'tailandés', '²': 'superíndice', '⑤': 'en círculo', 'Ⅴ': 'romano',
            '5 ': 'espacio no separable', ' 5': 'espacio Unicode',
        }
        for valor, alfabeto in casos.items():
            with self.subTest(alfabeto=alfabeto):
                self.assertRechazaFormato(valor)

    def test_rechaza_lo_que_no_es_texto(self):
        for valor in (500, 500.7, True, None, b'500', ['500']):
            with self.subTest(valor=valor), self.assertRaises(TypeError):
                episodios_validos(valor)

    # --- con argparse ---
    def test_argparse_usa_el_valor_por_defecto(self):
        self.assertEqual(parsear_episodios(), 500)

    def test_argparse_acepta_los_limites(self):
        self.assertEqual(parsear_episodios('--episodios', '1'), 1)
        self.assertEqual(parsear_episodios('--episodios', str(MAXIMO)), MAXIMO)
        self.assertEqual(parsear_episodios(f'--episodios={MAXIMO}'), MAXIMO)

    def test_argparse_muestra_el_mensaje_de_cada_limite(self):
        casos = {'0': 'el mínimo es 1', str(MAXIMO + 1): 'el máximo es 100000'}
        for valor, mensaje in casos.items():
            with self.subTest(valor=valor), self.assertRaises(ValueError) as error:
                parsear_episodios('--episodios', valor)
            self.assertIn(mensaje, str(error.exception))

    def test_argparse_rechaza_texto_no_numerico(self):
        for valor in ('mil', '1_000', '٥٠٠'):
            with self.subTest(valor=valor), self.assertRaises(ValueError) as error:
                parsear_episodios('--episodios', valor)
            self.assertIn('debe ser un número entero', str(error.exception))


class TestNumeroDecimal(unittest.TestCase):
    def test_acepta_formas_comunes_de_escribir_un_numero(self):
        casos = {'0.5': 0.5, '1': 1.0, '0': 0.0, '.5': 0.5, '1.': 1.0, '+0.5': 0.5, '-0.5': -0.5,
                 ' 0.5 ': 0.5, '5e-1': 0.5, '1E0': 1.0, '0.05E+1': 0.5, '00.5': 0.5}
        for valor, esperado in casos.items():
            with self.subTest(valor=valor):
                resultado = numero_decimal(valor)
                self.assertIs(type(resultado), float)
                self.assertEqual(resultado, esperado)

    def test_rechaza_nan_e_infinito(self):
        for valor in ('nan', 'NaN', '-nan', 'inf', '-inf', 'Infinity', 'infinity'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                numero_decimal(valor)
            self.assertIn('debe ser un número', str(error.exception))

    def test_rechaza_texto_que_no_es_un_numero(self):
        for valor in ('', '  ', '.', '+', '-', 'e1', '1e', '1e+', '0,5', '1/2', '½', '0.5.0',
                      '1_0', '0x1', 'medio', 'half', '0.5 0', '--0.5'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError):
                numero_decimal(valor)

    def test_rechaza_digitos_de_otros_alfabetos(self):
        for valor in ('٠.٥', '０.５', '۰.۵', '०.५', '0.5 '):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError):
                numero_decimal(valor)

    def test_rechaza_lo_que_no_es_texto(self):
        for valor in (0.5, float('nan'), 1, None, b'0.5'):
            with self.subTest(valor=valor), self.assertRaises(TypeError):
                numero_decimal(valor)


class TestAlphaValido(unittest.TestCase):
    def test_acepta_valores_entre_0_y_1(self):
        for valor, esperado in (('1', 1.0), ('0.5', 0.5), ('0.01', 0.01), ('1e-9', 1e-9),
                                ('1.0', 1.0)):
            with self.subTest(valor=valor):
                self.assertEqual(alpha_valido(valor), esperado)

    def test_rechaza_fuera_de_rango(self):
        for valor in ('0', '0.0', '-0', '-0.5', '-1', '1.0000001', '2', '1e400', '-1e400'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                alpha_valido(valor)
            self.assertEqual(str(error.exception), 'debe ser mayor que 0 y como mucho 1')

    def test_rechaza_nan_e_infinito(self):
        for valor in ('nan', 'inf', '-inf'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError):
                alpha_valido(valor)

    def test_cada_valor_valido_entrena_sin_errores(self):
        for valor in ('1', '0.5', '0.01'):
            with self.subTest(valor=valor):
                Q, _ = entrenar(50, alpha=alpha_valido(valor), verbose=False)
                self.assertTrue(np.isfinite(Q).all())


class TestGammaValido(unittest.TestCase):
    def test_acepta_valores_entre_0_y_1(self):
        for valor, esperado in (('0', 0.0), ('1', 1.0), ('0.9', 0.9), ('0.99', 0.99), ('-0', 0.0)):
            with self.subTest(valor=valor):
                self.assertEqual(gamma_valido(valor), esperado)

    def test_rechaza_fuera_de_rango(self):
        for valor in ('-0.0001', '-1', '1.0001', '1.5', '2', '1e400', '-1e400'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                gamma_valido(valor)
            self.assertEqual(str(error.exception), 'debe estar entre 0 y 1')

    def test_rechaza_nan_e_infinito(self):
        for valor in ('nan', 'inf', '-inf'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError):
                gamma_valido(valor)

    def test_cada_valor_valido_entrena_sin_errores(self):
        for valor in ('0', '0.9', '1'):
            with self.subTest(valor=valor):
                Q, _ = entrenar(50, gamma=gamma_valido(valor), verbose=False)
                self.assertTrue(np.isfinite(Q).all())


class TestSemillaValida(unittest.TestCase):
    def test_acepta_cero_y_positivos(self):
        for valor, esperado in (('0', 0), ('5', 5), ('42', 42), (' 7 ', 7), ('+3', 3), ('-0', 0),
                                ('4294967295', 2 ** 32 - 1), (str(2 ** 100), 2 ** 100)):
            with self.subTest(valor=valor):
                resultado = semilla_valida(valor)
                self.assertIs(type(resultado), int)
                self.assertEqual(resultado, esperado)

    def test_rechaza_negativos(self):
        for valor in ('-1', '-5', '-42', ' -1 ', str(-2 ** 100)):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                semilla_valida(valor)
            self.assertEqual(str(error.exception), 'debe ser 0 o mayor')

    def test_rechaza_texto_que_no_es_un_entero(self):
        for valor in ('', 'cinco', 'five', '5.0', '1e3', '0x10', '1_000', '٥', '５', 'nan', '--1'):
            with self.subTest(valor=valor), self.assertRaises(argparse.ArgumentTypeError) as error:
                semilla_valida(valor)
            self.assertIn('debe ser un número entero', str(error.exception))

    def test_rechaza_lo_que_no_es_texto(self):
        for valor in (5, -1, 5.0, None, True):
            with self.subTest(valor=valor), self.assertRaises(TypeError):
                semilla_valida(valor)

    def test_argparse_muestra_el_mensaje(self):
        parser = ParserDePrueba()
        parser.add_argument('--semilla', type=semilla_valida, default=5)
        self.assertEqual(parser.parse_args([]).semilla, 5)
        self.assertEqual(parser.parse_args(['--semilla', '0']).semilla, 0)
        with self.assertRaises(ValueError) as error:
            parser.parse_args(['--semilla=-1'])
        self.assertIn('debe ser 0 o mayor', str(error.exception))


class TestLineaDeComandos(unittest.TestCase):
    def test_cli_rechaza_semillas_negativas_sin_traza(self):
        for args in (('--semilla', '-1'), ('--semilla=-1',), ('--semilla=-100',)):
            with self.subTest(args=args):
                resultado = ejecutar(*args)
                self.assertEqual(resultado.returncode, 2)
                self.assertIn('argument --semilla: debe ser 0 o mayor', resultado.stderr)
                self.assertNotIn('Traceback', resultado.stderr)

    def test_cli_acepta_semilla_cero(self):
        resultado = ejecutar('--episodios', '20', '--semilla', '0')
        self.assertEqual(resultado.returncode, 0, resultado.stderr)

    def test_cli_rechaza_nan_e_infinito_sin_traza(self):
        for opcion in ('--alpha', '--gamma'):
            for valor in ('nan', 'inf'):
                with self.subTest(opcion=opcion, valor=valor):
                    resultado = ejecutar(opcion, valor)
                    self.assertEqual(resultado.returncode, 2)
                    self.assertIn(f"debe ser un número, por ejemplo 0.5 (no '{valor}')",
                                  resultado.stderr)
                    self.assertNotIn('Traceback', resultado.stderr)

    def test_cli_rechaza_alpha_y_gamma_fuera_de_rango(self):
        casos = [('--alpha', '0', 'debe ser mayor que 0 y como mucho 1'),
                 ('--alpha', '2', 'debe ser mayor que 0 y como mucho 1'),
                 ('--gamma', '-1', 'debe estar entre 0 y 1'),
                 ('--gamma', '1.5', 'debe estar entre 0 y 1')]
        for opcion, valor, mensaje in casos:
            with self.subTest(opcion=opcion, valor=valor):
                resultado = ejecutar(f'{opcion}={valor}')
                self.assertEqual(resultado.returncode, 2)
                self.assertIn(mensaje, resultado.stderr)
                self.assertNotIn('Traceback', resultado.stderr)

    def test_cli_acepta_los_limites_de_alpha_y_gamma(self):
        resultado = ejecutar('--episodios', '50', '--alpha', '1', '--gamma', '0')
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        resultado = ejecutar('--episodios', '50', '--gamma', '1')
        self.assertEqual(resultado.returncode, 0, resultado.stderr)

    def test_cli_rechaza_por_encima_del_maximo(self):
        resultado = ejecutar('--episodios', str(MAXIMO + 1))
        self.assertEqual(resultado.returncode, 2)
        self.assertIn('el máximo es 100000', resultado.stderr)

    def test_cli_rechaza_por_debajo_del_minimo(self):
        resultado = ejecutar('--episodios', '0')
        self.assertEqual(resultado.returncode, 2)
        self.assertIn('el mínimo es 1', resultado.stderr)

    def test_cli_rechaza_texto_no_numerico(self):
        resultado = ejecutar('--episodios', 'mil')
        self.assertEqual(resultado.returncode, 2)
        self.assertIn('--episodios', resultado.stderr)
        self.assertIn("debe ser un número entero, por ejemplo 500 (no 'mil')", resultado.stderr)

    def test_cli_acepta_el_minimo(self):
        resultado = ejecutar('--episodios', '1')
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertIn('Episodio    1', resultado.stdout)
        self.assertNotIn('Episodio    2', resultado.stdout)

    def test_cli_escribe_utf8_sin_configurar_la_consola(self):
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONIOENCODING', 'PYTHONUTF8')}
        resultado = subprocess.run(
            [sys.executable, str(SCRIPT), '--episodios', '100'],
            capture_output=True, env=env,
        )
        self.assertEqual(resultado.returncode, 0, resultado.stderr.decode('utf-8', 'replace'))
        salida = resultado.stdout.decode('utf-8')
        self.assertIn('ε', salida)
        self.assertIn('★', salida)


class TestEntrenamientoMaximo(unittest.TestCase):
    def test_entrena_el_maximo_de_episodios(self):
        Q, resultados = entrenar(episodios=MAXIMO, verbose=False)
        self.assertEqual(len(resultados), MAXIMO)
        self.assertEqual(camino_voraz(Q)[-1], META)


class TestEpsilon(unittest.TestCase):
    # Con los valores por defecto (decaimiento 0.99, mínimo 0.05) el suelo se
    # alcanza en el episodio 300: 0.99**298 > 0.05 y 0.99**299 < 0.05.
    PRIMER_EPISODIO_EN_SUELO = 300

    def test_primer_episodio_explora_siempre(self):
        self.assertEqual(epsilon(1), 1.0)

    def test_nunca_pasa_de_uno_con_episodios_menores_que_uno(self):
        for episodio in (0, -1, -100):
            with self.subTest(episodio=episodio):
                self.assertEqual(epsilon(episodio), 1.0)

    def test_valores_conocidos(self):
        casos = {2: 0.99, 3: 0.99 ** 2, 11: 0.99 ** 10, 101: 0.99 ** 100}
        for episodio, esperado in casos.items():
            with self.subTest(episodio=episodio):
                self.assertAlmostEqual(epsilon(episodio), esperado)

    def test_decae_geometricamente_antes_del_suelo(self):
        for episodio in range(1, self.PRIMER_EPISODIO_EN_SUELO - 1):
            with self.subTest(episodio=episodio):
                self.assertAlmostEqual(epsilon(episodio + 1) / epsilon(episodio), 0.99)

    def test_justo_antes_del_suelo(self):
        self.assertGreater(epsilon(self.PRIMER_EPISODIO_EN_SUELO - 1), 0.05)

    def test_alcanza_el_suelo(self):
        self.assertEqual(epsilon(self.PRIMER_EPISODIO_EN_SUELO), 0.05)

    def test_se_queda_en_el_suelo_hasta_el_maximo_de_episodios(self):
        for episodio in (self.PRIMER_EPISODIO_EN_SUELO + 1, 1_000, 10_000, MAXIMO):
            with self.subTest(episodio=episodio):
                self.assertEqual(epsilon(episodio), 0.05)

    def test_nunca_crece_y_siempre_esta_en_rango(self):
        anterior = epsilon(1)
        for episodio in range(1, MAXIMO + 1):
            actual = epsilon(episodio)
            self.assertGreaterEqual(actual, 0.05)
            self.assertLessEqual(actual, 1.0)
            self.assertLessEqual(actual, anterior)
            anterior = actual

    def test_devuelve_un_numero_real(self):
        for episodio in (1, 2, self.PRIMER_EPISODIO_EN_SUELO, MAXIMO):
            with self.subTest(episodio=episodio):
                self.assertIsInstance(epsilon(episodio), float)

    def test_eps_min_personalizado(self):
        self.assertEqual(epsilon(1, eps_min=0.2), 1.0)
        self.assertEqual(epsilon(MAXIMO, eps_min=0.2), 0.2)
        # 0.99**160 ≈ 0.2003 > 0.2 y 0.99**161 ≈ 0.1983 < 0.2
        self.assertGreater(epsilon(161, eps_min=0.2), 0.2)
        self.assertEqual(epsilon(162, eps_min=0.2), 0.2)

    def test_eps_min_cero_decae_sin_suelo(self):
        self.assertAlmostEqual(epsilon(1_001, eps_min=0.0), 0.99 ** 1_000)
        self.assertGreater(epsilon(1_001, eps_min=0.0), 0.0)

    def test_eps_min_uno_explora_siempre(self):
        for episodio in (1, 2, 500, MAXIMO):
            with self.subTest(episodio=episodio):
                self.assertEqual(epsilon(episodio, eps_min=1.0), 1.0)

    def test_decaimiento_personalizado(self):
        self.assertAlmostEqual(epsilon(2, decaimiento=0.5), 0.5)
        self.assertAlmostEqual(epsilon(4, decaimiento=0.5), 0.125)
        self.assertEqual(epsilon(6, decaimiento=0.5), 0.05)  # 0.5**5 = 0.03125

    def test_decaimiento_uno_no_decae(self):
        for episodio in (1, 300, MAXIMO):
            with self.subTest(episodio=episodio):
                self.assertEqual(epsilon(episodio, decaimiento=1.0), 1.0)

    def test_decaimiento_cero_salta_al_suelo(self):
        self.assertEqual(epsilon(1, decaimiento=0.0), 1.0)  # 0.0**0 == 1
        self.assertEqual(epsilon(2, decaimiento=0.0), 0.05)

    def test_primer_episodio_en_suelo_coincide_con_la_formula(self):
        # Primer n con 0.99**(n-1) <= 0.05  <=>  n >= 1 + ln(0.05)/ln(0.99)
        n = math.ceil(1 + math.log(0.05) / math.log(0.99))
        self.assertEqual(n, self.PRIMER_EPISODIO_EN_SUELO)


ARRIBA, ABAJO, IZQUIERDA, DERECHA = range(4)
TERMINALES = {META} | TRAMPAS


class TestPaso(unittest.TestCase):
    def test_tablero_esperado(self):
        self.assertEqual((FILAS, COLUMNAS), (4, 4))
        self.assertEqual(INICIO, (3, 0))
        self.assertEqual(META, (0, 3))
        self.assertEqual(TRAMPAS, {(1, 1), (2, 3)})

    def test_movimientos_libres_en_las_cuatro_direcciones(self):
        # Desde (2, 1) las cuatro casillas vecinas son normales salvo arriba (trampa).
        self.assertEqual(paso((2, 1), ABAJO), ((3, 1), -1, False))
        self.assertEqual(paso((2, 1), IZQUIERDA), ((2, 0), -1, False))
        self.assertEqual(paso((2, 1), DERECHA), ((2, 2), -1, False))
        self.assertEqual(paso((3, 2), ARRIBA), ((2, 2), -1, False))

    def test_choque_con_cada_borde_se_queda_quieto(self):
        casos = [
            ((0, 0), ARRIBA), ((0, 1), ARRIBA), ((0, 2), ARRIBA),
            ((3, 0), ABAJO), ((3, 1), ABAJO), ((3, 2), ABAJO), ((3, 3), ABAJO),
            ((0, 0), IZQUIERDA), ((2, 0), IZQUIERDA), ((3, 0), IZQUIERDA),
            ((3, 3), DERECHA),
        ]
        for estado, accion in casos:
            with self.subTest(estado=estado, accion=accion):
                self.assertEqual(paso(estado, accion), (estado, -1, False))

    def test_esquinas_chocan_en_dos_direcciones(self):
        esquinas = {
            (0, 0): (ARRIBA, IZQUIERDA), (3, 0): (ABAJO, IZQUIERDA),
            (3, 3): (ABAJO, DERECHA),
        }
        for esquina, acciones in esquinas.items():
            for accion in acciones:
                with self.subTest(esquina=esquina, accion=accion):
                    self.assertEqual(paso(esquina, accion)[0], esquina)

    def test_llegar_a_la_meta(self):
        for estado, accion in (((0, 2), DERECHA), ((1, 3), ARRIBA)):
            with self.subTest(estado=estado, accion=accion):
                self.assertEqual(paso(estado, accion), (META, 10, True))

    def test_caer_en_cada_trampa(self):
        casos = {
            (1, 1): [((0, 1), ABAJO), ((2, 1), ARRIBA), ((1, 0), DERECHA), ((1, 2), IZQUIERDA)],
            (2, 3): [((1, 3), ABAJO), ((3, 3), ARRIBA), ((2, 2), DERECHA)],
        }
        for trampa, entradas in casos.items():
            for estado, accion in entradas:
                with self.subTest(estado=estado, accion=accion):
                    self.assertEqual(paso(estado, accion), (trampa, -10, True))

    def test_primer_paso_desde_el_inicio(self):
        self.assertEqual(paso(INICIO, ARRIBA), ((2, 0), -1, False))
        self.assertEqual(paso(INICIO, DERECHA), ((3, 1), -1, False))
        self.assertEqual(paso(INICIO, ABAJO), (INICIO, -1, False))
        self.assertEqual(paso(INICIO, IZQUIERDA), (INICIO, -1, False))

    def test_todas_las_casillas_y_acciones(self):
        # Modelo independiente: recorta la posición al tablero en vez de comprobar límites.
        desplazamientos = {ARRIBA: (-1, 0), ABAJO: (1, 0), IZQUIERDA: (0, -1), DERECHA: (0, 1)}
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                for accion, (df, dc) in desplazamientos.items():
                    destino = (min(max(f + df, 0), FILAS - 1), min(max(c + dc, 0), COLUMNAS - 1))
                    if destino == META:
                        esperado = (destino, 10, True)
                    elif destino in TRAMPAS:
                        esperado = (destino, -10, True)
                    else:
                        esperado = (destino, -1, False)
                    with self.subTest(estado=(f, c), accion=accion):
                        self.assertEqual(paso((f, c), accion), esperado)

    def test_siempre_se_mueve_a_una_casilla_vecina_o_se_queda(self):
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                for accion in range(4):
                    (nf, nc), _, _ = paso((f, c), accion)
                    with self.subTest(estado=(f, c), accion=accion):
                        self.assertTrue(0 <= nf < FILAS and 0 <= nc < COLUMNAS)
                        self.assertLessEqual(abs(nf - f) + abs(nc - c), 1)

    def test_tipos_devueltos(self):
        siguiente, recompensa, terminado = paso(INICIO, ARRIBA)
        self.assertIsInstance(siguiente, tuple)
        self.assertIsInstance(recompensa, int)
        self.assertIsInstance(terminado, bool)

    def test_acepta_acciones_de_numpy(self):
        self.assertEqual(paso(INICIO, np.int64(ARRIBA)), ((2, 0), -1, False))

    def test_rechaza_acciones_fuera_de_rango(self):
        for accion in (4, 5, -1, -4, -5):
            with self.subTest(accion=accion), self.assertRaises(ValueError):
                paso(INICIO, accion)

    def test_rechaza_moverse_desde_la_meta_o_una_trampa(self):
        for estado in sorted(TERMINALES):
            for accion in range(4):
                with self.subTest(estado=estado, accion=accion), self.assertRaises(ValueError):
                    paso(estado, accion)

    def test_es_determinista(self):
        for _ in range(100):
            self.assertEqual(paso((2, 2), DERECHA), ((2, 3), -10, True))


class RngFalso:
    """Generador controlado: fija el valor de random() y registra las llamadas."""

    def __init__(self, aleatorio, entero=0, eleccion=None):
        self.aleatorio, self.entero, self.eleccion = aleatorio, entero, eleccion
        self.llamadas = []

    def random(self):
        self.llamadas.append('random')
        return self.aleatorio

    def integers(self, n):
        self.llamadas.append(('integers', n))
        return np.int64(self.entero)

    def choice(self, opciones):
        self.llamadas.append(('choice', [int(o) for o in opciones]))
        return opciones[0] if self.eleccion is None else self.eleccion


def q_con(valores, estado=INICIO):
    """Tabla Q a ceros salvo la fila de `estado`."""
    Q = np.zeros((FILAS, COLUMNAS, 4))
    Q[estado] = valores
    return Q


def secuencia(Q, eps, semilla, n=200):
    rng = np.random.default_rng(semilla)
    return [elegir_accion(Q, INICIO, eps, rng) for _ in range(n)]


class TestElegirAccion(unittest.TestCase):
    N = 20_000

    def frecuencias(self, Q, eps, estado=INICIO, semilla=0):
        rng = np.random.default_rng(semilla)
        conteo = Counter(elegir_accion(Q, estado, eps, rng) for _ in range(self.N))
        return {a: conteo[a] / self.N for a in range(4)}

    # --- aprovechar (eps = 0) ---
    def test_elige_la_mejor_accion_en_cada_casilla_y_posicion(self):
        rng = np.random.default_rng(0)
        for f in range(FILAS):
            for c in range(COLUMNAS):
                for mejor in range(4):
                    valores = [-1.0, -2.0, -3.0, -4.0]
                    valores[mejor] = 5.0
                    with self.subTest(estado=(f, c), mejor=mejor):
                        Q = q_con(valores, (f, c))
                        self.assertEqual(elegir_accion(Q, (f, c), 0.0, rng), mejor)

    def test_con_valores_todos_negativos_elige_el_menos_malo(self):
        Q = q_con([-9.0, -0.5, -3.0, -7.0])
        self.assertEqual(elegir_accion(Q, INICIO, 0.0, np.random.default_rng(0)), ABAJO)

    def test_distingue_diferencias_minimas(self):
        Q = q_con([1.0, 1.0 + 1e-12, 1.0, 1.0])
        self.assertEqual(set(secuencia(Q, 0.0, semilla=0)), {ABAJO})

    def test_solo_mira_la_fila_del_estado(self):
        Q = q_con([0.0, 0.0, 0.0, 9.0], INICIO)
        Q[0, 0] = [100.0, 0.0, 0.0, 0.0]
        self.assertEqual(elegir_accion(Q, INICIO, 0.0, np.random.default_rng(0)), DERECHA)

    def test_empate_doble_reparte_entre_las_empatadas(self):
        f = self.frecuencias(q_con([3.0, 1.0, 3.0, 0.0]), 0.0)
        self.assertEqual(f[ABAJO] + f[DERECHA], 0.0)
        self.assertAlmostEqual(f[ARRIBA], 0.5, delta=0.02)
        self.assertAlmostEqual(f[IZQUIERDA], 0.5, delta=0.02)

    def test_empate_triple_reparte_entre_las_empatadas(self):
        f = self.frecuencias(q_con([2.0, 2.0, -1.0, 2.0]), 0.0)
        self.assertEqual(f[IZQUIERDA], 0.0)
        for accion in (ARRIBA, ABAJO, DERECHA):
            with self.subTest(accion=accion):
                self.assertAlmostEqual(f[accion], 1 / 3, delta=0.02)

    def test_tabla_a_ceros_reparte_entre_las_cuatro(self):
        f = self.frecuencias(np.zeros((FILAS, COLUMNAS, 4)), 0.0)
        for accion in range(4):
            with self.subTest(accion=accion):
                self.assertAlmostEqual(f[accion], 0.25, delta=0.02)

    # --- explorar (eps = 1) ---
    def test_eps_uno_explora_uniformemente_aunque_haya_una_mejor(self):
        f = self.frecuencias(q_con([0.0, 0.0, 0.0, 100.0]), 1.0)
        for accion in range(4):
            with self.subTest(accion=accion):
                self.assertAlmostEqual(f[accion], 0.25, delta=0.02)

    # --- mezcla ---
    def test_proporcion_voraz_con_eps_intermedio(self):
        # Aprovecha con probabilidad 1 - eps y, al explorar, acierta 1 de cada 4 veces.
        for eps in (0.1, 0.3, 0.5, 0.9):
            f = self.frecuencias(q_con([0.0, 0.0, 0.0, 1.0]), eps)
            with self.subTest(eps=eps):
                self.assertAlmostEqual(f[DERECHA], 1 - eps + eps / 4, delta=0.02)
                for accion in (ARRIBA, ABAJO, IZQUIERDA):
                    self.assertAlmostEqual(f[accion], eps / 4, delta=0.02)

    # --- límite exacto de rng.random() < eps ---
    def test_explora_si_el_aleatorio_es_menor_que_eps(self):
        rng = RngFalso(aleatorio=0.299, entero=IZQUIERDA)
        self.assertEqual(elegir_accion(q_con([0, 0, 0, 1.0]), INICIO, 0.3, rng), IZQUIERDA)
        self.assertEqual(rng.llamadas, ['random', ('integers', 4)])

    def test_aprovecha_si_el_aleatorio_es_igual_a_eps(self):
        rng = RngFalso(aleatorio=0.3)
        self.assertEqual(elegir_accion(q_con([0, 0, 0, 1.0]), INICIO, 0.3, rng), DERECHA)
        self.assertEqual(rng.llamadas, ['random', ('choice', [DERECHA])])

    def test_eps_cero_nunca_explora_aunque_el_aleatorio_sea_cero(self):
        rng = RngFalso(aleatorio=0.0)
        self.assertEqual(elegir_accion(q_con([0, 0, 0, 1.0]), INICIO, 0.0, rng), DERECHA)
        self.assertNotIn(('integers', 4), rng.llamadas)

    def test_eps_uno_siempre_explora(self):
        rng = RngFalso(aleatorio=0.9999999)
        elegir_accion(q_con([0, 0, 0, 1.0]), INICIO, 1.0, rng)
        self.assertEqual(rng.llamadas, ['random', ('integers', 4)])

    def test_al_empatar_pasa_todas_las_empatadas_al_desempate(self):
        rng = RngFalso(aleatorio=0.5, eleccion=np.int64(DERECHA))
        Q = q_con([4.0, 1.0, 4.0, 4.0])
        self.assertEqual(elegir_accion(Q, INICIO, 0.0, rng), DERECHA)
        self.assertEqual(rng.llamadas, ['random', ('choice', [ARRIBA, IZQUIERDA, DERECHA])])

    # --- propiedades generales ---
    def test_devuelve_un_entero_de_python_en_rango(self):
        rng = np.random.default_rng(1)
        Q = np.random.default_rng(2).normal(size=(FILAS, COLUMNAS, 4))
        for eps in (0.0, 0.5, 1.0):
            for _ in range(500):
                accion = elegir_accion(Q, INICIO, eps, rng)
                self.assertIs(type(accion), int)
                self.assertIn(accion, range(4))

    def test_no_modifica_la_tabla_q(self):
        Q = np.random.default_rng(3).normal(size=(FILAS, COLUMNAS, 4))
        copia = Q.copy()
        rng = np.random.default_rng(0)
        for eps in (0.0, 0.5, 1.0):
            for _ in range(100):
                elegir_accion(Q, (1, 2), eps, rng)
        np.testing.assert_array_equal(Q, copia)

    def test_misma_semilla_misma_secuencia(self):
        Q = q_con([1.0, 1.0, 0.0, 1.0])
        self.assertEqual(secuencia(Q, 0.4, semilla=7), secuencia(Q, 0.4, semilla=7))

    def test_semillas_distintas_dan_secuencias_distintas(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        self.assertNotEqual(secuencia(Q, 0.5, semilla=1), secuencia(Q, 0.5, semilla=2))


def valores_optimos(gamma=0.9):
    """Iteración de valores: el valor real de cada casilla, calculado sin Q-learning."""
    V = np.zeros((FILAS, COLUMNAS))
    for _ in range(200):
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                opciones = []
                for accion in range(4):
                    siguiente, r, terminado = paso((f, c), accion)
                    opciones.append(r + (0.0 if terminado else gamma * V[siguiente]))
                V[f, c] = max(opciones)
    return V


class TestEntrenar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Q, cls.resultados = entrenar(verbose=False)

    # --- forma de lo que devuelve ---
    def test_forma_de_la_tabla_q(self):
        self.assertEqual(self.Q.shape, (FILAS, COLUMNAS, 4))
        self.assertEqual(self.Q.dtype, np.float64)

    def test_un_resultado_por_episodio(self):
        for episodios in (1, 7, 500):
            with self.subTest(episodios=episodios):
                _, resultados = entrenar(episodios, verbose=False)
                self.assertEqual(len(resultados), episodios)

    # --- validación de episodios y max_pasos ---
    def test_acepta_los_limites_de_episodios(self):
        for episodios in (1, np.int64(3)):
            with self.subTest(episodios=episodios):
                _, resultados = entrenar(episodios, verbose=False)
                self.assertEqual(len(resultados), episodios)

    def test_rechaza_episodios_fuera_de_rango(self):
        for episodios in (0, -1, -500, MAXIMO + 1, 10 ** 40, np.int64(0)):
            with self.subTest(episodios=episodios), self.assertRaises(ValueError) as error:
                entrenar(episodios, verbose=False)
            self.assertIn('episodios debe estar entre 1 y 100000', str(error.exception))

    def test_rechaza_max_pasos_menor_que_uno(self):
        for max_pasos in (0, -1, -60, np.int64(0)):
            with self.subTest(max_pasos=max_pasos), self.assertRaises(ValueError) as error:
                entrenar(1, max_pasos=max_pasos, verbose=False)
            self.assertIn('max_pasos debe ser al menos 1', str(error.exception))

    def test_acepta_max_pasos_grandes(self):
        for max_pasos in (1, 1000, np.int64(60)):
            with self.subTest(max_pasos=max_pasos):
                _, resultados = entrenar(5, max_pasos=max_pasos, verbose=False)
                self.assertEqual(len(resultados), 5)

    def test_rechaza_episodios_y_max_pasos_que_no_son_enteros(self):
        for nombre in ('episodios', 'max_pasos'):
            for valor in (5.0, 5.5, '5', None, True, False, float('nan'), float('inf'), [5],
                          np.float64(5)):
                with self.subTest(nombre=nombre, valor=valor), self.assertRaises(TypeError) as error:
                    entrenar(**{'verbose': False, 'episodios': 1, nombre: valor})
                self.assertIn(f'{nombre} debe ser un número entero', str(error.exception))

    def test_valores_por_defecto(self):
        Q, resultados = entrenar(500, alpha=0.5, gamma=0.9, max_pasos=60, semilla=5, verbose=False)
        np.testing.assert_array_equal(Q, self.Q)
        self.assertEqual(resultados, self.resultados)

    # --- coherencia de los resultados ---
    def test_cada_resultado_es_un_final_valido_con_recompensa_entera(self):
        for final, total in self.resultados:
            self.assertIn(final, ('meta', 'trampa', 'límite de pasos'))
            self.assertIs(type(total), int)

    def test_recompensas_posibles_segun_el_final(self):
        # Meta en k pasos: 10 - (k - 1), con k entre 6 (camino más corto) y 60.
        # Trampa en k pasos: -10 - (k - 1), con k entre 3 y 60.
        for final, total in self.resultados:
            with self.subTest(final=final, total=total):
                if final == 'meta':
                    self.assertTrue(10 - 59 <= total <= 10 - 5)
                elif final == 'trampa':
                    self.assertTrue(-10 - 59 <= total <= -10 - 2)
                else:
                    self.assertEqual(total, -60)

    def test_limite_de_pasos(self):
        # La trampa más cercana está a 3 pasos: con 2 no se puede terminar.
        Q, resultados = entrenar(50, max_pasos=2, verbose=False)
        self.assertEqual(resultados, [('límite de pasos', -2)] * 50)
        self.assertTrue((Q <= 0).all())

    # --- regla de actualización ---
    def test_primer_paso_aplica_la_regla_de_q_learning(self):
        # Tabla a ceros y un solo paso desde el inicio, que nunca termina:
        # Q += alpha * (-1 + gamma * 0 - 0)
        for alpha in (0.1, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                Q, resultados = entrenar(1, alpha=alpha, max_pasos=1, verbose=False)
                self.assertEqual(resultados, [('límite de pasos', -1)])
                self.assertEqual(np.count_nonzero(Q), 1)
                self.assertAlmostEqual(Q[INICIO].sum(), -alpha)

    def test_reproduce_la_regla_paso_a_paso(self):
        # Modelo independiente que repite el entrenamiento con el mismo generador.
        alpha, gamma, semilla = 0.3, 0.8, 11
        rng = np.random.default_rng(semilla)
        Q = np.zeros((FILAS, COLUMNAS, 4))
        for ep in range(1, 31):
            estado = INICIO
            for _ in range(60):
                a = elegir_accion(Q, estado, epsilon(ep), rng)
                siguiente, r, terminado = paso(estado, a)
                objetivo = r if terminado else r + gamma * Q[siguiente].max()
                Q[estado][a] = (1 - alpha) * Q[estado][a] + alpha * objetivo
                estado = siguiente
                if terminado:
                    break
        Q_entrenada, _ = entrenar(30, alpha=alpha, gamma=gamma, semilla=semilla, verbose=False)
        np.testing.assert_allclose(Q_entrenada, Q)

    # --- validación de alpha y gamma ---
    def test_rechaza_alpha_fuera_de_rango(self):
        for alpha in (0, 0.0, -0.0, -0.5, -1, 1.0000001, 2, float('inf'), float('-inf'),
                      float('nan'), np.nan, np.float64(1.5)):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError) as error:
                entrenar(1, alpha=alpha, verbose=False)
            self.assertIn('alpha debe ser mayor que 0 y como mucho 1', str(error.exception))

    def test_rechaza_gamma_fuera_de_rango(self):
        for gamma in (-0.0001, -1, 1.0001, 1.5, float('inf'), float('-inf'), float('nan'),
                      np.float64(-1)):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError) as error:
                entrenar(1, gamma=gamma, verbose=False)
            self.assertIn('gamma debe estar entre 0 y 1', str(error.exception))

    def test_rechaza_alpha_y_gamma_que_no_son_numeros(self):
        for nombre in ('alpha', 'gamma'):
            for valor in ('0.5', None, True, False, [0.5], np.array([0.5]), 0.5j):
                with self.subTest(nombre=nombre, valor=valor), self.assertRaises(TypeError) as error:
                    entrenar(1, verbose=False, **{nombre: valor})
                self.assertIn(f'{nombre} debe ser un número', str(error.exception))

    def test_acepta_los_limites_de_alpha_y_gamma(self):
        for alpha, gamma in ((1, 0), (1.0, 1.0), (1e-9, 0.0), (np.float64(0.5), np.float64(0.9)),
                             (1, 1)):
            with self.subTest(alpha=alpha, gamma=gamma):
                Q, resultados = entrenar(20, alpha=alpha, gamma=gamma, verbose=False)
                self.assertEqual(len(resultados), 20)
                self.assertTrue(np.isfinite(Q).all())

    # --- validación de la semilla ---
    def test_rechaza_semillas_negativas(self):
        for semilla in (-1, -5, -2 ** 100, np.int64(-1)):
            with self.subTest(semilla=semilla), self.assertRaises(ValueError) as error:
                entrenar(1, semilla=semilla, verbose=False)
            self.assertIn('semilla debe ser 0 o mayor', str(error.exception))
            self.assertIn(f'(no {semilla!r})', str(error.exception))

    def test_rechaza_semillas_que_no_son_enteros(self):
        for semilla in (5.0, 5.5, '5', True, False, float('nan'), [5], np.float64(5)):
            with self.subTest(semilla=semilla), self.assertRaises(TypeError) as error:
                entrenar(1, semilla=semilla, verbose=False)
            self.assertIn('semilla debe ser un número entero', str(error.exception))

    def test_acepta_semillas_validas(self):
        for semilla in (0, 1, 2 ** 32 - 1, 2 ** 100, np.int64(7), np.uint32(7)):
            with self.subTest(semilla=semilla):
                _, resultados = entrenar(3, semilla=semilla, verbose=False)
                self.assertEqual(len(resultados), 3)

    def test_semilla_none_es_aleatoria(self):
        _, resultados = entrenar(3, semilla=None, verbose=False)
        self.assertEqual(len(resultados), 3)

    def test_semilla_de_numpy_da_el_mismo_resultado_que_la_de_python(self):
        Q1, r1 = entrenar(50, semilla=7, verbose=False)
        Q2, r2 = entrenar(50, semilla=np.int64(7), verbose=False)
        np.testing.assert_array_equal(Q1, Q2)
        self.assertEqual(r1, r2)

    def test_valida_antes_de_empezar(self):
        # El error llega antes de imprimir nada, aunque verbose sea True.
        salida = io.StringIO()
        with redirect_stdout(salida), self.assertRaises(ValueError):
            entrenar(10, alpha=float('nan'))
        self.assertEqual(salida.getvalue(), '')

    def test_mensaje_incluye_el_valor_recibido(self):
        with self.assertRaises(ValueError) as error:
            entrenar(1, alpha=2, verbose=False)
        self.assertIn('(no 2)', str(error.exception))
        with self.assertRaises(ValueError) as error:
            entrenar(1, gamma=float('nan'), verbose=False)
        self.assertIn('(no nan)', str(error.exception))

    def test_gamma_cero_solo_mira_la_recompensa_inmediata(self):
        Q, _ = entrenar(2000, gamma=0.0, verbose=False)
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                for accion in range(4):
                    _, r, _ = paso((f, c), accion)
                    with self.subTest(estado=(f, c), accion=accion):
                        # Q solo se acerca a r desde 0, sin pasarse nunca.
                        self.assertTrue(min(r, 0) <= Q[f, c, accion] <= max(r, 0))

    def test_meta_y_trampas_no_se_actualizan(self):
        for estado in TERMINALES:
            with self.subTest(estado=estado):
                np.testing.assert_array_equal(self.Q[estado], np.zeros(4))

    def test_valores_acotados(self):
        # Las recompensas van de -10 a 10 y gamma < 1, así que Q nunca sale de [-10, 10].
        Q, _ = entrenar(5000, verbose=False)
        self.assertTrue(np.isfinite(Q).all())
        self.assertTrue((Q >= -10).all() and (Q <= 10).all())

    # --- aprendizaje ---
    def test_aprende_un_camino_optimo_a_la_meta(self):
        for semilla in range(1, 6):
            with self.subTest(semilla=semilla):
                Q, _ = entrenar(semilla=semilla, verbose=False)
                camino = camino_voraz(Q)
                self.assertEqual(camino[-1], META)
                self.assertEqual(len(camino) - 1, 6)  # el camino más corto posible
                self.assertTrue(TRAMPAS.isdisjoint(camino))

    def test_valores_del_camino_coinciden_con_los_optimos(self):
        V = valores_optimos()
        for estado in camino_voraz(self.Q)[:-1]:
            with self.subTest(estado=estado):
                self.assertAlmostEqual(self.Q[estado].max(), V[estado], places=3)

    def test_el_valor_del_inicio_es_el_esperado(self):
        # Seis pasos hasta la meta: -1 - 0.9 - 0.9² - 0.9³ - 0.9⁴ + 0.9⁵ · 10
        esperado = -sum(0.9 ** k for k in range(5)) + 0.9 ** 5 * 10
        self.assertAlmostEqual(self.Q[INICIO].max(), esperado, places=4)

    def test_mejora_con_la_practica(self):
        metas = lambda tramo: sum(final == 'meta' for final, _ in tramo)
        self.assertLess(metas(self.resultados[:100]), metas(self.resultados[-100:]))
        self.assertGreaterEqual(metas(self.resultados[-100:]), 90)

    def test_la_recompensa_media_sube(self):
        media = lambda tramo: sum(total for _, total in tramo) / len(tramo)
        self.assertLess(media(self.resultados[:50]), media(self.resultados[-50:]))

    # --- semilla ---
    def test_misma_semilla_mismo_resultado(self):
        Q1, r1 = entrenar(300, semilla=42, verbose=False)
        Q2, r2 = entrenar(300, semilla=42, verbose=False)
        np.testing.assert_array_equal(Q1, Q2)
        self.assertEqual(r1, r2)

    def test_semillas_distintas_dan_entrenamientos_distintos(self):
        _, r1 = entrenar(300, semilla=1, verbose=False)
        _, r2 = entrenar(300, semilla=2, verbose=False)
        self.assertNotEqual(r1, r2)

    # --- salida por pantalla ---
    def capturar(self, **kwargs):
        salida = io.StringIO()
        with redirect_stdout(salida):
            _, resultados = entrenar(**kwargs)
        return salida.getvalue().splitlines(), resultados

    def test_sin_verbose_no_imprime_nada(self):
        lineas, _ = self.capturar(episodios=300, verbose=False)
        self.assertEqual(lineas, [])

    def test_verbose_imprime_los_cinco_primeros_y_cada_cien(self):
        lineas, _ = self.capturar(episodios=500)
        numeros = [int(linea.split('|')[0].split()[1]) for linea in lineas]
        self.assertEqual(numeros, [1, 2, 3, 4, 5, 100, 200, 300, 400, 500])

    def test_verbose_muestra_epsilon_final_y_recompensa(self):
        lineas, resultados = self.capturar(episodios=100)
        self.assertIn('ε = 1.00', lineas[0])
        self.assertIn('ε = 0.99', lineas[1])
        final, total = resultados[0]
        self.assertIn(final, lineas[0])
        self.assertTrue(lineas[0].endswith(f'recompensa total {total:+d}'))

    def test_verbose_no_cambia_el_resultado(self):
        _, con = self.capturar(episodios=200)
        _, sin = self.capturar(episodios=200, verbose=False)
        self.assertEqual(con, sin)


def q_de_politica(politica, por_defecto=None):
    """Tabla Q cuya mejor acción en cada casilla es la indicada en `politica`."""
    Q = np.zeros((FILAS, COLUMNAS, 4))
    for f in range(FILAS):
        for c in range(COLUMNAS):
            accion = politica.get((f, c), por_defecto)
            if accion is not None:
                Q[f, c, accion] = 1.0
    return Q


def camino_esperado(Q, max_pasos=20):
    """Modelo independiente: recorta la posición al tablero en vez de usar paso()."""
    desplazamientos = {ARRIBA: (-1, 0), ABAJO: (1, 0), IZQUIERDA: (0, -1), DERECHA: (0, 1)}
    f, c = INICIO
    camino = [INICIO]
    for _ in range(max_pasos):
        df, dc = desplazamientos[max(range(4), key=lambda a: (Q[f, c, a], -a))]
        f, c = min(max(f + df, 0), FILAS - 1), min(max(c + dc, 0), COLUMNAS - 1)
        camino.append((f, c))
        if (f, c) in TERMINALES:
            break
    return camino


class TestCaminoVoraz(unittest.TestCase):
    # Subir por la columna izquierda y luego ir a la derecha por la fila de arriba.
    POR_LA_IZQUIERDA = {(3, 0): ARRIBA, (2, 0): ARRIBA, (1, 0): ARRIBA,
                        (0, 0): DERECHA, (0, 1): DERECHA, (0, 2): DERECHA}
    # Ir a la derecha por abajo, subir por la tercera columna y girar a la derecha arriba.
    POR_LA_DERECHA = {(3, 0): DERECHA, (3, 1): DERECHA, (3, 2): ARRIBA, (2, 2): ARRIBA,
                      (1, 2): ARRIBA, (0, 2): DERECHA}

    # --- caminos construidos a mano ---
    def test_camino_por_la_izquierda(self):
        camino = camino_voraz(q_de_politica(self.POR_LA_IZQUIERDA))
        self.assertEqual(camino, [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3)])

    def test_camino_por_la_derecha(self):
        camino = camino_voraz(q_de_politica(self.POR_LA_DERECHA))
        self.assertEqual(camino, [(3, 0), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3)])

    def test_subir_por_la_derecha_cae_en_la_trampa(self):
        politica = {(3, 0): DERECHA, (3, 1): DERECHA, (3, 2): DERECHA, (3, 3): ARRIBA}
        camino = camino_voraz(q_de_politica(politica))
        self.assertEqual(camino, [(3, 0), (3, 1), (3, 2), (3, 3), (2, 3)])

    def test_se_detiene_al_caer_en_cada_trampa(self):
        casos = {
            (1, 1): {(3, 0): ARRIBA, (2, 0): DERECHA, (2, 1): ARRIBA},
            (2, 3): {(3, 0): DERECHA, (3, 1): DERECHA, (3, 2): ARRIBA, (2, 2): DERECHA},
        }
        for trampa, politica in casos.items():
            with self.subTest(trampa=trampa):
                camino = camino_voraz(q_de_politica(politica))
                self.assertEqual(camino[-1], trampa)
                self.assertEqual(camino.count(trampa), 1)
                self.assertEqual(len(camino) - 1, 3 if trampa == (1, 1) else 4)

    def test_no_sigue_despues_de_la_meta_aunque_queden_pasos(self):
        camino = camino_voraz(q_de_politica(self.POR_LA_IZQUIERDA), max_pasos=1000)
        self.assertEqual(len(camino), 7)
        self.assertEqual(camino.count(META), 1)

    # --- límite de pasos ---
    def test_tabla_a_ceros_sube_y_choca_con_el_borde(self):
        # Con empate total, argmax elige la primera acción (arriba).
        camino = camino_voraz(np.zeros((FILAS, COLUMNAS, 4)))
        self.assertEqual(camino[:4], [(3, 0), (2, 0), (1, 0), (0, 0)])
        self.assertEqual(camino[4:], [(0, 0)] * 17)
        self.assertEqual(len(camino), 21)

    def test_bucle_entre_dos_casillas_se_corta_en_max_pasos(self):
        politica = {(3, 0): DERECHA, (3, 1): IZQUIERDA}
        camino = camino_voraz(q_de_politica(politica))
        self.assertEqual(len(camino), 21)
        self.assertEqual(camino, [(3, 0), (3, 1)] * 10 + [(3, 0)])

    def test_max_pasos_controla_la_longitud(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))  # nunca termina
        for max_pasos in (0, 1, 2, 5, 20, 100):
            with self.subTest(max_pasos=max_pasos):
                self.assertEqual(len(camino_voraz(Q, max_pasos)), max_pasos + 1)

    def test_max_pasos_cero_solo_devuelve_el_inicio(self):
        self.assertEqual(camino_voraz(q_de_politica(self.POR_LA_IZQUIERDA), 0), [INICIO])

    def test_llega_justo_en_el_ultimo_paso_permitido(self):
        Q = q_de_politica(self.POR_LA_IZQUIERDA)
        self.assertEqual(camino_voraz(Q, 6)[-1], META)
        self.assertEqual(camino_voraz(Q, 5)[-1], (0, 2))

    # --- cómo elige la acción ---
    def test_en_empate_elige_la_primera_accion(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [0.0, 3.0, 3.0, 3.0]  # empatan abajo, izquierda y derecha
        self.assertEqual(camino_voraz(Q, 1), [INICIO, INICIO])  # abajo choca con el borde
        Q[INICIO] = [0.0, 0.0, 3.0, 3.0]
        self.assertEqual(camino_voraz(Q, 1), [INICIO, INICIO])  # izquierda choca con el borde
        Q[INICIO] = [0.0, 0.0, 0.0, 3.0]
        self.assertEqual(camino_voraz(Q, 1), [INICIO, (3, 1)])

    def test_funciona_con_valores_negativos(self):
        Q = np.full((FILAS, COLUMNAS, 4), -5.0)
        for estado, accion in self.POR_LA_IZQUIERDA.items():
            Q[estado][accion] = -0.1
        self.assertEqual(camino_voraz(Q)[-1], META)

    def test_ignora_las_casillas_fuera_del_camino(self):
        Q = q_de_politica(self.POR_LA_IZQUIERDA)
        Q_alterada = Q.copy()
        for estado in [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (1, 2), (1, 3)]:
            Q_alterada[estado] = np.random.default_rng(0).normal(size=4) * 100
        self.assertEqual(camino_voraz(Q_alterada), camino_voraz(Q))

    # --- propiedades generales ---
    def test_coincide_con_un_modelo_independiente_en_tablas_aleatorias(self):
        rng = np.random.default_rng(0)
        for i in range(2000):
            Q = rng.normal(size=(FILAS, COLUMNAS, 4))
            if i % 2:
                Q = np.round(Q)  # muchos empates
            with self.subTest(i=i):
                self.assertEqual(camino_voraz(Q), camino_esperado(Q))

    def test_todas_las_politicas_fijas(self):
        # La misma acción en todas las casillas: 4 políticas posibles.
        esperados = {
            ARRIBA: [(3, 0), (2, 0), (1, 0), (0, 0)] + [(0, 0)] * 17,
            ABAJO: [(3, 0)] * 21,
            IZQUIERDA: [(3, 0)] * 21,
            DERECHA: [(3, 0), (3, 1), (3, 2), (3, 3)] + [(3, 3)] * 17,
        }
        for accion, esperado in esperados.items():
            with self.subTest(accion=accion):
                self.assertEqual(camino_voraz(q_de_politica({}, por_defecto=accion)), esperado)

    def test_estructura_del_camino(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            camino = camino_voraz(rng.normal(size=(FILAS, COLUMNAS, 4)))
            self.assertEqual(camino[0], INICIO)
            self.assertTrue(1 <= len(camino) <= 21)
            for (f, c), (nf, nc) in zip(camino, camino[1:]):
                self.assertLessEqual(abs(nf - f) + abs(nc - c), 1)
            # Solo la última casilla puede ser la meta o una trampa.
            self.assertTrue(TERMINALES.isdisjoint(camino[:-1]))
            if len(camino) < 21:
                self.assertIn(camino[-1], TERMINALES)

    def test_devuelve_una_lista_de_tuplas(self):
        camino = camino_voraz(q_de_politica(self.POR_LA_IZQUIERDA))
        self.assertIsInstance(camino, list)
        for estado in camino:
            self.assertIsInstance(estado, tuple)
            self.assertEqual(len(estado), 2)

    def test_no_modifica_la_tabla_q(self):
        Q = np.random.default_rng(2).normal(size=(FILAS, COLUMNAS, 4))
        copia = Q.copy()
        camino_voraz(Q)
        np.testing.assert_array_equal(Q, copia)

    def test_es_determinista(self):
        Q = np.round(np.random.default_rng(3).normal(size=(FILAS, COLUMNAS, 4)))
        self.assertEqual(camino_voraz(Q), camino_voraz(Q))

    def test_con_una_tabla_entrenada_llega_a_la_meta(self):
        Q, _ = entrenar(verbose=False)
        camino = camino_voraz(Q)
        self.assertEqual(camino[-1], META)
        self.assertEqual(len(camino) - 1, 6)


def capturar_salida(funcion, *args):
    salida = io.StringIO()
    with redirect_stdout(salida):
        devuelto = funcion(*args)
    return salida.getvalue(), devuelto


def tablero_politica(Q):
    """Símbolos que imprime imprimir_politica, como una lista de filas."""
    texto, _ = capturar_salida(imprimir_politica, Q)
    return [linea.split() for linea in texto.splitlines()]


class TestImprimirPolitica(unittest.TestCase):
    SIMBOLO = {ARRIBA: '↑', ABAJO: '↓', IZQUIERDA: '←', DERECHA: '→'}

    # --- salida exacta ---
    def test_tabla_a_ceros(self):
        # Con empate total, argmax elige la primera acción (arriba).
        texto, _ = capturar_salida(imprimir_politica, np.zeros((FILAS, COLUMNAS, 4)))
        self.assertEqual(texto, '  ↑ ↑ ↑ ★\n'
                                '  ↑ X ↑ ↑\n'
                                '  ↑ ↑ ↑ X\n'
                                '  ↑ ↑ ↑ ↑\n')

    def test_tabla_entrenada(self):
        Q, _ = entrenar(verbose=False)
        texto, _ = capturar_salida(imprimir_politica, Q)
        self.assertEqual(texto, '  → → → ★\n'
                                '  ↑ X ↑ ↑\n'
                                '  ↑ ← ↑ X\n'
                                '  ↑ ← ↑ →\n')

    def test_la_misma_accion_en_todas_las_casillas(self):
        for accion, a in self.SIMBOLO.items():
            with self.subTest(accion=accion):
                texto, _ = capturar_salida(imprimir_politica, q_de_politica({}, por_defecto=accion))
                self.assertEqual(texto, f'  {a} {a} {a} ★\n'
                                        f'  {a} X {a} {a}\n'
                                        f'  {a} {a} {a} X\n'
                                        f'  {a} {a} {a} {a}\n')

    def test_camino_a_la_meta(self):
        Q = q_de_politica(TestCaminoVoraz.POR_LA_IZQUIERDA)
        texto, _ = capturar_salida(imprimir_politica, Q)
        self.assertEqual(texto, '  → → → ★\n'
                                '  ↑ X ↑ ↑\n'
                                '  ↑ ↑ ↑ X\n'
                                '  ↑ ↑ ↑ ↑\n')

    # --- casilla a casilla ---
    def test_cada_casilla_con_cada_accion(self):
        # Cambia una sola casilla y comprueba que solo cambia su símbolo.
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                for accion, flecha in self.SIMBOLO.items():
                    tablero = tablero_politica(q_de_politica({(f, c): accion}, por_defecto=ABAJO))
                    otras = {tablero[i][j] for i in range(FILAS) for j in range(COLUMNAS)
                             if (i, j) not in TERMINALES and (i, j) != (f, c)}
                    with self.subTest(estado=(f, c), accion=accion):
                        self.assertEqual(tablero[f][c], flecha)
                        self.assertEqual(otras, {'↓'})

    def test_meta_y_trampas_ignoran_sus_valores(self):
        Q = np.random.default_rng(0).normal(size=(FILAS, COLUMNAS, 4)) * 100
        tablero = tablero_politica(Q)
        self.assertEqual(tablero[0][3], '★')
        self.assertEqual(tablero[1][1], 'X')
        self.assertEqual(tablero[2][3], 'X')

    def test_cuenta_de_simbolos(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            simbolos = [s for fila in tablero_politica(rng.normal(size=(FILAS, COLUMNAS, 4)))
                        for s in fila]
            self.assertEqual(len(simbolos), FILAS * COLUMNAS)
            self.assertEqual(simbolos.count('★'), 1)
            self.assertEqual(simbolos.count('X'), 2)
            self.assertEqual(sum(s in '↑↓←→' for s in simbolos), 13)

    # --- cómo elige la flecha ---
    def test_en_empate_muestra_la_primera_accion(self):
        casos = {
            ARRIBA: [1.0, 1.0, 1.0, 1.0],
            ABAJO: [0.0, 2.0, 2.0, 2.0],
            IZQUIERDA: [0.0, 0.0, 2.0, 2.0],
            DERECHA: [0.0, 0.0, 0.0, 2.0],
        }
        for accion, valores in casos.items():
            with self.subTest(accion=accion):
                Q = np.zeros((FILAS, COLUMNAS, 4))
                Q[INICIO] = valores
                self.assertEqual(tablero_politica(Q)[3][0], self.SIMBOLO[accion])

    def test_valores_negativos(self):
        Q = np.full((FILAS, COLUMNAS, 4), -5.0)
        Q[INICIO] = [-9.0, -8.0, -7.0, -0.1]
        self.assertEqual(tablero_politica(Q)[3][0], '→')

    def test_diferencias_minimas(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [1.0, 1.0, 1.0 + 1e-12, 1.0]
        self.assertEqual(tablero_politica(Q)[3][0], '←')

    def test_coincide_con_un_modelo_independiente_en_tablas_aleatorias(self):
        rng = np.random.default_rng(2)
        for i in range(1000):
            Q = rng.normal(size=(FILAS, COLUMNAS, 4))
            if i % 2:
                Q = np.round(Q)  # muchos empates
            esperado = []
            for f in range(FILAS):
                fila = []
                for c in range(COLUMNAS):
                    if (f, c) == META:
                        fila.append('★')
                    elif (f, c) in TRAMPAS:
                        fila.append('X')
                    else:
                        fila.append(self.SIMBOLO[max(range(4), key=lambda a: (Q[f, c, a], -a))])
                esperado.append(fila)
            with self.subTest(i=i):
                self.assertEqual(tablero_politica(Q), esperado)

    # --- formato ---
    def test_formato_de_cada_linea(self):
        texto, _ = capturar_salida(imprimir_politica, np.random.default_rng(3).normal(size=(4, 4, 4)))
        self.assertTrue(texto.endswith('\n'))
        lineas = texto.splitlines()
        self.assertEqual(len(lineas), FILAS)
        for linea in lineas:
            with self.subTest(linea=linea):
                self.assertTrue(linea.startswith('  '))
                self.assertEqual(len(linea), 2 + 2 * COLUMNAS - 1)  # sin espacio final
                self.assertEqual(linea[3::2], ' ' * (COLUMNAS - 1))  # un espacio entre símbolos

    def test_devuelve_none(self):
        _, devuelto = capturar_salida(imprimir_politica, np.zeros((FILAS, COLUMNAS, 4)))
        self.assertIsNone(devuelto)

    def test_no_modifica_la_tabla_q(self):
        Q = np.random.default_rng(4).normal(size=(FILAS, COLUMNAS, 4))
        copia = Q.copy()
        capturar_salida(imprimir_politica, Q)
        np.testing.assert_array_equal(Q, copia)


CABECERA_TABLA = 'Casilla          ↑       ↓       ←       →'
CASILLAS_NUMERADAS = [1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16]  # sin meta (4) ni trampas (6, 12)


def filas_tabla(Q):
    """Filas de imprimir_tabla como {número de casilla: [cuatro textos]}."""
    texto, _ = capturar_salida(imprimir_tabla, Q)
    return {int(linea.split()[0]): linea.split()[1:] for linea in texto.splitlines()[1:]}


class TestImprimirTabla(unittest.TestCase):
    # --- salida exacta ---
    def test_tabla_a_ceros(self):
        texto, _ = capturar_salida(imprimir_tabla, np.zeros((FILAS, COLUMNAS, 4)))
        esperado = [CABECERA_TABLA] + [f'{n:>7}        0.0     0.0     0.0     0.0'
                                       for n in CASILLAS_NUMERADAS]
        self.assertEqual(texto, '\n'.join(esperado) + '\n')

    def test_tabla_entrenada(self):
        Q, _ = entrenar(verbose=False)
        texto, _ = capturar_salida(imprimir_tabla, Q)
        self.assertEqual(texto, CABECERA_TABLA + '\n'
                                '      1        4.6     3.1     4.6     6.2\n'
                                '      2        6.2   -10.0     4.6     8.0\n'
                                '      3        8.0     6.1     6.2    10.0\n'
                                '      5        4.6     1.8     3.1   -10.0\n'
                                '      7        8.0    -1.3    -7.5     1.5\n'
                                '      8        7.5     0.0     0.0     0.0\n'
                                '      9        3.1     0.6     1.8     0.6\n'
                                '     10      -10.0    -2.0     1.8     0.0\n'
                                '     11        3.1    -1.9    -2.3    -9.9\n'
                                '     13        1.8     0.6     0.6    -0.4\n'
                                '     14       -0.5    -3.0     0.6    -2.0\n'
                                '     15       -1.0    -2.2    -1.9    -1.8\n'
                                '     16       -9.7    -2.1    -2.2    -1.7\n')

    # --- casillas ---
    def test_numera_las_casillas_de_1_a_16_por_filas(self):
        self.assertEqual(list(filas_tabla(np.zeros((FILAS, COLUMNAS, 4)))), CASILLAS_NUMERADAS)

    def test_omite_la_meta_y_las_trampas_aunque_tengan_valores(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        for estado in TERMINALES:
            Q[estado] = [1.0, 2.0, 3.0, 4.0]
        texto, _ = capturar_salida(imprimir_tabla, Q)
        self.assertNotIn('1.0', texto)
        self.assertNotIn('4.0', texto)

    def test_cada_casilla_y_accion_en_su_sitio(self):
        # Pone un valor distinto en una sola posición y lo busca en la tabla.
        for f in range(FILAS):
            for c in range(COLUMNAS):
                if (f, c) in TERMINALES:
                    continue
                for accion in range(4):
                    Q = np.zeros((FILAS, COLUMNAS, 4))
                    Q[f, c, accion] = 7.0
                    filas = filas_tabla(Q)
                    with self.subTest(estado=(f, c), accion=accion):
                        for numero, valores in filas.items():
                            for a, texto in enumerate(valores):
                                es_la_posicion = numero == f * COLUMNAS + c + 1 and a == accion
                                self.assertEqual(texto, '7.0' if es_la_posicion else '0.0')

    # --- valores ---
    def test_redondea_a_un_decimal(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [1.23, -4.56, 7.89, -0.12]
        self.assertEqual(filas_tabla(Q)[13], ['1.2', '-4.6', '7.9', '-0.1'])

    def test_nunca_muestra_menos_cero(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [-0.0, -0.01, -0.04, -0.05]
        self.assertEqual(filas_tabla(Q)[13], ['0.0', '0.0', '0.0', '0.0'])

    def test_los_medios_se_redondean_al_par(self):
        # numpy redondea x.x5 al decimal par: 0.05 -> 0.0, 0.15 -> 0.2, 0.25 -> 0.2, 0.35 -> 0.4
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [0.05, 0.15, 0.25, 0.35]
        self.assertEqual(filas_tabla(Q)[13], ['0.0', '0.2', '0.2', '0.4'])

    def test_extremos_del_rango_de_q(self):
        Q = np.zeros((FILAS, COLUMNAS, 4))
        Q[INICIO] = [10.0, -10.0, 9.99, -9.99]
        self.assertEqual(filas_tabla(Q)[13], ['10.0', '-10.0', '10.0', '-10.0'])

    def test_coincide_con_un_modelo_independiente_en_tablas_aleatorias(self):
        rng = np.random.default_rng(0)
        for i in range(500):
            Q = rng.uniform(-10, 10, size=(FILAS, COLUMNAS, 4))
            esperado = {f * COLUMNAS + c + 1: [f'{np.round(v, 1) + 0.0:.1f}' for v in Q[f, c]]
                        for f in range(FILAS) for c in range(COLUMNAS) if (f, c) not in TERMINALES}
            with self.subTest(i=i):
                self.assertEqual(filas_tabla(Q), esperado)

    # --- formato ---
    def test_cabecera(self):
        texto, _ = capturar_salida(imprimir_tabla, np.zeros((FILAS, COLUMNAS, 4)))
        self.assertEqual(texto.splitlines()[0], CABECERA_TABLA)

    def test_numero_de_lineas(self):
        texto, _ = capturar_salida(imprimir_tabla, np.zeros((FILAS, COLUMNAS, 4)))
        self.assertTrue(texto.endswith('\n'))
        self.assertEqual(len(texto.splitlines()), 1 + FILAS * COLUMNAS - len(TERMINALES))

    def test_columnas_alineadas_con_valores_entre_menos_diez_y_diez(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            texto, _ = capturar_salida(imprimir_tabla, rng.uniform(-10, 10, size=(FILAS, COLUMNAS, 4)))
            lineas = texto.splitlines()
            for linea in lineas:
                self.assertEqual(len(linea), 12 + 4 * 6 + 3 * 2)
            # Cada flecha queda encima del último carácter de su columna de valores.
            for k in range(4):
                fin = 12 + 6 * (k + 1) + 2 * k - 1
                self.assertIn(lineas[0][fin], '↑↓←→')
                for linea in lineas[1:]:
                    self.assertTrue(linea[fin].isdigit())

    def test_devuelve_none(self):
        _, devuelto = capturar_salida(imprimir_tabla, np.zeros((FILAS, COLUMNAS, 4)))
        self.assertIsNone(devuelto)

    def test_no_modifica_la_tabla_q(self):
        Q = np.random.default_rng(2).normal(size=(FILAS, COLUMNAS, 4))
        copia = Q.copy()
        capturar_salida(imprimir_tabla, Q)
        np.testing.assert_array_equal(Q, copia)


if __name__ == '__main__':
    unittest.main()
