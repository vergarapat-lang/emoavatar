"""
EmoAvatar - logica compartida (sin nada de camara ni ventanas).

Este modulo lo usan tanto avatar_pilot.py (version de escritorio, con
cv2.imshow) como streamlit_app.py (version web, con streamlit-webrtc).
Contiene: manejo de blendshapes, calibracion, suavizado temporal, calculo
de las 6 emociones objetivo, y el dibujo del avatar + dashboard.
"""

import json
import os

import cv2
import numpy as np

CANVAS_SIZE = (480, 480)  # ancho, alto del lienzo del avatar
CALIBRATION_PATH = "calibracion.json"

# --- Parametros de suavizado y calibracion ---
SMOOTHING_ALPHA = 0.35   # 0 = sin cambio (ignora frame nuevo), 1 = sin suavizado
CALIBRATION_FRAMES = 30  # ~1 segundo a 30 fps
UMBRAL_EMOCION = 0.30    # score minimo para que una emocion "gane" sobre Neutral

# Blendshapes que nos interesan calibrar/suavizar (los demas se ignoran)
BLENDSHAPES_CLAVE = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
    "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthStretchLeft", "mouthStretchRight",
    "browInnerUp", "browDownLeft", "browDownRight",
    "browOuterUpLeft", "browOuterUpRight",
]


def get_blendshape_dict(result):
    """Convierte la lista de blendshapes de MediaPipe en un dict {nombre: score}."""
    if not result.face_blendshapes:
        return {}
    return {b.category_name: b.score for b in result.face_blendshapes[0]}


def cargar_calibracion(path=CALIBRATION_PATH):
    """Carga el baseline guardado en disco, si existe."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def guardar_calibracion(baseline, path=CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def aplicar_calibracion(bs, baseline):
    """Resta el baseline personal a cada blendshape y recorta a [0, 1]."""
    if not baseline:
        return bs
    corregido = {}
    for k, v in bs.items():
        base = baseline.get(k, 0.0)
        rango = max(1e-6, 1.0 - base)
        corregido[k] = max(0.0, min(1.0, (v - base) / rango))
    return corregido


def suavizar(bs_nuevo, bs_suavizado_prev, alpha=SMOOTHING_ALPHA):
    """Media movil exponencial: mezcla el valor nuevo con el historico."""
    if bs_suavizado_prev is None:
        return dict(bs_nuevo)
    resultado = {}
    claves = set(bs_nuevo) | set(bs_suavizado_prev)
    for k in claves:
        nuevo = bs_nuevo.get(k, 0.0)
        prev = bs_suavizado_prev.get(k, nuevo)
        resultado[k] = alpha * nuevo + (1 - alpha) * prev
    return resultado


def calcular_metricas(bs):
    """Calcula un score 0-1 para cada una de las 6 emociones objetivo,
    combinando varios blendshapes relacionados con cada una (heuristica
    simple, no un modelo entrenado)."""
    sonrisa = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2
    frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2
    brow_inner_up = bs.get("browInnerUp", 0.0)
    brow_outer = (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)) / 2
    cejas_arriba = (brow_inner_up + bs.get("browOuterUpLeft", 0.0)
                    + bs.get("browOuterUpRight", 0.0)) / 3
    cejas_abajo = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2
    ojos_grandes = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2
    boca_abierta = bs.get("jawOpen", 0.0)
    boca_tensa = (bs.get("mouthStretchLeft", 0.0) + bs.get("mouthStretchRight", 0.0)) / 2

    alegria = sonrisa
    # El blendshape "frown" de boca suele dar valores bajos aunque la persona
    # frunza bastante; se amplifica y se combina con "ceja solo interior
    # arriba" (signo clasico de tristeza/preocupacion, distinto de la ceja
    # completa arriba de sorpresa/miedo).
    frown_amplificado = min(1.0, frown * 1.8)
    ceja_solo_interior = max(0.0, brow_inner_up - 0.6 * brow_outer)
    tristeza = max(0.0, 0.55 * frown_amplificado + 0.45 * ceja_solo_interior - 0.3 * sonrisa)
    enojo = max(0.0, cejas_abajo - 0.3 * cejas_arriba)
    miedo = max(0.0, 0.6 * ojos_grandes + 0.25 * cejas_arriba + 0.15 * boca_tensa)
    sorpresa = max(0.0, 0.55 * boca_abierta + 0.45 * cejas_arriba - 0.3 * ojos_grandes)

    principales = {"Alegria": alegria, "Tristeza": tristeza, "Enojo": enojo,
                   "Miedo": miedo, "Sorpresa": sorpresa}
    neutral = max(0.0, 1.0 - max(principales.values()))

    return [
        ("Alegria", alegria), ("Tristeza", tristeza), ("Enojo", enojo),
        ("Miedo", miedo), ("Sorpresa", sorpresa), ("Neutral", neutral),
    ]


def clasificar_expresion(metricas):
    """Elige la emocion con mayor score (excepto Neutral) si supera el umbral;
    si ninguna lo supera, devuelve Neutral."""
    principales = [(n, v) for n, v in metricas if n != "Neutral"]
    nombre, valor = max(principales, key=lambda item: item[1])
    if valor >= UMBRAL_EMOCION:
        return nombre
    return "Neutral"


def _curva_labio(cx, y_base, ancho, curva, n=13):
    """Genera puntos de una curva parabolica para un labio.
    curva > 0 => corners levantados (sonrisa). curva < 0 => corners caidos (tristeza)."""
    pts = []
    for i in range(n):
        t = -1.0 + 2.0 * i / (n - 1)
        x = int(cx + t * ancho)
        y = int(y_base - curva * (t ** 2))
        pts.append((x, y))
    return pts


def dibujar_avatar(bs, expresion=None):
    """Dibuja un avatar caricaturesco (piel, pelo, ojos con iris, labios),
    animado según blendshapes, con soporte visual para 6 emociones."""
    w, h = CANVAS_SIZE
    canvas = np.full((h, w, 3), (250, 245, 240), dtype=np.uint8)

    cx, cy = w // 2, h // 2

    COLOR_PELO = (40, 70, 100)
    COLOR_PIEL = (185, 210, 245)
    COLOR_PIEL_SOMBRA = (165, 190, 225)
    COLOR_OJO_BLANCO = (250, 250, 250)
    COLOR_IRIS = (90, 60, 30)
    COLOR_PUPILA = (20, 15, 10)
    COLOR_CEJA = (35, 60, 90)
    COLOR_LABIO = (95, 90, 200)
    COLOR_BOCA_INTERIOR = (60, 45, 150)
    COLOR_MEJILLA = (170, 175, 250)
    AA = cv2.LINE_AA

    # --- Pelo, orejas, cara, flequillo ---
    cv2.ellipse(canvas, (cx, cy - 15), (158, 190), 0, 0, 360, COLOR_PELO, -1, AA)
    for signo in (-1, 1):
        cv2.ellipse(canvas, (cx + signo * 138, cy + 5), (18, 26), 0, 0, 360, COLOR_PIEL, -1, AA)
    cv2.ellipse(canvas, (cx, cy + 8), (132, 158), 0, 0, 360, COLOR_PIEL, -1, AA)
    cv2.ellipse(canvas, (cx, cy + 70), (110, 60), 0, 0, 180, COLOR_PIEL_SOMBRA, 2, AA)
    cv2.ellipse(canvas, (cx, cy - 95), (140, 60), 0, 180, 360, COLOR_PELO, -1, AA)

    # --- Valores base de blendshapes ---
    blink_l = bs.get("eyeBlinkLeft", 0.0)
    blink_r = bs.get("eyeBlinkRight", 0.0)
    ojos_grandes = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2
    brow_inner_up = bs.get("browInnerUp", 0.0)
    brow_down_l = bs.get("browDownLeft", 0.0)
    brow_down_r = bs.get("browDownRight", 0.0)
    brow_outer_l = bs.get("browOuterUpLeft", 0.0)
    brow_outer_r = bs.get("browOuterUpRight", 0.0)

    eye_w = 32 + int(6 * ojos_grandes)
    eye_h_base = 24 + int(8 * ojos_grandes)
    left_eye_center = (cx - 52, cy - 20)
    right_eye_center = (cx + 52, cy - 20)

    # --- Ojos ---
    for center, blink in [(left_eye_center, blink_l), (right_eye_center, blink_r)]:
        eye_h = max(1, int(eye_h_base * (1 - blink)))
        if eye_h > 5:
            cv2.ellipse(canvas, center, (eye_w, eye_h), 0, 0, 360, COLOR_OJO_BLANCO, -1, AA)
            iris_r = min(15, eye_h)
            cv2.circle(canvas, center, iris_r, COLOR_IRIS, -1, AA)
            cv2.circle(canvas, center, max(3, iris_r // 3), COLOR_PUPILA, -1, AA)
            cv2.circle(canvas, (center[0] - 4, center[1] - 4), 2, (255, 255, 255), -1, AA)
            cv2.ellipse(canvas, center, (eye_w, eye_h), 0, 0, 360, (60, 50, 40), 2, AA)
        else:
            cv2.line(canvas, (center[0] - eye_w, center[1]), (center[0] + eye_w, center[1]),
                     (60, 50, 40), 3, AA)

    # --- Cejas: puntos independientes interior/exterior para permitir el
    # gesto "triste" (solo interior sube) distinto de "miedo/sorpresa"
    # (interior y exterior suben juntos) ---
    base_y = left_eye_center[1] - 42
    for lado, eye_center, b_out, b_down in [
        (-1, left_eye_center, brow_outer_l, brow_down_l),
        (1, right_eye_center, brow_outer_r, brow_down_r),
    ]:
        ex = eye_center[0]
        off_outer = int(-15 * b_out + 15 * b_down)
        off_inner = int(-26 * brow_inner_up + 15 * b_down)
        outer_pt = (ex - lado * 30, base_y + off_outer)
        inner_pt = (ex + lado * 30, base_y + off_inner)
        mid_pt = ((outer_pt[0] + inner_pt[0]) // 2,
                  (outer_pt[1] + inner_pt[1]) // 2 - 3)
        cv2.polylines(canvas, [np.array([outer_pt, mid_pt, inner_pt])],
                       False, COLOR_CEJA, 5, AA)

    # --- Mejillas sonrosadas ---
    sonrisa = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2
    blush_r = int(20 + 6 * sonrisa)
    cv2.circle(canvas, (cx - 85, cy + 35), blush_r, COLOR_MEJILLA, -1, AA)
    cv2.circle(canvas, (cx + 85, cy + 35), blush_r, COLOR_MEJILLA, -1, AA)

    # --- Nariz ---
    cv2.ellipse(canvas, (cx, cy + 15), (6, 10), 0, 20, 160, (140, 160, 210), 2, AA)

    # --- Boca: curva parabolica que soporte sonrisa Y tristeza ---
    jaw_open = bs.get("jawOpen", 0.0)
    frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2
    boca_tensa = (bs.get("mouthStretchLeft", 0.0) + bs.get("mouthStretchRight", 0.0)) / 2

    mouth_cy = cy + 78
    mouth_w = 46 + int(20 * sonrisa) + int(16 * boca_tensa)
    mouth_h = int(8 + 42 * jaw_open)
    curvatura = int(16 * (sonrisa - frown))
    curvatura = max(-16, min(16, curvatura))

    labio_sup = _curva_labio(cx, mouth_cy, mouth_w, curvatura)

    if jaw_open > 0.12:
        labio_inf = _curva_labio(cx, mouth_cy + max(4, mouth_h), int(mouth_w * 0.85), curvatura * 0.5)
        poligono = np.array(labio_sup + labio_inf[::-1])
        cv2.fillPoly(canvas, [poligono], COLOR_BOCA_INTERIOR)
        if jaw_open > 0.4:
            cv2.rectangle(canvas, (cx - mouth_w // 2 + 6, mouth_cy - 2),
                           (cx + mouth_w // 2 - 6, mouth_cy + 6), (245, 245, 245), -1, AA)
        cv2.polylines(canvas, [np.array(labio_sup)], False, COLOR_LABIO, 5, AA)
        cv2.polylines(canvas, [np.array(labio_inf)], False, COLOR_LABIO, 4, AA)
    else:
        # boca cerrada: una sola linea curva (evita el efecto "ondulado"
        # de dos lineas casi superpuestas)
        cv2.polylines(canvas, [np.array(labio_sup)], False, COLOR_LABIO, 5, AA)

    # --- Etiqueta de expresión detectada ---
    if expresion:
        cv2.putText(canvas, expresion, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (40, 30, 20), 2, AA)

    return canvas


def crear_dashboard(frame, avatar, metricas, fps, calibrado, calibrando, frames_restantes):
    """Compone el panel final: header, panel de camara, panel de metricas
    (6 emociones), panel de estado de conexion, y panel del avatar."""
    DASH_W, DASH_H = 1000, 720
    canvas = np.full((DASH_H, DASH_W, 3), (248, 246, 244), dtype=np.uint8)
    AA = cv2.LINE_AA

    COLOR_TITULO = (50, 35, 20)
    COLOR_SUBT = (120, 105, 90)
    PANEL_OSCURO = (38, 28, 24)
    TEXT_CLARO = (235, 235, 235)
    TEXT_TENUE = (175, 175, 180)
    ACCENT = (246, 158, 59)
    BAR_BG = (70, 60, 55)
    VERDE = (90, 200, 90)
    NARANJA = (0, 150, 255)
    ROJO = (60, 60, 210)

    cv2.putText(canvas, "EMOAVATAR", (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_TITULO, 2, AA)
    cv2.putText(canvas, "Reconocimiento e imitacion de emociones", (32, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SUBT, 1, AA)

    # --- Panel camara ---
    cx0, cy0, cx1, cy1 = 30, 110, 460, 430
    cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), PANEL_OSCURO, -1, AA)
    cv2.putText(canvas, "TU CAMARA", (cx0 + 15, cy0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_CLARO, 1, AA)
    fw, fh = cx1 - cx0 - 30, cy1 - cy0 - 60
    frame_resized = cv2.resize(frame, (fw, fh))
    fx, fy = cx0 + 15, cy0 + 45
    canvas[fy:fy + fh, fx:fx + fw] = frame_resized

    # --- Panel metricas (6 emociones) ---
    mx0, my0, mx1, my1 = 30, 440, 460, 650
    cv2.rectangle(canvas, (mx0, my0), (mx1, my1), PANEL_OSCURO, -1, AA)
    cv2.putText(canvas, "EMOCIONES DETECTADAS", (mx0 + 15, my0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_CLARO, 1, AA)
    bar_x0 = mx0 + 110
    bar_w = mx1 - bar_x0 - 55
    for i, (nombre, valor) in enumerate(metricas):
        y = my0 + 55 + i * 27
        cv2.putText(canvas, nombre, (mx0 + 15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_TENUE, 1, AA)
        cv2.rectangle(canvas, (bar_x0, y - 7), (bar_x0 + bar_w, y + 2), BAR_BG, -1, AA)
        relleno = int(bar_w * max(0.0, min(1.0, valor)))
        if relleno > 0:
            cv2.rectangle(canvas, (bar_x0, y - 7), (bar_x0 + relleno, y + 2), ACCENT, -1, AA)
        cv2.putText(canvas, f"{valor:.2f}", (mx1 - 42, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_CLARO, 1, AA)

    # --- Panel estado de conexion / calibracion ---
    sx0, sy0, sx1, sy1 = 30, 660, 460, 705
    cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), PANEL_OSCURO, -1, AA)
    cv2.circle(canvas, (sx0 + 18, sy0 + 22), 6, VERDE, -1, AA)
    cv2.putText(canvas, "En tiempo real", (sx0 + 32, sy0 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_CLARO, 1, AA)
    cv2.putText(canvas, f"FPS: {fps:.0f}", (sx1 - 95, sy0 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_CLARO, 1, AA)

    # --- Panel avatar ---
    ax0, ay0, ax1, ay1 = 480, 110, 970, 705
    cv2.rectangle(canvas, (ax0, ay0), (ax1, ay1), (238, 242, 247), -1, AA)
    cv2.rectangle(canvas, (ax0, ay0), (ax1, ay1), (210, 215, 222), 2, AA)
    cv2.putText(canvas, "AVATAR", (ax0 + 15, ay0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TITULO, 1, AA)

    if calibrando:
        estado, color = f"Calibrando... {frames_restantes}", NARANJA
    elif calibrado:
        estado, color = "Calibrado", VERDE
    else:
        estado, color = "Sin calibrar (c)", ROJO
    (tw, _), _ = cv2.getTextSize(estado, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(canvas, estado, (ax1 - tw - 15, ay0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, AA)

    avatar_size = min(ax1 - ax0 - 40, ay1 - ay0 - 70)
    avatar_resized = cv2.resize(avatar, (avatar_size, avatar_size))
    ax_off = ax0 + ((ax1 - ax0) - avatar_size) // 2
    ay_off = ay0 + 55
    canvas[ay_off:ay_off + avatar_size, ax_off:ax_off + avatar_size] = avatar_resized

    return canvas
