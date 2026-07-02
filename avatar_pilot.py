"""
EmoAvatar - version de escritorio (Mac/Windows/Linux con webcam local).

Requisitos:
  pip install -r requirements-desktop.txt
  Descargar face_landmarker.task (ver README.md)

Controles:
  q  -> salir
  s  -> guardar captura de pantalla del avatar actual
  c  -> calibrar (mantén el rostro neutral ~1 segundo)
  r  -> reiniciar calibracion
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from emoavatar_core import (
    BLENDSHAPES_CLAVE,
    CALIBRATION_PATH,
    CALIBRATION_FRAMES,
    aplicar_calibracion,
    cargar_calibracion,
    calcular_metricas,
    clasificar_expresion,
    crear_dashboard,
    dibujar_avatar,
    get_blendshape_dict,
    guardar_calibracion,
    suavizar,
)

MODEL_PATH = "face_landmarker.task"
CAM_INDEX = 0


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Revisa CAM_INDEX o permisos de cámara en macOS "
              "(Ajustes del Sistema > Privacidad y seguridad > Cámara).")
        return

    baseline = cargar_calibracion()
    calibrado = bool(baseline)
    calibrando = False
    muestras_calibracion = []

    bs_suavizado = None
    frame_idx = 0
    fps_suavizado = 30.0
    tiempo_anterior = time.time()

    print("EmoAvatar corriendo.")
    print("  q -> salir | s -> guardar captura | c -> calibrar | r -> reiniciar calibracion")
    if calibrado:
        print("Calibracion previa cargada desde", CALIBRATION_PATH)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        bs_crudo = get_blendshape_dict(result)

        if calibrando and bs_crudo:
            muestras_calibracion.append(bs_crudo)
            if len(muestras_calibracion) >= CALIBRATION_FRAMES:
                nuevo_baseline = {}
                for k in BLENDSHAPES_CLAVE:
                    valores = [m.get(k, 0.0) for m in muestras_calibracion]
                    nuevo_baseline[k] = sum(valores) / len(valores)
                baseline = nuevo_baseline
                guardar_calibracion(baseline)
                calibrado = True
                calibrando = False
                muestras_calibracion = []
                bs_suavizado = None
                print("Calibracion guardada en", CALIBRATION_PATH)

        bs_calibrado = aplicar_calibracion(bs_crudo, baseline) if calibrado else bs_crudo
        bs_suavizado = suavizar(bs_calibrado, bs_suavizado)

        ahora = time.time()
        dt = ahora - tiempo_anterior
        tiempo_anterior = ahora
        if dt > 0:
            fps_suavizado = 0.9 * fps_suavizado + 0.1 * (1.0 / dt)

        frames_restantes = max(0, CALIBRATION_FRAMES - len(muestras_calibracion))
        metricas = calcular_metricas(bs_suavizado)
        expresion = clasificar_expresion(metricas)
        avatar = dibujar_avatar(bs_suavizado, expresion=expresion)
        dashboard = crear_dashboard(frame, avatar, metricas, fps_suavizado,
                                     calibrado, calibrando, frames_restantes)
        cv2.imshow("EmoAvatar - q:salir s:captura c:calibrar r:reset", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("captura_avatar.png", avatar)
            print("Captura guardada en captura_avatar.png")
        elif key == ord('c'):
            print("Calibrando... mantén el rostro neutral y quieto.")
            calibrando = True
            muestras_calibracion = []
        elif key == ord('r'):
            baseline = {}
            calibrado = False
            calibrando = False
            muestras_calibracion = []
            if os.path.exists(CALIBRATION_PATH):
                os.remove(CALIBRATION_PATH)
            print("Calibracion reiniciada.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
