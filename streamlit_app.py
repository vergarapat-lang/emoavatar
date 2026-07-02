"""
EmoAvatar - version web (Streamlit + streamlit-webrtc).

La camara se captura en el navegador de quien usa la app (via WebRTC) y
se procesa aca con MediaPipe. No usa cv2.imshow/VideoCapture porque el
servidor no tiene camara fisica: por eso esta version es un archivo
separado de avatar_pilot.py, aunque comparten toda la logica en
emoavatar_core.py.

Deploy en Streamlit Community Cloud: apuntar a este archivo como
"Main file path" al crear la app desde el repo de GitHub.
"""

import os
import time
import urllib.request

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from emoavatar_core import (
    BLENDSHAPES_CLAVE,
    CALIBRATION_FRAMES,
    aplicar_calibracion,
    calcular_metricas,
    clasificar_expresion,
    crear_dashboard,
    dibujar_avatar,
    get_blendshape_dict,
    suavizar,
)

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

st.set_page_config(page_title="EmoAvatar", page_icon="🙂", layout="wide")


@st.cache_resource
def descargar_modelo():
    """Descarga el modelo de MediaPipe una sola vez por instancia del servidor."""
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class EmoAvatarProcessor(VideoProcessorBase):
    """Mantiene el estado (calibracion, suavizado, FPS) entre frames de un
    mismo usuario conectado. streamlit-webrtc crea una instancia de esta
    clase por sesion y llama a recv() por cada frame de video."""

    def __init__(self):
        model_path = descargar_modelo()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.baseline = {}
        self.calibrado = False
        self.calibrando = False
        self.muestras_calibracion = []

        self.bs_suavizado = None
        self.fps_suavizado = 30.0
        self.tiempo_anterior = time.time()
        self.tiempo_inicio = time.time()

    def iniciar_calibracion(self):
        self.calibrando = True
        self.muestras_calibracion = []

    def reiniciar_calibracion(self):
        self.baseline = {}
        self.calibrado = False
        self.calibrando = False
        self.muestras_calibracion = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.tiempo_inicio) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        bs_crudo = get_blendshape_dict(result)

        if self.calibrando and bs_crudo:
            self.muestras_calibracion.append(bs_crudo)
            if len(self.muestras_calibracion) >= CALIBRATION_FRAMES:
                nuevo_baseline = {}
                for k in BLENDSHAPES_CLAVE:
                    valores = [m.get(k, 0.0) for m in self.muestras_calibracion]
                    nuevo_baseline[k] = sum(valores) / len(valores)
                self.baseline = nuevo_baseline
                self.calibrado = True
                self.calibrando = False
                self.muestras_calibracion = []
                self.bs_suavizado = None

        bs_calibrado = aplicar_calibracion(bs_crudo, self.baseline) if self.calibrado else bs_crudo
        self.bs_suavizado = suavizar(bs_calibrado, self.bs_suavizado)

        ahora = time.time()
        dt = ahora - self.tiempo_anterior
        self.tiempo_anterior = ahora
        if dt > 0:
            self.fps_suavizado = 0.9 * self.fps_suavizado + 0.1 * (1.0 / dt)

        frames_restantes = max(0, CALIBRATION_FRAMES - len(self.muestras_calibracion))
        metricas = calcular_metricas(self.bs_suavizado)
        expresion = clasificar_expresion(metricas)
        avatar = dibujar_avatar(self.bs_suavizado, expresion=expresion)
        dashboard = crear_dashboard(img, avatar, metricas, self.fps_suavizado,
                                     self.calibrado, self.calibrando, frames_restantes)

        return av.VideoFrame.from_ndarray(dashboard, format="bgr24")


def main():
    st.title("🙂 EmoAvatar")
    st.caption("Reconocimiento e imitación de emociones — piloto")

    st.info(
        "La cámara se procesa en tu navegador y no se guarda en ningún servidor. "
        "Al iniciar, tu navegador pedirá permiso de cámara."
    )

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="emoavatar",
        video_processor_factory=EmoAvatarProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Calibrar (mantén cara neutral)"):
            if ctx.video_processor:
                ctx.video_processor.iniciar_calibracion()
            else:
                st.warning("Primero inicia la cámara arriba.")
    with col2:
        if st.button("↺ Reiniciar calibración"):
            if ctx.video_processor:
                ctx.video_processor.reiniciar_calibracion()
    with col3:
        st.write("")

    with st.expander("Acerca de EmoAvatar"):
        st.markdown(
            "Detecta 6 emociones (Alegría, Tristeza, Enojo, Miedo, Sorpresa, "
            "Neutral) a partir de los movimientos faciales, pensado como "
            "herramienta de apoyo para el reconocimiento e imitación de "
            "emociones en niños y jóvenes con TEA. Es un piloto: la "
            "clasificación usa reglas heurísticas sobre blendshapes de "
            "MediaPipe, no un modelo clínicamente validado."
        )


if __name__ == "__main__":
    main()
