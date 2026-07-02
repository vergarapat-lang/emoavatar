# EmoAvatar — Reconocimiento e imitación de emociones (piloto)

Detecta movimientos faciales por webcam y los refleja en un avatar 2D,
clasificando en tiempo real 6 emociones pensadas para trabajo con niños y
jóvenes con TEA: **Alegría, Tristeza, Enojo, Miedo, Sorpresa y Neutral/calma**.

Este proyecto tiene **dos versiones que comparten la misma lógica**
(`emoavatar_core.py`):

| Archivo             | Dónde corre                         | Cómo ve la cámara                     |
|----------------------|--------------------------------------|----------------------------------------|
| `avatar_pilot.py`    | Tu computador (escritorio)          | `cv2.VideoCapture` directo             |
| `streamlit_app.py`   | Servidor web (Streamlit Cloud)      | `streamlit-webrtc` (cámara del navegador de quien usa la app) |

## Estructura del proyecto

```
emoavatar/
  emoavatar_core.py       # logica compartida: blendshapes, calibracion,
                           # dibujo del avatar, dashboard (sin camara)
  avatar_pilot.py          # version de escritorio (cv2.imshow)
  streamlit_app.py         # version web (streamlit-webrtc)
  requirements.txt         # dependencias para Streamlit Cloud
  requirements-desktop.txt # dependencias para uso local
  README.md
  .gitignore
```

---

## 1. Uso local (escritorio)

```bash
python3 -m venv venv
source venv/bin/activate   # en Mac/Linux
pip install -r requirements-desktop.txt
```

Descarga el modelo de landmarks faciales (una sola vez, ~4 MB):

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Ejecuta:

```bash
python avatar_pilot.py
```

Controles: `q` salir · `s` guardar captura · `c` calibrar · `r` reiniciar calibración.

---

## 2. Subir el proyecto a GitHub

Desde la carpeta `emoavatar/` en tu Mac:

```bash
git init
git add .
git commit -m "EmoAvatar: piloto de reconocimiento de emociones"
```

Crea el repositorio vacío en GitHub (en el navegador):
1. Ve a https://github.com/new
2. Nombre sugerido: `emoavatar`
3. Déjalo **público** si quieres desplegarlo gratis en Streamlit Cloud
4. NO marques "Add a README" (ya tienes uno) — déjalo vacío
5. Clic en "Create repository"

GitHub te mostrará comandos; usa estos (reemplaza si tu rama por defecto es distinta):

```bash
git branch -M main
git remote add origin https://github.com/vergarapat-lang/emoavatar.git
git push -u origin main
```

Si te pide usuario/contraseña y falla, probablemente necesites un
**Personal Access Token** en vez de tu contraseña normal de GitHub
(GitHub → Settings → Developer settings → Personal access tokens).

Nota: el `.gitignore` ya excluye `face_landmarker.task` (se descarga
automáticamente) y `calibracion.json` (datos personales), así que no se
subirán al repo.

---

## 3. Desplegar en Streamlit Community Cloud

1. Ve a https://share.streamlit.io
2. Inicia sesión con tu cuenta de GitHub
3. "New app" → selecciona el repo `vergarapat-lang/emoavatar`
4. **Main file path**: `streamlit_app.py`
5. Deploy

La primera vez que alguien abra la app, `streamlit_app.py` descarga el
modelo `.task` automáticamente (función `descargar_modelo()`, cacheada con
`st.cache_resource` para no re-descargarlo en cada sesión).

**Importante sobre la cámara en la nube**: cada persona que use la app
verá un botón "START" del componente `streamlit-webrtc`; el navegador le
pedirá permiso de cámara. La calibración se hace con los botones
"🎯 Calibrar" y "↺ Reiniciar calibración" debajo del video (no con teclas,
porque el teclado no llega al servidor).

### Limitaciones conocidas de la versión web
- El primer frame puede tardar unos segundos (carga del modelo + conexión WebRTC).
- La calibración es por sesión de navegador; no se guarda entre visitas
  (a diferencia de la versión de escritorio, que guarda `calibracion.json`
  en disco).
- Streamlit Community Cloud gratuito duerme la app tras un tiempo sin uso;
  el primer acceso del día puede tardar en despertar.

---

## Cómo funciona (resumen técnico)

1. **MediaPipe Face Landmarker** convierte cada frame de cámara en 52
   "blendshapes" (valores 0.0–1.0: parpadeo, apertura de boca, sonrisa,
   cejas, etc.).
2. `calcular_metricas()` combina varios blendshapes en un score 0–1 para
   cada una de las 6 emociones objetivo (heurística por reglas, no un
   modelo entrenado).
3. `clasificar_expresion()` elige la emoción con mayor score si supera un
   umbral; si no, devuelve "Neutral".
4. `dibujar_avatar()` anima un rostro caricaturesco según esos mismos
   blendshapes (cejas con punto interior/exterior independientes para que
   Tristeza se vea distinta de Miedo/Sorpresa, boca con curva
   parabólica que soporta sonrisa y frown).
5. `crear_dashboard()` compone el panel final: cámara, barras de las 6
   emociones, estado de conexión, y el avatar.

## Próximos pasos posibles
- Reemplazar las formas geométricas del avatar por imágenes ilustradas
  (sprites PNG generados con IA) para un acabado más realista.
- Guardar la calibración de la versión web en `st.session_state` con
  persistencia opcional (ej. localStorage vía componente custom).
- Modo "juego de imitación": mostrar una emoción objetivo y medir cuánto
  tarda la persona en producirla.
- Validar la heurística de clasificación con un terapeuta/muestra real de
  niños, y ajustar umbrales por edad o perfil individual.
