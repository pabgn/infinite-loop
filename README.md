# INFINITE LOOP — Eternal Music Engine

Analiza una pista de YouTube, encuentra beats musicalmente similares y la
reproduce **para siempre**, con saltos coherentes y progresión de energía.

El motor de análisis (v2) hace:

1. **Chroma invariante a tonalidad** — normalización tonal vía detección de key.
2. **Ventanas de contexto** — la similitud compara frases, no beats aislados.
3. **Detección de downbeats** — los saltos solo ocurren al inicio de compás.
4. **Segmentación por novedad de Foote** — fronteras estructurales reales.
5. **Penalización anti-recencia** — evita revisitar las mismas zonas.
6. **Curva de arco de energía** — progresión siguiendo la envolvente del tema.

## Estructura del proyecto

```
infinite-loop/
├── app.py                  # App Flask + rutas (punto de entrada)
├── requirements.txt        # Dependencias Python
├── pyproject.toml          # Metadatos del proyecto / instalación
├── infinite_loop/          # Paquete principal
│   ├── __init__.py
│   ├── config.py           # CACHE_DIR, host/puerto, estado compartido
│   ├── analysis.py         # Motor de análisis (DSP, jump graph, render)
│   └── pipeline.py         # Descarga + análisis en segundo plano
├── templates/
│   └── index.html          # Frontend (HTML)
├── static/
│   ├── css/style.css       # Estilos
│   └── js/app.js           # Lógica del player en el navegador
└── patterns/               # (reservado)
```

## Requisitos

- Python 3.9+
- **ffmpeg** instalado en el sistema (lo usan `yt-dlp` y `pydub`):
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt install ffmpeg`

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

Abre <http://localhost:8149> en tu navegador, pega una URL de YouTube y dale a
analizar.

### Modo desarrollo (auto-reload)

```bash
flask --app app run --debug
```

## Configuración

Variables de entorno opcionales:

| Variable                 | Por defecto                       | Descripción                  |
|--------------------------|-----------------------------------|------------------------------|
| `INFINITE_LOOP_HOST`     | `0.0.0.0`                         | Host del servidor            |
| `INFINITE_LOOP_PORT`     | `8149`                           | Puerto                       |
| `INFINITE_LOOP_DEBUG`    | `false`                          | Modo debug de Flask          |
| `INFINITE_LOOP_CACHE`    | `<tmp>/infinite_loop_cache`      | Carpeta de caché (audio+JSON)|

## API

| Método | Ruta                    | Descripción                          |
|--------|-------------------------|--------------------------------------|
| `POST` | `/api/analyze`          | Inicia análisis (`{"url": "..."}`)   |
| `GET`  | `/api/status/<hash>`    | Progreso del análisis                |
| `GET`  | `/api/graph/<hash>`     | Grafo de saltos (JSON)               |
| `GET`  | `/api/audio/<hash>`     | Audio descargado (MP3)               |
