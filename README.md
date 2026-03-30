# RMBG-2-Studio

Aplicacion local para eliminar fondos, componer imagenes y procesar lotes usando BRIA RMBG 2.0 desde una interfaz Gradio.

Ya no depende de Pinokio. Se ejecuta directamente con Python.

## Ejecutarlo

Desde la raiz del proyecto:

```bash
chmod +x setup.sh run.sh
./run.sh
```

El primer arranque:

- crea `.venv/`
- instala PyTorch segun tu plataforma
- instala las dependencias de `app/requirements.txt`
- descarga el modelo la primera vez que abras la app

Cuando arranque, abre en el navegador:

```text
http://127.0.0.1:7860
```

Las imagenes generadas se guardan en `output_images/`.

## Arranque manual

Si prefieres hacerlo a mano:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
python -m pip install -r app/requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python app/app.py
```

Ese bloque es el adecuado para macOS Intel. Si cambias de plataforma, `setup.sh` ya resuelve la version de PyTorch automaticamente.

## Variables opcionales

- `RMBG_HOST`: host de Gradio. Por defecto `127.0.0.1`
- `RMBG_PORT`: puerto. Por defecto `7860`
- `RMBG_MODEL_ID`: id o ruta local del modelo. Por defecto `cocktailpeanut/rm`

Ejemplo:

```bash
RMBG_PORT=7861 ./run.sh
```

## Notas

- La primera ejecucion necesita internet para descargar dependencias y el modelo, salvo que apuntes `RMBG_MODEL_ID` a una copia local ya descargada.
- El proyecto sigue dependiendo de librerias de terceros como `torch`, `transformers` y `gradio`; lo que se elimino fue la capa de Pinokio.
