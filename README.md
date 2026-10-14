# posters-generation

Generación de carteles de cine a partir de sinopsis. El objetivo es entrenar un LoRA de Stable Diffusion 1.5 con carteles reales y, dada una sinopsis nueva, generar un cartel acorde.

El proyecto está en fase **MVP** y trabaja con un subconjunto: películas de la década de 2010 (2010-2019) y 6 géneros (terror, comedia, ciencia ficción, romance, animación y acción).

## Fases del MVP

| Fase | Contenido | Estado |
|---|---|---|
| 1. Dataset | Descarga de carteles y metadatos desde TMDB, estandarización y particiones | Hecha (`MVP/dataset_creation.ipynb`) |
| 2. Prompts | Conversión de sinopsis en prompts visuales para Stable Diffusion con un LLM local | En desarrollo (`MVP/prompt_generator.ipynb`) |
| 3. Entrenamiento | LoRA sobre Stable Diffusion 1.5 | Pendiente |
| 4. Evaluación | Clasificador de género (ViT/CNN) sobre los carteles generados | Pendiente |

## Estructura

```
MVP/
├── dataset_creation.ipynb   # Fase 1: crea data_v2/
├── prompt_generator.ipynb   # Fase 2: sinopsis -> prompt de difusión
├── .env.example             # plantilla de la clave de TMDB
└── data_v2/
    ├── metadata.csv         # una fila por película
    ├── posters/{id}.jpg     # carteles 512x768
    └── raw/{id}.json        # respuesta cruda de TMDB
```

## Requisitos

- Python 3.10 o superior.
- Una clave de la API de TMDB (v3).
- [Ollama](https://ollama.com) en local, solo para la fase 2.

```bash
pip install requests pandas pillow python-dotenv matplotlib openai pydantic tqdm
```

## Fase 1: creación del dataset

Notebook: [MVP/dataset_creation.ipynb](MVP/dataset_creation.ipynb)

1. Crea `MVP/.env` a partir de la plantilla y pon tu clave. El archivo está en `.gitignore`, **no subas la clave a git**:
   ```bash
   cd MVP
   cp .env.example .env
   # edita .env:  TMDB_API_KEY=tu_clave
   ```
2. Ejecuta el notebook. Es reanudable: reutiliza los `raw/*.json` y los carteles ya descargados.

### Qué hace

- **Selección:** películas de 2010-2019, de un único género objetivo, con idioma original inglés, al menos 10 votos y cartel disponible. Toma hasta 340 por género de un conjunto ordenado por número de votos y mezclado con semilla fija (`SEED = 42`).
- **Filtro de calidad:** descarta sinopsis de menos de 20 palabras, para que el LLM tenga contexto narrativo suficiente.
- **Imágenes:** descarga el cartel en `w780`, recorta al centro a proporción 2:3 y reduce con Lanczos a **512×768**, de modo que nunca se amplía ni se deforma.
- **Particiones deterministas:** hash MD5 de la **colección** de la película (o de su id si no tiene), de modo que las secuelas caen siempre en la misma partición y no hay fuga de datos.

| Partición | Proporción | Uso previsto |
|---|---|---|
| `train` | 70 % | entrenamiento |
| `val` | 10 % | validación |
| `test` | 10 % | prueba |
| `holdout_gen` | 10 % | películas que el LoRA no verá, para evaluar la generación |

### Resultado

1.992 películas: 340 por género salvo animación (292). Particiones: `train` 1.351, `val` 208, `test` 200 y `holdout_gen` 233. Todos los carteles miden 512×768.

Columnas de `metadata.csv`: `tmdb_id`, `title`, `year`, `genre`, `overview`, `tagline`, `keywords`, `collection`, `poster_file`, `split`.

## Fase 2: generación de prompts

Notebook: [MVP/prompt_generator.ipynb](MVP/prompt_generator.ipynb)

Los carteles de TMDB no vienen con una descripción visual, solo con la sinopsis. Este notebook convierte la sinopsis en un prompt para Stable Diffusion 1.5 usando un LLM local (`llama3:instruct` a través de Ollama, sin coste de API).

### Preparación

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:instruct
curl localhost:11434        # debe responder "Ollama is running"
```

Para lanzar varias peticiones en paralelo, configura `OLLAMA_NUM_PARALLEL` en el servicio de Ollama. Debe ser al menos `WORKERS × N_CANDIDATES` (por defecto 2 × 2). En una GPU de 4 GB conviene bajar `WORKERS` a 1 si hay errores de memoria.

### Cómo funciona

El trabajo se reparte entre el código y el LLM:

| Lo decide el código | Lo escribe el LLM |
|---|---|
| Encuadre del cartel (primer plano, plano general, silueta, objeto simbólico, dúo...), con pesos por género | Sujeto, vestuario y objetos, pose, escenario y clima, en 2-7 palabras cada uno |
| Iluminación, paleta y estilo según el género (3 variantes por género) | |

Pipeline por película:

1. Se elige un encuadre y un estilo con una semilla fija por película, así el resultado es reproducible.
2. El LLM genera `N_CANDIDATES` borradores en paralelo, en JSON validado con Pydantic.
3. Se descartan los borradores con campos vacíos, más de 60 palabras, palabras prohibidas (`poster`, `text`, `logo`, `watermark`...) o el título de la película. Se reintenta hasta `GEN_TRIES` veces.
4. Si quedan dos borradores válidos, un LLM-juez elige el más fiel a la sinopsis y más visual. Con `JUDGE_PASSES = 2` compara dos veces con el orden invertido.
5. El prompt final es `etiqueta de encuadre, contenido del LLM, estilo del género` y se devuelve en el campo `diffusion_prompt`, junto con los dos borradores, el ganador, el acuerdo del juez y los tiempos.

### Uso

- `run_test()` procesa 6 películas de ejemplo (una por género) y muestra los prompts, los tiempos y la distribución de encuadres y ganadores.
- `run_dataset()` procesa `data_v2/metadata.csv` y escribe `data_v2/metadata_prompts_local.csv`. Es reanudable y está **comentada** en el notebook: quita las comillas triples para usarla. Por defecto procesa solo `holdout_gen`; con `split=None` procesa todas las particiones.

Parámetros ajustables (modelo, candidatos, temperatura, juez, paralelismo, pesos de encuadre y estilos por género) en la celda de configuración del notebook.

## Limitaciones conocidas y siguientes pasos

- **Los prompts salen de la sinopsis, no de la imagen.** Un cartel puede no mostrar lo que cuenta su sinopsis. Hay que validar si estos prompts sirven como captions de entrenamiento del LoRA o si hace falta describir los carteles con un modelo de visión.
- **El encuadre y el estilo se asignan por código**, no se extraen del cartel real.
- **Los carteles reales contienen texto** (títulos y créditos), mientras que los prompts lo excluyen. Hay que decidir cómo tratarlo antes de entrenar.
- Los metadatos `keywords` y `tagline` se descargan pero el generador de prompts aún no los usa.
- Pendiente: entrenamiento del LoRA, inferencia sobre `holdout_gen` y evaluación con el clasificador de género.

## Créditos

Este producto usa la API de TMDB pero no está avalado ni certificado por TMDB. Licencia: ver [LICENSE](LICENSE).
