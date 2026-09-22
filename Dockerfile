# CPU-only, offline tests. Rendering and the MiniLM download are skipped
# inside the container (no GL context, no network assumed); the smoke test
# that records demonstrations skips with the render tests.
#   docker build -t so-arm100-vla . && docker run --rm so-arm100-vla
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE requirements.txt ./
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir -r requirements.txt
COPY so_arm100_vla ./so_arm100_vla
COPY tests ./tests
COPY train.py eval_language.py record_data.py smolvla_reference.py lora_gate.py verify.sh ./

# pyproject already passes -q; a second one here would suppress the summary line,
# and the summary is the only thing the image is run for.
CMD ["python", "-m", "pytest", "tests", "--deselect", "tests/test_text.py::test_minilm_encoder_and_cache"]
