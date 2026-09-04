FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME="/home/user/.cache/huggingface"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# Hosts hand us a port via $PORT; fall back to 7860 when run locally.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
