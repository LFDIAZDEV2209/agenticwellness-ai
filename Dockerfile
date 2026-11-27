FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq5 \
       libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*

# Solo la imagen de piloto necesita codecs y síntesis local.
ARG INSTALL_PRIVATE_TAP=false
RUN if [ "$INSTALL_PRIVATE_TAP" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
      && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements-prod.txt requirements-prod-constraints.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Vocabulario público, sin credenciales. El runtime lo carga sin descargarlo.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
