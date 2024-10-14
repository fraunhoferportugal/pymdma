FROM python:3.9.19-slim

# Maintainer info
LABEL authors.auth1="ivo.facoco@fraunhofer.pt"\
    authors.auth2="telmo.baptista@fraunhofer.pt"

WORKDIR /app
COPY requirements/ requirements/

RUN apt-get update && apt-get upgrade -y && \
apt-get install -y --no-install-recommends git ffmpeg libsm6 libxext6 && \
apt-get clean && rm -rf /var/lib/apt/lists/* && \
pip install --upgrade pip setuptools --no-cache-dir && \
pip install -r requirements/requirements-prod.txt --no-cache-dir \
--find-links="https://download.pytorch.org/whl/cpu/torch_stable.html"

# copy code to /app
COPY .env .env
COPY src src

ENV PYTHONPATH="/app/src"
ENV TOKENIZERS_PARALLELISM="false"

# The code to run when container is started
ENTRYPOINT [ "python", "src/pymdma/api/run_api.py"]
