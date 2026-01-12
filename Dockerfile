ARG PYTHON_VERSION="3.12"

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=2.2.1

WORKDIR /app

COPY . .

RUN python -m ensurepip --upgrade \
    && apt-get update && apt-get dist-upgrade -y \
    && apt-get install -y git curl libleveldb-dev ffmpeg lsof \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN curl -sSL https://install.python-poetry.org | python3 -

# Install NVM
ENV NODE_VERSION=25.2.1
ENV NVM_DIR=/root/.nvm

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
 
# Load NVM and install Node.js
RUN bash -c "source $NVM_DIR/nvm.sh && \
             nvm install $NODE_VERSION && \
             nvm alias default $NODE_VERSION && \
             nvm use default"

ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm install acorn acorn-walk \
    && pip install javascript

RUN poetry install --no-root

ENTRYPOINT [ "poetry", "run", "python", "fansly_downloader_ng.py"]
