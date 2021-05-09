FROM ovhcom/ai-training-transformers

RUN apt-get update && \
    apt install -y bash \
    build-essential \
    libsndfile1-dev \
    git-lfs \
    sox

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade \
    wandb \
    transformers \
    datasets \
    unidecode \
    torch>=1.5.0 \
    torchaudio \
    jiwer==2.2.0 \
    lang-trans==0.6.0 \
    librosa==0.8.0 \
    soundfile

COPY train.py /workspace/

COPY docker/home-server.html /usr/bin/

RUN chown -R 42420:42420 /workspace

RUN alias python=python3

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /workspace
ENTRYPOINT []
CMD ["supervisord", "-n", "-u", "42420", "-c", "/etc/supervisor/supervisor.conf"]