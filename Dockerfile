# NVIDIAのCUDA 12.0.0のdevel版（Ubuntu 20.04ベース）を利用
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# 環境変数の設定（非対話モード＆タイムゾーンの自動設定）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    libsndfile1 \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# deadsnakes PPAを追加して、Python 3.9と関連パッケージをインストール
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# pip のインストール
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# python コマンドでpython3.9が呼ばれるようにシンボリックリンクを設定（必要に応じて）
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# 作業ディレクトリの設定（適宜変更してください）
WORKDIR /workspace

# pip のアップグレードと必要な Python パッケージのインストール
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu120 && \
    pip install transformers soundfile xcodec2==0.1.3

# コンテナ起動時のデフォルトコマンド（適宜変更してください）
CMD ["python"]
