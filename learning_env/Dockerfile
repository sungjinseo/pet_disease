# local 빌드시
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS nvidia

# CUDA
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=8
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH_NO_STUBS="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/conda/lib"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/opt/conda/lib"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 카카오 ubuntu archive mirror server 추가. 다운로드 속도 향상
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list && \
    apt-get update && apt-get install alien -y

# openjdk java vm 설치
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    g++ \
    gcc \
    openjdk-8-jdk \
    python3-dev \
    python3-pip \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    libzmq3-dev \
    vim \
    git &&\
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update

ARG CONDA_DIR=/opt/conda

# add to path
ENV PATH $CONDA_DIR/bin:$PATH

# Install miniconda
RUN echo "export PATH=$CONDA_DIR/bin:"'$PATH' > /etc/profile.d/conda.sh && \
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Conda 가상환경 생성
RUN conda config --set always_yes yes --set changeps1 no && \
    conda create -y -q -n py39 python=3.9

ENV PATH /opt/conda/envs/py39/bin:$PATH
ENV CONDA_DEFAULT_ENV py39
ENV CONDA_PREFIX /opt/conda/envs/py39

# 패키지 설치 177s
# 필요없는건 삭제 필요함
RUN pip install setuptools && \
    pip install mkl && \
    pip install numpy && \
    pip install scipy && \
    pip install pandas && \
    pip install jupyterlab && \
    pip install jupyterthemes && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install hyperopt && \
    pip install optuna && \
    pip install missingno && \
    pip install mlxtend && \
    pip install catboost && \
    pip install folium && \
    pip install librosa && \
    pip install nbconvert && \
    pip install Pillow && \
    pip install tqdm && \
    pip install statsmodels && \
    apt-get install -y graphviz && pip install graphviz && \
    pip install cupy-cuda11x

RUN pip install --upgrade cython && \
    pip install --upgrade cysignals && \
    pip install transformers

# 114s
RUN pip install pystan==2.19.1.1 && \
    pip install prophet && \
    pip install torchsummary

# vision 342s
RUN pip install wandb tensorboard albumentations pydicom opencv-python scikit-image pyarrow kornia \
    catalyst captum

RUN pip install fastai && \
    pip install fvcore 

ENV PATH=/usr/local/bin:${PATH}
ENV PATH=/usr/local/bin:${PATH}

# 나눔고딕 폰트 설치
# matplotlib에 Nanum 폰트 추가
RUN apt-get install fonts-nanum* && \
    cp /usr/share/fonts/truetype/nanum/Nanum* /opt/conda/envs/py39/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/ && \
    fc-cache -fv && \
    rm -rf ~/.cache/matplotlib/*

# XGBoost (GPU 설치)
RUN pip install xgboost

# Install OpenCL & libboost (required by LightGBM GPU version)
RUN apt-get install -y ocl-icd-libopencl1 clinfo libboost-all-dev && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
	
RUN pip uninstall -y lightgbm && \
    cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
    git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j$(nproc) OPENCL_HEADERS=/usr/local/cuda-11.2/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda-11.2/targets/x86_64-linux/lib && \
    cd /usr/local/src/lightgbm/LightGBM/python-package && python setup.py install --precompile

# PyTorch 2.0 설치
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 &&\
    pip install torchtext==0.15.1

# TensorFlow 2.12.0rc1 설치
# Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>. 오류발생
# RUN pip install tensorflow==2.12.0rc1
RUN pip install tensorflow==2.9.1

# elyra 설치
RUN pip install nodejs && \
    pip install --upgrade elyra[all]

# Remove the CUDA stubs.
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NO_STUBS"

# locale 설정
RUN apt-get update && apt-get install -y locales tzdata && \
    locale-gen ko_KR.UTF-8 && locale -a && \
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -a -y

# LANG 환경변수 설정
ENV LANG ko_KR.UTF-8

# Jupyter Notebook config 파일 생성
RUN jupyter lab --generate-config

# jupyter server setting
RUN echo "c.NotebookApp.password = 'sha1:b00ee0c0e13c:9b3f0c11356d567b4b9c516af4aeae5df5a2f1e4'" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.NotebookApp.password_require=True" >> /root/.jupyter/jupyter_notebook_config.py110a0e6ab9fe &&\
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.NotebookApp.port = 8000" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.LabBuildApp.minimize = False" >> /root/.jupyter/jupyter_notebook_config.py &&\
    echo "c.LabBuildApp.dev_build = False" >> /root/.jupyter/jupyter_notebook_config.py

# 설치 완료 후 테스트용 ipynb
COPY ./01-GPU-TEST/GPU-Test.ipynb /home/jupyter/GPU-Test.ipynb

CMD jupyter lab --allow-root