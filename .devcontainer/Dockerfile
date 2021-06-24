FROM tensorflow/tensorflow:2.4.1-gpu
LABEL maintainer="benjamin.wilhelm@uni-konstanz.de"

ARG USERNAME=vsdocker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install essentials
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends git vim zsh tmux ssh less

# Install opencv2 dependency
RUN apt-get -y install --no-install-recommends libgl1-mesa-glx

# Runtime dependencies
RUN pip install tensorflow-datasets tensorflow-probability image-similarity-measures==0.3.5
RUN pip install git+https://github.com/HedgehogCode/tensorflow-datasets-bw.git@0.6.4

# Develop dependencies
RUN pip install flake8 autopep8 matplotlib jupyter
