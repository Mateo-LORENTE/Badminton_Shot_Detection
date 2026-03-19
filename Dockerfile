# syntax=docker/dockerfile:1
#

#version qui marche
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# Installation de Python3 et des dépendances
RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0


#Installation de Pytorch pour mmpose
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install tensorflow==2.12.*



#Installation de mmlab :
RUN pip3 install -U openmim
RUN mim install mmengine
RUN mim install mmcv

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de code source dans le conteneur
COPY . /app

VOLUME ["/app/scripts"]

# Déclare le point de montage pour le répertoire d'entrée
VOLUME ["/app/docker_inputs"]

# Déclare le point de montage pour le répertoire de sortie
VOLUME ["/app/docker_predictions"]

RUN pip3 install numpy==1.22.4

# Installation des dépendances Python
RUN pip3 install -r requirements.txt

RUN pip3 install numpy --upgrade


CMD python3 /1_FFBAD/Inférence/complete_tracking_TrackNetV2.py  --inputs_path "2_vidIN/videoplay.mp4" --outputs_path $OUTPUT_PATH


