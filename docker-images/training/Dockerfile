# FROM tensorflow/tensorflow:devel-gpu
# ADD src/fontai /root/fontai 
# ADD scripts /root/scripts
# RUN pip install /root/fontai
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENTRYPOINT python /root/scripts/base/run-stage.py 

FROM continuumio/anaconda3
COPY env/conda-env.yaml /root/conda-env.yaml 
ADD scripts /root/scripts
ADD src/fontai /root/fontai 
#install gcc in order to compile beam[gcp] binaries
RUN apt update && apt install build-essential -y
RUN conda env create -f /root/conda-env.yaml && conda activate textfont-ai && pip install /root/fontai 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true
ENTRYPOINT python /root/scripts/base/run-stage.py 
