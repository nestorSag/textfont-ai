# build from root folder with docker build -t <user>/textfont-ai-runner:1.0.0 -f docker-images/training/Dockerfile .
# FROM continuumio/anaconda3
# WORKDIR /textfont-ai
# COPY env/conda-env.yaml ./conda-env.yaml 
# ADD scripts ./scripts
# ADD src/fontai ./fontai 
# #install gcc in order to compile beam[gcp] binaries
# RUN apt update && apt install build-essential -y
# # setup conda env
# RUN conda update -n base -c defaults conda &&\
#   conda env create -f ./conda-env.yaml &&\
#   conda run -n textfont-ai pip install ./fontai &&\
#   conda clean --all -y

# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV CONTAINER_ENV=true
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "textfont-ai", "python", "./scripts/base/run-stage.py"]


FROM tensorflow/tensorflow:latest-gpu
WORKDIR /textfont-ai
ADD scripts ./scripts
ADD src/fontai ./fontai 
RUN pip install ./fontai
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CONTAINER_ENV=true
ENTRYPOINT ["python", "./scripts/base/run-stage.py"]
