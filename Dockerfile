
FROM continuumio/anaconda3
WORKDIR /root/textfont-ai
COPY env/conda-env.yaml ./conda-env.yaml 
ADD src/fontai ./src/fontai 
#install gcc in order to compile beam[gcp] binaries
RUN apt update && apt install build-essential -y
# setup conda env
RUN conda update -n base -c defaults conda &&\
  conda env create -f ./conda-env.yaml &&\
  conda run -n textfont-ai pip install ./src/fontai &&\
  conda clean --all -y

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CONTAINER_ENV=true
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "textfont-ai", "fontairun"]


# FOR GOOGLE AI PLATFORM COMMENT THE ABOVE AND USE THE FOLLOWING INSTEAD

# FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-2
# WORKDIR /textfont-ai
# ADD ./src/fontai ./src/fontai 
# RUN pip install ./src/fontai
# RUN pip install typing-extensions==3.7.4.3
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV CONTAINER_ENV=true
# ENTRYPOINT ["fontairun"]
