FROM conda/miniconda3
COPY ./src /root/src
COPY ./scripts /root/scripts
COPY ./conda-env.yaml /root/conda-end.yaml
RUN conda create --file /root/conda-env.yaml && conda activate textfont-ai && pip install -e /root/src
ENTRYPOINT python /root/scripts/base/run-stage.py 
