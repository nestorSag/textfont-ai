FROM tensorflow/tensorflow:devel-gpu
COPY ./src/fontai /root/fontai 
COPY ./scripts /root/scripts
RUN pip install /root/fontai

ENTRYPOINT python /root/scripts/base/run-stage.py 

