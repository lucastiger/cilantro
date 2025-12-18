FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential git python3 python3-pip wget curl ffmpeg
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
WORKDIR /workspace
COPY .. /workspace
ENV PYTHONPATH=/workspace
CMD ["bash"]
