## NOTE: There are two Dockerfiles, this is for Google Run
## Another under .devcontainer is for development

# Miniconda3
FROM continuumio/miniconda3:4.12.0

WORKDIR /app
COPY .devcontainer/environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml && conda clean --all -y

RUN apt-get update && apt-get install -y libgl1-mesa-glx


SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate machine_learning-docker-3.8" >> ~/.bashrc

COPY . /app/
ENV PATH="/opt/conda/envs/machine_learning-docker-3.8/bin:$PATH"

# Cloud Run needs FastAPI directly running
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
