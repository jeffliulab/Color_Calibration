# Select the official Miniconda3 image (Python 3.8)
FROM continuumio/miniconda3:4.12.0

# Set the working directory
WORKDIR /workspace

# Copy environment.yml to the container
COPY environment.yml /workspace/environment.yml


# Create a Conda environment (make sure the Python version is consistent with `environment.yml`)
RUN conda env create -f /workspace/environment.yml

# **Activate Conda environment and set default environment**
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate machine_learning-docker-3.8" >> ~/.bashrc

#Copy the project code to the container (avoid the influence of `.git`)
COPY .. /workspace/

# Enter the Bash terminal and make sure the Conda environment is automatically activated
CMD ["/bin/bash", "-l"]
