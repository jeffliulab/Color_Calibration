# Select the official Miniconda3 image (Python 3.8)
FROM continuumio/miniconda3:4.12.0

# Set the working directory
WORKDIR /workspace

# Copy environment.yml to the container
COPY environment.yml /workspace/environment.yml

# **Install system dependencies 
RUN apt-get update
RUN if command -v lspci >/dev/null 2>&1 && lspci | grep -i nvidia; then \
        apt-get install -y --no-install-recommends nvidia-cuda-toolkit; \
    else \
        echo "Skipping CUDA installation (no NVIDIA detected)"; \
    fi
COPY installed_packages.txt /tmp/installed_packages.txt
RUN grep -vE '^#|^$' /tmp/installed_packages.txt | awk '{print $1}' | \
    xargs -I{} bash -c 'apt-cache show {} >/dev/null 2>&1 && apt-get install -y --no-install-recommends {} || echo "Skipping {}"' && \
    rm -rf /var/lib/apt/lists/*




# Create a Conda environment (make sure the Python version is consistent with `environment.yml`)
RUN conda env create -f /workspace/environment.yml

# **Activate Conda environment and set default environment**
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate machine_learning-docker-3.8" >> ~/.bashrc

#Copy the project code to the container (avoid the influence of `.git`)
COPY .. /workspace/

# Enter the Bash terminal and make sure the Conda environment is automatically activated
CMD ["/bin/bash", "-l"]
