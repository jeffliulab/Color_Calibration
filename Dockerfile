# 选择官方 Miniconda 镜像（Python 3.8）
FROM continuumio/miniconda3:4.12.0

# 设置工作目录
WORKDIR /workspace

# 复制 environment.yml 到容器
COPY environment.yml /workspace/environment.yml

# 创建 Conda 环境（确保 Python 版本与 `environment.yml` 一致）
RUN conda env create -f /workspace/environment.yml

# 激活 Conda 环境并设置默认启动环境
RUN echo "conda activate machine_learning-docker-3.8" >> ~/.bashrc

# 复制项目代码到容器（避免 `.git` 影响）
COPY . /workspace/

# 进入 Bash 终端
CMD ["/bin/bash"]
