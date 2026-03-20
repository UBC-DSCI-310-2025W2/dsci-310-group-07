# set base image  
FROM condaforge/miniforge3:25.9.1-0

# copy lock file
COPY conda-lock.yml /tmp/conda-lock.yml

# install conda-lock, create environment, and install Quarto manually
RUN conda install -c conda-forge conda-lock wget -y && \
    conda-lock install -n project_env /tmp/conda-lock.yml && \
    wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.36/quarto-1.9.36-linux-arm64.tar.gz -O /tmp/quarto.tar.gz && \
    mkdir -p /opt/quarto && \
    tar -xzf /tmp/quarto.tar.gz -C /opt/quarto --strip-components=1 && \
    ln -s /opt/quarto/bin/quarto /usr/local/bin/quarto && \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate project_env" >> ~/.bashrc


SHELL ["/bin/bash", "-l", "-c"]

# set container workding dir
WORKDIR /workplace

# copy project files into container
COPY . .

# expose port for jupyter lab
EXPOSE 8888

# run jupyter lab inside the environment
CMD ["conda", "run", "--no-capture-output", "-n", "project_env", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]
 
