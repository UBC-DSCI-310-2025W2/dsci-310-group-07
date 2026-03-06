# set base image  
FROM condaforge/miniforge3:25.9.1-0

# copy lock file
COPY conda-lock.yml /tmp/conda-lock.yml

# install conda forge and cond lock
RUN conda install -c conda-forge conda-lock -y

# create environment and install its corresponding packages from lock file
RUN conda-lock install -n project_env /tmp/conda-lock.yml

# activate environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate project_env" >> ~/.bashrc

SHELL ["/bin/bash", "-l", "-c"]

# set container workding dir
WORKDIR /workplace

# copy project files into container
COPY . .

# expose port for jupyter lab
EXPOSE 8888

# run jupyter lab inside the environment
CMD ["conda", "run", "--no-capture-output", "-n", "project_env", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]
