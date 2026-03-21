# set base image
FROM condaforge/miniforge3:25.9.1-0

# copy lock file
COPY conda-lock.yml /tmp/conda-lock.yml

# install system dependencies for Quarto and TinyTeX
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    curl \
    wget \
    gdebi-core \
    xz-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install conda-lock and create environment from lock file
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n project_env /tmp/conda-lock.yml && \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate project_env" >> ~/.bashrc

# install Quarto using architecture-aware .deb package
RUN ARCH=$(dpkg --print-architecture) && \
    curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.8.26/quarto-1.8.26-linux-${ARCH}.deb && \
    gdebi --non-interactive quarto-1.8.26-linux-${ARCH}.deb && \
    rm quarto-1.8.26-linux-${ARCH}.deb

# install TinyTeX for PDF rendering
RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh && \
    /root/.TinyTeX/bin/*/tlmgr path add

# pre-install common LaTeX packages used in Quarto PDFs
RUN /root/.TinyTeX/bin/*/tlmgr install koma-script caption xcolor

SHELL ["/bin/bash", "-l", "-c"]

# set container working directory
WORKDIR /workplace

# copy project files into container
COPY . .

# start shell
CMD ["/bin/bash", "--login", "-i"]
