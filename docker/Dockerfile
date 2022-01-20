FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# apt-get install some things
RUN apt-get update && apt-get -y --no-install-recommends install \
    ca-certificates \
    curl \
    python3 \
    python3-dev \
    sudo \
    python3-pip \
    vim \
    git

# Install pytorch 1.10.1, torchvision 0.11.2, for CUDA 11.3
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
RUN git clone https://github.com/NVIDIA/apex /root/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /root/apex/

# Install gosu for dealing with user perms
RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
	gosu nobody true

# Install project dependencies.
COPY ../requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt

# Copy the entrypoint script to the container.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh 
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
