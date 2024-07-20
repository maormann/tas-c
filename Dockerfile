FROM debian@sha256:1dc55ed6871771d4df68d393ed08d1ed9361c577cfeb903cd684a182e8a3e3ae
#FROM debian:bookworm

ENV NO_DOCKER_ENV=true \
    PRISM_JAVAMAXMEM=4g \
    PRISM_JAVASTACKSIZE=1g

# architecture = [x86|arm]
ARG prism_version=4.8.1 \
    prism_arch=x86 \
    uid=1000 \
    gid=1000

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y \
                    default-jdk \
                    wget \
                    bash-completion \
                    python3 \
                    python3-venv \
                    python3-pip \
                    maven \
                    git \
                    yamllint \
 && rm -rf /var/lib/apt/lists/*

# install prism
RUN wget -q -O - https://www.prismmodelchecker.org/dl/prism-${prism_version}-linux64-${prism_arch}.tar.gz | tar -xzf - -C /opt \
 && mv /opt/prism-* /opt/prism \
 && cd /opt/prism \
 && /opt/prism/install.sh \
 && ln -s /opt/prism/bin/prism /usr/local/bin/prism \
 && ln -s /opt/prism/bin/xprism /usr/local/bin/xprism \
 && ln -s /opt/prism/bin/ngprism /usr/local/bin/ngprism

# install Task
RUN sh -c "$(wget -q -O - https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin \
 && wget https://raw.githubusercontent.com/go-task/task/main/completion/bash/task.bash -O /usr/share/bash-completion/completions/task.bash

# add prism user
RUN useradd -ms /bin/bash prism \
 && usermod -u ${uid} prism \
 && groupmod -g ${gid} prism \
 && touch /home/prism/.bashrc

# add .bashrc scripts
RUN echo 'export PATH=${PATH}:/opt/prism/bin' >> /etc/profile \
 && echo '. /usr/share/bash-completion/bash_completion' >> /etc/profile \
 && echo 'if test -f venv/bin/activate; then . venv/bin/activate; fi' >> /root/.bashrc \
 && echo 'if test -f venv/bin/activate; then . venv/bin/activate; fi' >> /home/prism/.bashrc

USER prism

WORKDIR /app
