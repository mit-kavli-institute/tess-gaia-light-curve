FROM postgres:14-bullseye as pg-base

# Install and set up packages for q3c
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    make \
    gcc \
    libc6-dev \
    liblz4-dev \
    zlib1g-dev \
    libreadline-dev \
    postgresql-server-dev-14 \
    ca-certificates \
    && update-ca-certificates

RUN git clone https://github.com/segasai/q3c /tmp/q3c \
    && cd /tmp/q3c && make && make install && cd .. \
    && rm -rf /tmp/q3c && apt-get clean && rm -rf /var/lib/apt/lists/*

# .sql and .sh scripts in /docker-entrypoint-initdb.d are automatically executed
# in alphabetical order
COPY ./00_install_q3c.sql /docker-entrypoint-initdb.d/

FROM pg-base AS tic
COPY ./ticentries.sql ./ticentries.csv ./dr2_to_dr3.sql ./dr2_to_dr3.csv /docker-entrypoint-initdb.d/

FROM pg-base AS gaia
COPY ./gaia_source.sql ./gaia_source.csv /docker-entrypoint-initdb.d/
