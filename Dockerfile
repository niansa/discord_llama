FROM debian:bookworm
COPY . /app/
RUN cd /app \
    && apt update \
    && apt -y install git g++ make cmake zlib1g-dev libssl-dev libsqlite3-dev \
    && apt clean \
    && cmake . \
    && make -j$(nproc)
CMD cd /app && ./discord_llama ./config.txt
