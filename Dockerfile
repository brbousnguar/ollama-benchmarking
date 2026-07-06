# Dashboard-only image. This containerizes scripts/dashboard_server.py, which is
# a read-only HTTP server over the Markdown reports — nothing else.
#
# The benchmark scripts (ollama_bench.py / foundry_bench.py) are intentionally
# NOT run here: they need native MLX/Metal on Apple Silicon (or the host GPU) and
# talk to a host inference engine, none of which is available inside a Linux
# container. Keep running the benchmarks natively; they write to reports/, and
# this container just serves what's there.
FROM python:3.13-slim

# Skip the repo-local venv bootstrap: the container is already an isolated,
# dependency-free interpreter, so there is nothing to install.
ENV OLLAMA_BENCH_NO_VENV=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only the code the server needs. Reports come in at runtime via a bind mount.
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

EXPOSE 8680

# Reports are mounted read-only at /app/reports (see docker-compose.yml).
CMD ["python3", "scripts/dashboard_server.py", \
     "--host", "0.0.0.0", "--port", "8680", \
     "--reports-dir", "/app/reports"]
