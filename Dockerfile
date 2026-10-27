FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/
WORKDIR /app
COPY . /app
RUN uv sync --frozen --no-cache
EXPOSE 8000
CMD ["uv", "run", "fastapi", "run", "src/qsar_hepatotoxicity_predictor/api/main.py", "--port", "8000"]