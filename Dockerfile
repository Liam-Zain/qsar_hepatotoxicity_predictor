FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY . /app
RUN uv sync --frozen --no-cache
EXPOSE 8000
CMD ["uv", "run", "fastapi", "run", "src/qsar-hepatotoxicity-predictor/api/main.py", "--port", "8000"]