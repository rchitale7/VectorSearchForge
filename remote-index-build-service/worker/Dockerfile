FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .. .

# Create user and set permissions
RUN if ! id "appuser" &>/dev/null; then \
    useradd -m appuser; \
fi

RUN chown -R appuser:appuser /app

RUN mkdir -p /app/logs && \
    chmod -R 777 /app/logs