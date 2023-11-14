# x-ai-ploit

For dev work:

```bash
pre-commit install
```

# starting

#### Launch Application
```bash
# launch backend
docker build -t xaiploit -f src/backend/Dockerfile .
```

```bash
# launch frontend
docker build -t xaiploit -f src/frontend/Dockerfile .
```

```bash
# launch backend and frontend
docker compose down && docker compose build && docker compose up --build -d && docker image prune -f
```
