# x-ai-ploit

For dev work:

```bash
pre-commit install
```

# starting

#### Launch Application
```bash
docker compose down && docker compose build && docker compose up --build -d && docker image prune -f
```

#### Generate Features
```python
# generating features and cleaning data
docker exec -it xaiploit-backend-1 python main.py generate-features
```

#### Train Model
```python
docker exec -it xaiploit-backend-1 python main.py train-model
```

#### Score Data
```python
docker exec -it xaiploit-backend-1 python main.py score-data
```
