# x-ai-ploit

For dev work:

```bash
pre-commit install
conda create -n xploit python=3.10
conda activate xploit
pip install -r docker_requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -e .
```

```python
#very your gpu installation with
python3
>>> import torch
>>> torch.cuda.is_available()
True
```


# starting

#### Launch Application
`docker context use default`
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
