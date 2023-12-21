# x-ai-ploit

For dev work:

```bash
pre-commit install
conda create -n xploit python=3.10
conda activate xploit
pip install -r docker_dev_requirements.txt
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

```bash
python generate_er_datasets.py [-h] [--directory DIRECTORY] [--output_csv OUTPUT_CSV] [--output_cve_csv OUTPUT_CVE_CSV] [--output_cveless_csv OUTPUT_CVELESS_CSV] [--regen_data REGEN_DATA] [--limit LIMIT]

python generate_er_graphs.py [-h] [--read_path READ_PATH] [--graph_save_path GRAPH_SAVE_PATH] [--feature_save_path FEATURE_SAVE_PATH] [--vectorizer_save_path VECTORIZER_SAVE_PATH] [--limit LIMIT]

python train_er_model.py [-h] [--graph_save_path GRAPH_SAVE_PATH] [--feature_save_path FEATURE_SAVE_PATH]
                              [--vectorizer_save_path VECTORIZER_SAVE_PATH] [--train_perc TRAIN_PERC] [--valid_perc VALID_PERC] [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE]
                              [--weight_decay WEIGHT_DECAY] [--hidden_dim HIDDEN_DIM][--dropout_rate DROPOUT_RATE] [--logging_interval LOGGING_INTERVAL] [--checkpoint_path CHECKPOINT_PATH]
                              [--load_from_checkpoint LOAD_FROM_CHECKPOINT]


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
