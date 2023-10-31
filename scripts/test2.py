from sentence_transformers.readers import InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.wandb_ import WandbClient
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


# Load the dataset
model_name = "roberta-large"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

model = SentenceTransformer(model_name,wandbc=wandbc)