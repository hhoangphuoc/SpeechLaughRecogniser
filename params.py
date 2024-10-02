import os

GLOBAL_DATA_PATH = "/deepstore/datasets/hmi/speechlaugh-corpus"
HUGGINGFACE_DATA_PATH = "/deepstore/datasets/hmi/speechlaugh-corpus/huggingface_data"
NOISE_DATA_PATH = "/deepstore/datasets/hmi/speechlaugh-corpus/noise_data"

PRECOMPUTED_NOISE_PATH = "datasets/precomputed_noise/"
NUM_NOISE_SEGMENTS = 1500 # 1% of training samples but with 50% random selection