import os
from .configs import *
import datasets
from datasets import load_dataset_builder

# ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']
ds_builder = load_dataset_builder("wmt14", "de-en")

