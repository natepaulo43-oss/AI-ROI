import pandas as pd
from pathlib import Path

def build_dataset():
    raw_data_path = Path("../data/raw")
    processed_data_path = Path("../data/processed")
    
    print("Building dataset...")
