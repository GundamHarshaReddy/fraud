import pandas as pd
from pathlib import Path
from train_thagat import build_hetero_graph

def main():
    processed_dir = Path("data/processed")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    print(f"Total nodes: {len(train_df)}")
    
    # We just run graph building and intercept logs
    data = build_hetero_graph(train_df)
    
    edge_index = data[1]
    edge_type = data[2]
    
    total_edges = edge_index.shape[1] // 2  # Undirected edges
    print(f"Total undirected edges: {total_edges}")
    
    for r in range(6):
        count = (edge_type == r).sum().item() // 2
        print(f"Edge type {r} count: {count}")

if __name__ == '__main__':
    main()
