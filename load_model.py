import torch
import numpy as np

def load_model(path):
    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file with torch.load: {e}")
        raise e
        
    return data


def main():
    path = "saved_models/task-0percent_dim-5_class-HypCones_init_class-PoincareNIPS_neg_sampl_strategy-true_neg_lr-0.0003_epochs-300_opt-rsgd_where_not_to_sample-children_neg_edges_attach-parent_lr_init-0.03_epochs_init-100_n_model.pt"
    model_data = load_model(path)

    print("=== Keys in the saved dictionary ===")
    for k in model_data.keys():
        print(f" - {k}")

    print("\n=== Embedding vectors shape ===")
    vectors = model_data["vectors"]
    
    if isinstance(vectors, torch.Tensor):
        vectors_shape = vectors.shape
        print(f"PyTorch Tensor Shape: {vectors_shape}")
        
    elif isinstance(vectors, np.ndarray):
        print(f"NumPy Array Shape: {vectors.shape}")
        
    else:
        print(f"Type: {type(vectors)}")

    print("\n=== First 5 vocab items ===")
    vocab = model_data["vocab"]
    print(vocab[:5])

    print("\n=== Best α (ranking alpha) ===")
    print(model_data.get("best_alpha"))

    print("\n=== Best F1 values ===")
    print("INIT  test:", model_data.get("best_init_test_f1"))
    print("INIT  valid:", model_data.get("best_init_valid_f1"))
    print("CONES test:", model_data.get("best_cones_test_f1"))
    print("CONES valid:", model_data.get("best_cones_valid_f1"))

    print("\n=== Hyperparameters ===")
    params = model_data["params"]
    if isinstance(params, dict):
        for k in sorted(params.keys()):
            print(f"{k}: {params[k]}")
    else:
        print(f"Params is not a dictionary: {type(params)}")


if __name__ == "__main__":
    main()