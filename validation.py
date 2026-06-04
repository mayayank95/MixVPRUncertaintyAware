import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable

# Same k list as legacy MixVPR Lightning validation (``on_validation_epoch_end``).
MIXVPR_VAL_RECALL_K_VALUES = [1, 5, 10, 15, 20, 50, 100]


def print_recall_pretty_table(
    recalls,
    k_values,
    dataset_name: str,
) -> None:
    """Print the legacy PrettyTable (recalls in percent, 0–100)."""
    print()
    table = PrettyTable()
    k_list = list(k_values)
    table.field_names = ["K"] + [str(k) for k in k_list]
    row = []
    for i in range(len(k_list)):
        row.append(f"{float(recalls[i]) if i < len(recalls) else 0.0:.2f}")
    table.add_row(["Recall@K"] + row)
    print(table.get_string(title=f"Performances on {dataset_name}"))


def _faiss_inputs(r_list, q_list):
    """FAISS torch utils require float32 (Lightning 16-mixed val may output float16)."""
    if torch.is_tensor(r_list):
        return r_list.float().contiguous(), q_list.float().contiguous()
    return np.asarray(r_list, dtype=np.float32), np.asarray(q_list, dtype=np.float32)


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        r_list, q_list = _faiss_inputs(r_list, q_list)
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            # --- old inline PrettyTable (kept for reference) ---
            # print()
            # table = PrettyTable()
            # table.field_names = ['K'] + [str(k) for k in k_values]
            # table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
            # print(table.get_string(title=f"Performances on {dataset_name}"))
            print_recall_pretty_table(
                [100 * float(v) for v in correct_at_k], k_values, dataset_name
            )
        
        return d
