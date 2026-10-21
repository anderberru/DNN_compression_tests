import pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def get_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_sliding_window_dataset_cnn_soh(
    cells, 
    window_size=5, 
    fixed_length=256, 
    mode='relative', 
    as_percentage=False
):
    X, y, cell_ids = [], [], []
    
    for cell_idx, cell in enumerate(cells):
        cell_identifier = get_val(cell, 'cell_id', f"Cell_{cell_idx}")
        cycle_data = get_val(cell, 'cycle_data', [])
        
        cycle_matrices = []
        discharge_capacities = []
        
        for cycle in cycle_data:
            def get_clean_array(obj, attr_name):
                val = get_val(obj, attr_name, [])
                if val is None:
                    return np.array([])
                if not isinstance(val, (list, tuple, np.ndarray)):
                    val = [val]
                clean_val = [x for x in val if x is not None]
                return np.array(clean_val, dtype=float)
            
            v = get_clean_array(cycle, 'voltage_in_V')
            c = get_clean_array(cycle, 'current_in_A')
            t = get_clean_array(cycle, 'time_in_s')
            
            cycle_matrix = np.zeros((3, fixed_length))
            
            if len(v) > 1 and len(c) > 1 and len(t) > 1:
                idx_new = np.linspace(0, 1, fixed_length)
                v_resampled = np.interp(idx_new, np.linspace(0, 1, len(v)), v)
                c_resampled = np.interp(idx_new, np.linspace(0, 1, len(c)), c)
                t_resampled = np.interp(idx_new, np.linspace(0, 1, len(t)), t)
                
                cycle_matrix[0, :] = v_resampled
                cycle_matrix[1, :] = c_resampled
                cycle_matrix[2, :] = t_resampled
                
            cycle_matrices.append(cycle_matrix)
            
            cap = get_val(cycle, 'discharge_capacity_in_Ah', 0.0)
            if isinstance(cap, (list, np.ndarray, torch.Tensor)):
                cap = float(np.max(cap)) if len(cap) > 0 else 0.0
            else:
                cap = float(cap) if cap is not None else 0.0
            discharge_capacities.append(cap)
            
        total_cycles = len(cycle_matrices)
        if total_cycles < window_size:
            continue
            
        if mode == 'relative':
            nominal_cap = get_val(cell, 'nominal_capacity_in_Ah', None)
            if nominal_cap is not None:
                nominal_capacity = float(nominal_cap)
            else:
                nominal_capacity = discharge_capacities[0] if discharge_capacities[0] > 0 else 1.0

        for i in range(total_cycles - window_size + 1):
            window = cycle_matrices[i : i + window_size]
            window_array = np.array(window)
            
            tensor_cnn = np.transpose(window_array, (1, 0, 2))
            
            X.append(tensor_cnn)
            
            current_cycle_idx = i + window_size - 1
            Qd = discharge_capacities[current_cycle_idx]
            
            if mode == 'relative':
                soh = Qd / nominal_capacity
                if as_percentage:
                    soh *= 100.0
            else:
                soh = Qd 
                
            y.append(float(soh))
            cell_ids.append(cell_identifier)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(cell_ids)


CALCE_FILES = [
    "calce_processed/CALCE_CS2_33.pkl",
    "calce_processed/CALCE_CS2_34.pkl",
    "calce_processed/CALCE_CS2_35.pkl",
    "calce_processed/CALCE_CS2_36.pkl",
    "calce_processed/CALCE_CS2_37.pkl",
    "calce_processed/CALCE_CS2_38.pkl",
    "calce_processed/CALCE_CX2_16.pkl",
    "calce_processed/CALCE_CX2_33.pkl",
    "calce_processed/CALCE_CX2_34.pkl",
    "calce_processed/CALCE_CX2_35.pkl",
    "calce_processed/CALCE_CX2_36.pkl",
    "calce_processed/CALCE_CX2_37.pkl",
    "calce_processed/CALCE_CX2_38.pkl",
]


def load_dataset(
    split: str = "test", 
    window_size: int = 5, 
    fixed_length: int = 256
):
    from dataset_folder.myDataset import MyDataset

    class PreWindowedDataset(MyDataset):
        @property
        def sequence_length(self):
            return None

        @sequence_length.setter
        def sequence_length(self, value):
            pass

    project_root = Path(__file__).resolve().parent.parent
    
    cells = []
    for rel_path in CALCE_FILES:
        file_path = project_root / rel_path
        if file_path.exists():
            with file_path.open("rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    cells.extend(data)
                elif isinstance(data, dict):
                    cells.append(data)
                else:
                    cells.append(data)

    n_cells = len(cells)
    n_train = max(1, int(n_cells * 0.8))
    
    if split == "train":
        selected_cells = cells[:n_train]
    elif split == "test":
        selected_cells = cells[n_train:]
    else:
        selected_cells = cells

    X, y, _ = create_sliding_window_dataset_cnn_soh(
        selected_cells, 
        window_size=window_size, 
        fixed_length=fixed_length
    )

    if len(X) > 0:
        N, C, H, W = X.shape
        X_reshaped = X.transpose(0, 2, 3, 1).reshape(-1, C)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        X = X_scaled.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    dataset = PreWindowedDataset(features=X, labels=y, split="all")
    dataset.split = "test"
    return dataset