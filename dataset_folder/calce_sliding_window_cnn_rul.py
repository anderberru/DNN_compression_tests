import pickle
from pathlib import Path
import numpy as np
import torch
import joblib


def get_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_sliding_window_dataset_cnn_rul(
    cells, 
    window_size=5, 
    fixed_length=256,
    piecewise_cap=None  # Optional RUL cap to avoid overly large target values.
):
    X, y, cell_ids = [], [], []
    
    for cell_idx, cell in enumerate(cells):
        cell_identifier = get_val(cell, 'cell_id', f"Cell_{cell_idx}")
        cycle_data = get_val(cell, 'cycle_data', [])
        cycle_matrices = []
        
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
            
            # Normalize time so it always starts at 0, which avoids instability.
            t = t - t[0] if len(t) > 0 else t
            
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
            
        total_cycles = len(cycle_matrices)
        if total_cycles < window_size:
            continue

        for i in range(total_cycles - window_size + 1):
            window = cycle_matrices[i : i + window_size]
            window_array = np.array(window)
            
            # REORDER DIMENSIONS FOR PYTORCH -> (channels, window, length).
            tensor_cnn = np.transpose(window_array, (1, 0, 2))
            X.append(tensor_cnn)
            
            current_cycle_idx = i + window_size - 1
            rul_lineal = total_cycles - current_cycle_idx - 1
            
            if piecewise_cap is not None:
                rul_final = min(rul_lineal, piecewise_cap)
            else:
                rul_final = rul_lineal
                
            y.append(float(rul_final))
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
    fixed_length: int = 64,
    piecewise_cap: int | None = None,
    scaler_path: str = "models/rul_files/scaler64normal_rul.joblib"
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

    test_ids = [
        'RWTH_015', 'UL-PUR_N15-OV3_18650_NCA_23C_0-100_0.5-0.5C_c',
        'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_j', 'RWTH_048',
        'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_n', 'CALCE_CX2_16',
        'RWTH_010', 'RWTH_005', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_l',
        'RWTH_018', 'RWTH_029', 'RWTH_032', 'CALCE_CX2_35', 'RWTH_043',
        'RWTH_014', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_f', 'RWTH_007',
        'CALCE_CX2_38', 'RWTH_046', 'RWTH_011',
        'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_e', 'RWTH_039', 'RWTH_002',
        'RWTH_049', 'RWTH_020', 'RWTH_036', 'RWTH_013', 'RWTH_026',
        'RWTH_037', 'RWTH_040', 'RWTH_008', 'CALCE_CS2_38', 'RWTH_028',
        'RWTH_024'
    ]
    
    train_cells = []
    test_cells = []
    for rel_path in CALCE_FILES:
        file_path = project_root / rel_path
        if file_path.exists():
            is_test = any(test_id in str(file_path) for test_id in test_ids)

            with file_path.open("rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    (test_cells if is_test else train_cells).extend(data)
                else:
                    (test_cells if is_test else train_cells).append(data)

    if split == "train":
        selected_cells = train_cells
    elif split == "test":
        selected_cells = test_cells
    else:
        selected_cells = train_cells + test_cells

    X, y, _ = create_sliding_window_dataset_cnn_rul(
        selected_cells, 
        window_size=window_size, 
        fixed_length=fixed_length,
        piecewise_cap=piecewise_cap
    )

    if len(X) > 0:
        N, C, H, W = X.shape
        X_flat = X.reshape(N, -1)
        
        scaler = joblib.load(project_root / scaler_path)
        X_scaled = scaler.transform(X_flat)
        
        X = X_scaled.reshape(N, C, H, W)

    dataset = PreWindowedDataset(features=X, labels=y, split="all")
    dataset.split = "test"
    return dataset
