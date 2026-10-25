import pickle
from pathlib import Path
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler


def get_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_sliding_window_dataset_soh(
    cells, 
    window_size=5, 
    feature_names=None, 
    flatten=True, 
    mode='relative', 
    as_percentage=False
):
    if feature_names is None:
        feature_names = [
            'discharge_duration_s', 
            'time_CV_s', 
            'delta_V_drop'
        ]
        
    X, y, cell_ids = [], [], []
    
    for cell_idx, cell in enumerate(cells):
        cell_identifier = get_val(cell, 'cell_id', f"Cell_{cell_idx}")
        cycle_data = get_val(cell, 'cycle_data', [])
        
        cycle_features = []
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
            temp = get_clean_array(cycle, 'temperature_in_C')
            r = get_clean_array(cycle, 'internal_resistance_in_ohm')
            
            # 1. Discharge duration
            discharge_idx = np.where(c < -0.05)[0] 
            discharge_duration = (t[discharge_idx[-1]] - t[discharge_idx[0]]) if (len(discharge_idx) > 1 and len(t) > 0) else 0.0
                
            # 2. Time in CV phase
            cv_idx = np.where((v > 4.15) & (c > 0.01))[0]
            time_cv = (t[cv_idx[-1]] - t[cv_idx[0]]) if (len(cv_idx) > 1 and len(t) > 0) else 0.0
                
            # 3. Initial voltage drop
            if len(discharge_idx) > 0 and len(v) > 0:
                start_idx = discharge_idx[0]
                v_before = v[start_idx - 1] if start_idx > 0 else v[start_idx]
                v_after = v[min(start_idx + 5, len(v) - 1)]
                delta_v_drop = v_before - v_after
            else:
                delta_v_drop = 0.0
                
            # 4. Temperature rise
            delta_t = (np.max(temp) - np.min(temp)) if len(temp) > 0 else 0.0
                
            # 5. Mean internal resistance
            valid_r = r[r > 0]
            r_mean = np.mean(valid_r) if len(valid_r) > 0 else 0.0

            extracted_features_dict = {
                'discharge_duration_s': float(discharge_duration),
                'time_CV_s': float(time_cv),
                'delta_V_drop': float(delta_v_drop),
                'delta_T': float(delta_t),
                'R_internal': float(r_mean)
            }
            
            current_cycle_vector = [extracted_features_dict[feat] for feat in feature_names]
            cycle_features.append(current_cycle_vector)
            
            cap = get_val(cycle, 'discharge_capacity_in_Ah', 0.0)
            if isinstance(cap, (list, np.ndarray, torch.Tensor)):
                cap = float(np.max(cap)) if len(cap) > 0 else 0.0
            else:
                cap = float(cap) if cap is not None else 0.0
            discharge_capacities.append(cap)
                
        total_cycles = len(cycle_features)
        if total_cycles < window_size:
            continue
            
        if mode == 'relative':
            nominal_cap = get_val(cell, 'nominal_capacity_in_Ah', None)
            if nominal_cap is not None:
                nominal_capacity = float(nominal_cap)
            else:
                nominal_capacity = discharge_capacities[0] if discharge_capacities[0] > 0 else 1.0

        for i in range(total_cycles - window_size + 1):
            window = cycle_features[i : i + window_size]
            window_array = np.array(window)
            
            if window_array.shape == (window_size, len(feature_names)):
                if flatten:
                    X.append(window_array.flatten())
                else:
                    X.append(window_array)
                    
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


def load_dataset(split: str = "test", window_size: int = 5, flatten: bool = True):
    from dataset_folder.myDataset import MyDataset

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

            is_test = False
            for test_id in test_ids:
                if test_id in str(file_path):
                    is_test = True
                    break

            with file_path.open("rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    if is_test:
                        test_cells.extend(data)
                    else:
                        train_cells.extend(data)
                elif isinstance(data, dict):
                    if is_test:
                        test_cells.append(data)
                    else:
                        train_cells.append(data)
                else:
                    if is_test:
                        test_cells.append(data)
                    else:
                        train_cells.append(data)

    if split == "train":
        selected_cells = train_cells
    elif split == "test":
        selected_cells = test_cells
    else:
        selected_cells = train_cells + test_cells

    X, y, _ = create_sliding_window_dataset_soh(
        selected_cells, 
        window_size=window_size, 
        flatten=flatten
    )

    # scaler = StandardScaler()
    scaler = joblib.load('dataset_folder/scaler_mlp.joblib')
    X = scaler.transform(X)

    return MyDataset(features=X, labels=y, split="all")