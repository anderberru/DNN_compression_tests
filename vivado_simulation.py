import numpy as np
import pandas as pd
from hls_convert import hls_convert, extract_latency
# import tensorflow as tf
# from tensorflow.keras import layers, models
import torch
from hls_convert import hls_convert, extract_latency
import os
import shutil


def simulate_vivado(model_cache, grid_results, dataset_cache, input_shape_cache):
    grid_vivado = []
    for i in range(len(grid_results)):
        model_name = grid_results[i]["model"]
        framework = grid_results[i]["framework"]
        technique = grid_results[i]["technique"]
        rmse_original = grid_results[i]["original_rmse"]
        latency_original = grid_results[i]["original_latency_ms"]
        
        full_name = f"{model_name}-{framework}-{technique}"
        hls_project_dir = 'hls/' + full_name
        quant_precision = 'fixed<16,6>'

        pt_model = model_cache[i]
        dataset = dataset_cache[model_name]
        input_shape = input_shape_cache[i]

        # convert the model to HLS and compile it to simulate it with Vivado
        hls_model = hls_convert(model=pt_model, full_name=full_name, input_shape=input_shape, precision=quant_precision)

        x_test = dataset.features
        y_test = dataset.labels

        if hasattr(x_test, 'detach'):
            x_test_np = x_test.detach().cpu().numpy()
        elif isinstance(x_test, (pd.DataFrame, pd.Series)):
            x_test_np = x_test.values
        else:
            x_test_np = np.array(x_test)


        X_test_np = np.ascontiguousarray(x_test_np, dtype=np.float32)

        y_pred_hls = hls_model.predict(X_test_np)

        if hasattr(y_test, 'detach'):
            y_true = y_test.detach().cpu().numpy().ravel()
        elif isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_true = y_test.values.ravel()
        else:
            y_true = np.array(y_test).ravel()

        y_pred = np.array(y_pred_hls).ravel()

        rmse_test = np.sqrt(np.mean((y_true - y_pred) ** 2))

        _, latency_us = extract_latency(project_path=hls_project_dir)
        # print(f"Latency: {latency_us:.2f} us")
        latency_ms = latency_us * 0.001
        # print(f"Latency: {latency_ms:.4f} ms")
        rmse_increase = rmse_test - rmse_original
        speedup_factor = latency_original / latency_ms

        grid_vivado.append(
            {
                "model": model_name,
                "framework": framework,
                "technique": technique,
                "quant_precision": quant_precision,
                "original_rmse": rmse_original,
                "compressed_rmse": rmse_test,
                "rmse_increase": rmse_increase,
                "original_latency_ms": latency_original,
                "compressed_latency_ms": latency_ms,
                "speedup_factor": speedup_factor,
            }
        )

        if os.path.exists(hls_project_dir):
            try:
                shutil.rmtree(hls_project_dir, ignore_errors=True)
                print(f" [Clean] HLS project deleted: {hls_project_dir}")
            except Exception as err:
                print(f" [Warning] Could not delete {hls_project_dir}: {err}")

    return grid_vivado


