import numpy as np
import pandas as pd
from hls_convert import hls_convert, extract_latency
import tensorflow as tf
from tensorflow.keras import layers, models
from hls_convert import hls_convert, extract_latency

def pytorch_to_keras_rul(pt_model):
    """Crea el modelo equivalente en Keras 1D y transfiere los pesos entrenados de PyTorch."""
    keras_model = models.Sequential([
        layers.Input(shape=(256, 15)),
        layers.Conv1D(filters=16, kernel_size=5, activation='relu', name='encoder_conv1'),
        layers.AveragePooling1D(pool_size=5, name='encoder_pool1'),
        layers.Conv1D(filters=16, kernel_size=5, activation='relu', name='encoder_conv2'),
        layers.AveragePooling1D(pool_size=5, name='encoder_pool2'),
        layers.Conv1D(filters=16, kernel_size=9, activation='relu', name='proj'),
        layers.Flatten(name='flatten'),
        layers.Dense(1, name='fc')
    ])



    # encoder.conv1
    w_conv1 = pt_model.encoder.conv1.weight.detach().cpu().numpy().transpose(2, 1, 0)
    b_conv1 = pt_model.encoder.conv1.bias.detach().cpu().numpy()
    keras_model.get_layer('encoder_conv1').set_weights([w_conv1, b_conv1])

    # encoder.conv2
    w_conv2 = pt_model.encoder.conv2.weight.detach().cpu().numpy().transpose(2, 1, 0)
    b_conv2 = pt_model.encoder.conv2.bias.detach().cpu().numpy()
    keras_model.get_layer('encoder_conv2').set_weights([w_conv2, b_conv2])

    # proj
    w_proj = pt_model.proj.weight.detach().cpu().numpy().transpose(2, 1, 0)
    b_proj = pt_model.proj.bias.detach().cpu().numpy()
    keras_model.get_layer('proj').set_weights([w_proj, b_proj])

    # fc
    w_fc = pt_model.fc.weight.detach().cpu().numpy().T
    b_fc = pt_model.fc.bias.detach().cpu().numpy()
    keras_model.get_layer('fc').set_weights([w_fc, b_fc])

    return keras_model


def simulate_vivado(model_cache, grid_results, dataset_cache, input_shape_cache):
    grid_vivado = []
    for i in range(len(grid_results)):
        model_name = grid_results[i]["model"]
        framework = grid_results[i]["framework"]
        technique = grid_results[i]["technique"]
        rmse_original = grid_results[i]["original_rmse"]
        latency_original = grid_results[i]["original_latency_ms"]
        
        full_name = f"{model_name}-{framework}-{technique}"
        quant_precision = 'fixed<16,6>'
        # print(f"Evaluando: {full_name}")
        # print(f"Speedup: {grid_results[i]['speedup_factor']:.2f}x")
        # print(f"Aumento RMSE: {grid_results[i]['rmse_increase']:.4f}\n")
        # model = model_cache[i]
        pt_model = model_cache[i]
        dataset = dataset_cache[model_name]
        input_shape = input_shape_cache[i]
        # print(input_shape)
        keras_model = pytorch_to_keras_rul(pt_model)

        # convert the model to HLS and compile it to simulate it with Vivado
        hls_model = hls_convert(model=keras_model, full_name=full_name, input_shape=(256, 15), precision=quant_precision)

        x_test = dataset.features
        y_test = dataset.labels

        if hasattr(x_test, 'detach'):
            x_test_np = x_test.detach().cpu().numpy()
        elif isinstance(x_test, (pd.DataFrame, pd.Series)):
            x_test_np = x_test.values
        else:
            x_test_np = np.array(x_test)

        # if x_test_np.ndim == 4:
        #     x_test_np = np.transpose(x_test_np, (0, 2, 3, 1))
        
        # Adaptar dimensiones para Keras (Batch, 256 instantes, 15 canales)
        if x_test_np.ndim == 4:
            b, c, h, w = x_test_np.shape
            x_test_np = x_test_np.reshape(b, c * h, w)

        if x_test_np.ndim == 3 and x_test_np.shape[1] == 15:
            x_test_np = np.transpose(x_test_np, (0, 2, 1))

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
        # print(f"RMSE test: {rmse_test:.6f}")

        _, latency_us = extract_latency(project_path='hls/'+full_name)
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
    return grid_vivado


