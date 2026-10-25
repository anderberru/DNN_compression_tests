import time
import pyRAPL
import torch
import numpy as np

pyRAPL.setup()


def unpack_batch(batch):
    if isinstance(batch, dict):
        if "feature" in batch and "label" in batch:
            return batch["feature"], batch["label"]
        if "input" in batch and "target" in batch:
            return batch["input"], batch["target"]
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch
    raise ValueError(f"Unsupported batch format: {type(batch).__name__}")

def forward_generic(model, batch):
    feature, label = unpack_batch(batch)
    with torch.no_grad():
        try:
            return model(feature, label)
        except TypeError:
            return model(feature)

def compute_comparison_metrics(metric_values, metric_name="rmse_increase"):

    value = 0.0
    metric_key_map = {
        "abs_error_increase": "max_abs_error",
        "mse_increase": "mse",
        "mae_increase": "mae",
        "rmse_increase": "rmse",
        "mape_increase": "mape",
    }
    if metric_name in metric_key_map:
        metric_key = metric_key_map[metric_name]
        value = (
            metric_values["compressed"][metric_key]
            - metric_values["original"][metric_key]
        )
    if metric_name == "speedup_factor":
        value = (
            metric_values["original"]["latency_ms_per_sample"]
            / metric_values["compressed"]["latency_ms_per_sample"]
        )
    return value

def compare_models(original_model, compressed_model, dataloader, device="cpu"):
    original_model.eval()
    compressed_model.eval()

    metrics = {
        "original": {
            "mse": [],
            "mae": [],
            "rmse": [],
            # "latency_ms_per_sample": 0.0,
        },
        "compressed": {
            "mse": [],
            "mae": [],
            "rmse": [],
            # "mape": [],
            # "max_abs_error": [],
            # "latency_ms_per_sample": 0.0,
        },
    }
    sample_count = 0
    inference_time = {"original": 0.0, "compressed": 0.0}

    original_energy_consumption = []
    compressed_energy_consumption = []
    meter_orig = pyRAPL.Measurement("original_model")
    meter_comp = pyRAPL.Measurement("compressed_model")

    total_squared_error = {"original": 0.0, "compressed": 0.0}
    total_abs_error = {"original": 0.0, "compressed": 0.0}

    with torch.no_grad():
        for batch in dataloader:
            feature, label = unpack_batch(batch)

            feature = feature.to(device)
            label = label.to(device)
            sample_count += feature.shape[0]

            if feature.is_cuda:
                torch.cuda.synchronize(feature.device)
            meter_orig.begin()
            start_time = time.perf_counter()
            # print(f"modelo", {original_model})
            # print(f"FORMA DEL DATO EN EVALUACIÓN: {feature.shape}")
            y_orig = forward_generic(original_model, {"feature": feature, "label": label})
            
            if feature.is_cuda:
                torch.cuda.synchronize(feature.device)
            inference_time["original"] += time.perf_counter() - start_time
            meter_orig.end()
            original_energy_consumption.append(sum(meter_orig.result.pkg) / 1e6)

            if feature.is_cuda:
                torch.cuda.synchronize(feature.device)
            meter_comp.begin()
            start_time = time.perf_counter()
            y_comp = forward_generic(compressed_model, {"feature": feature, "label": label})
            if feature.is_cuda:
                torch.cuda.synchronize(feature.device)
            inference_time["compressed"] += time.perf_counter() - start_time
            meter_comp.end()
            compressed_energy_consumption.append(sum(meter_comp.result.pkg) / 1e6)

            y_orig = y_orig.reshape_as(label)
            y_comp = y_comp.reshape_as(label)

            for model_name, prediction in (
                ("original", y_orig),
                ("compressed", y_comp),
            ):
                error = prediction - label
                batch_squared_error = torch.sum(error.pow(2)).item()
                batch_abs_error = torch.sum(torch.abs(error)).item()
                
                total_squared_error[model_name] += batch_squared_error
                total_abs_error[model_name] += batch_abs_error

                

    for model_name in ["original", "compressed"]:
        final_mse = total_squared_error[model_name] / sample_count
        final_rmse = final_mse ** 0.5
        final_mae = total_abs_error[model_name] / sample_count

        metrics[model_name]["mse"].append(final_mse)
        metrics[model_name]["mae"].append(final_mae)
        metrics[model_name]["rmse"].append(final_rmse)
        # metrics[model_name]["mape"].append(mape_value)
        # metrics[model_name]["max_abs_error"].append(
        #     torch.max(torch.abs(error)).item()
        # )

    mean_original_energy = np.mean(original_energy_consumption)
    mean_compressed_energy = np.mean(compressed_energy_consumption)
    
    comparison_results = {
        model_name: {
            metric_name: sum(values) / len(values)
            for metric_name, values in model_metrics.items()
        }
        for model_name, model_metrics in metrics.items()
    }
    for model_name in comparison_results:
        comparison_results[model_name]["latency_ms_per_sample"] = (
            inference_time[model_name] * 1000 / sample_count
        )

    comparison_results["original"]["energy_consumption_J"] = mean_original_energy
    comparison_results["compressed"]["energy_consumption_J"] = mean_compressed_energy

    return comparison_results
