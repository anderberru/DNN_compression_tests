import torch


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

def compare_models(original_model, compressed_model, dataloader, device="cpu"):
    original_model.eval()
    compressed_model.eval()

    metrics = {
        "mse": [],
        "mae": [],
        "max_abs_error": [],
    }

    for batch in dataloader:
        feature, label = unpack_batch(batch)

        feature = feature.to(device)
        label = label.to(device)

        y_orig = forward_generic(original_model, {"feature": feature, "label": label})
        y_comp = forward_generic(compressed_model, {"feature": feature, "label": label})

        y_orig = y_orig.reshape_as(label)
        y_comp = y_comp.reshape_as(label)

        diff = y_orig - y_comp
        metrics["mse"].append(torch.mean(diff.pow(2)).item())
        metrics["mae"].append(torch.mean(torch.abs(diff)).item())
        metrics["max_abs_error"].append(torch.max(torch.abs(diff)).item())

    return {k: sum(v) / len(v) for k, v in metrics.items()}
