from copy import deepcopy
from typing import Any
import torch
from .framework import Framework

class LowRankFactorization(Framework):

    def __init__(self, technique: str, **kwargs: Any) -> None:
        super().__init__(technique=technique, **kwargs)

    def compress(self, model: Any, technique: str, **kwargs: Any) -> Any:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("LowRankFactorization requires a torch.nn.Module model.")

        config = {**self.config, **kwargs}
        target_sparsity = float(config.pop("target_sparsity", 0.5))
        inplace = bool(config.pop("inplace", False))

        factorized_model = model if inplace else deepcopy(model)

        def set_module(model_obj, module_name, new_module):
            parts = module_name.split(".")
            parent = model_obj
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        modules_to_replace = []
        for name, module in factorized_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                out_f, in_f = weight.shape
                
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                k = max(1, int(len(S) * (1.0 - target_sparsity)))
                
                U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
                sqrt_S_k = torch.diag(torch.sqrt(S_k))
                w1 = torch.matmul(U_k, sqrt_S_k)
                w2 = torch.matmul(sqrt_S_k, Vh_k)
                
                l1 = torch.nn.Linear(in_f, k, bias=False)
                l1.weight.data = w2
                
                l2 = torch.nn.Linear(k, out_f, bias=(module.bias is not None))
                l2.weight.data = w1
                if module.bias is not None:
                    l2.bias.data = module.bias.data.clone()
                    
                modules_to_replace.append((name, torch.nn.Sequential(l1, l2)))
            if isinstance(module, torch.nn.Conv2d):
                if module.groups != 1:
                    continue

                weight = module.weight.data 
                out_c, in_c, kh, kw = weight.shape

                weight_2d = weight.view(out_c, in_c * kh * kw)

                U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
                k = max(1, int(len(S) * (1.0 - target_sparsity)))

                U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
                sqrt_S_k = torch.diag(torch.sqrt(S_k))

                w1 = torch.matmul(U_k, sqrt_S_k)
                w2 = torch.matmul(sqrt_S_k, Vh_k)

                w2_conv = w2.view(k, in_c, kh, kw)
                w1_conv = w1.view(out_c, k, 1, 1)

                c1 = torch.nn.Conv2d(
                    in_channels=in_c,
                    out_channels=k,
                    kernel_size=(kh, kw),
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=1,
                    bias=False
                )
                c1.weight.data = w2_conv

                c2 = torch.nn.Conv2d(
                    in_channels=k,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=(module.bias is not None)
                )
                c2.weight.data = w1_conv
                if module.bias is not None:
                    c2.bias.data = module.bias.data.clone()

                modules_to_replace.append((name, torch.nn.Sequential(c1, c2)))
            if isinstance(module, torch.nn.Conv1d):
                if module.groups != 1:
                    continue

                weight = module.weight.data
                out_c, in_c, kw = weight.shape

                weight_2d = weight.view(out_c, in_c * kw)

                U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
                k = max(1, int(len(S) * (1.0 - target_sparsity)))

                U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
                sqrt_S_k = torch.diag(torch.sqrt(S_k))

                w1 = torch.matmul(U_k, sqrt_S_k)
                w2 = torch.matmul(sqrt_S_k, Vh_k)

                w2_conv = w2.view(k, in_c, kw)
                w1_conv = w1.view(out_c, k, 1)

                c1 = torch.nn.Conv1d(
                    in_channels=in_c,
                    out_channels=k,
                    kernel_size=kw,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=1,
                    bias=False
                )
                c1.weight.data = w2_conv

                c2 = torch.nn.Conv1d(
                    in_channels=k,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=(module.bias is not None)
                )
                c2.weight.data = w1_conv
                if module.bias is not None:
                    c2.bias.data = module.bias.data.clone()

                modules_to_replace.append((name, torch.nn.Sequential(c1, c2)))

        for name, new_module in modules_to_replace:
            set_module(factorized_model, name, new_module)

        return factorized_model
    