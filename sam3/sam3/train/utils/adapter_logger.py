import fnmatch
import logging
import math
from typing import Optional

import torch


class AdapterParamLogger:
    """
    在训练过程中周期性打印 Adapter 参数与梯度的范数，用于确认参数在更新。
    """

    def __init__(
        self,
        param_pattern: str = "backbone.vision_backbone.trunk.prompt_generator.*",
        log_every: int = 100,
    ) -> None:
        self.param_pattern = param_pattern
        self.log_every = log_every
        self._step = 0

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        return model

    def __call__(self, model: torch.nn.Module, rank: int = 0, where: Optional[str] = None):
        if rank != 0:
            return
        self._step += 1
        if self._step % self.log_every != 0:
            return

        model = self._unwrap_model(model)
        param_sq_sum = 0.0
        grad_sq_sum = 0.0
        matched = 0
        grad_matched = 0

        for name, param in model.named_parameters():
            if not fnmatch.fnmatchcase(name, self.param_pattern):
                continue
            matched += 1
            if param.data is not None:
                param_sq_sum += float(param.data.norm(2).item() ** 2)
            if param.grad is not None:
                grad_sq_sum += float(param.grad.data.norm(2).item() ** 2)
                grad_matched += 1

        if matched == 0:
            logging.warning(
                "AdapterParamLogger matched no params with pattern: %s",
                self.param_pattern,
            )
            return

        param_norm = math.sqrt(param_sq_sum)
        grad_norm = math.sqrt(grad_sq_sum) if grad_matched > 0 else 0.0
        where_str = f"{where}/" if where else ""
        logging.info(
            "[%sAdapter] step=%d params=%d param_norm=%.6f grad_norm=%.6f",
            where_str,
            self._step,
            matched,
            param_norm,
            grad_norm,
        )
