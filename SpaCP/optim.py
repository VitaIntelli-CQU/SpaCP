from __future__ import annotations

from math import exp
import logging

import numpy as np
import torch
import torch.nn as nn

__all__ = ["PIDControl", "EarlyStopping"]

class PIDControl():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, init_beta, min_beta, max_beta):
        """define them out of loop"""
        self.W_k1 = init_beta
        self.W_min = min_beta
        self.W_max = max_beta
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = (exp_KL - kl_loss) * 5.   # we enlarge the error 5 times to allow faster tuning of beta
        ## comput U as the control factor
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k

        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk
        self.e_k1 = error_k

        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        if Wk > self.W_max:
            Wk = self.W_max

        return Wk, error_k

# ---------------------------------------------------------------------
# LORD Encoder
# ---------------------------------------------------------------------

class EarlyStopping:
    """Early stopping with best-weight saving based on validation loss."""

    def __init__(self, patience: int = 10, verbose: bool = False, modelfile: str = "model.pt") -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile

    def __call__(self, loss: float, model: nn.Module) -> None:
        if np.isnan(loss):
            self.early_stop = True
            return
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self._save(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self._save(loss, model)
            self.counter = 0

    def _save(self, loss: float, model: nn.Module) -> None:
        if self.verbose:
            logging.info("Validation loss improved: %.6f → %.6f (saving)", self.loss_min, loss)
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
