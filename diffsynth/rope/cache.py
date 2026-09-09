import torch
import hashlib
from typing import Optional

import torch
from typing import Optional
from collections import OrderedDict

import torch
from typing import Optional, Deque, Tuple
from collections import deque

class SOFCache:
    def __init__(self, eps: float = 1e-4, max_entries: int = 6):
        """
        Cache that stores up to `max_entries` (M, value, P) tuples globally.
        Each P can have multiple M entries.
        
        On get(M, P):
          - Only consider entries with matching P
          - Return value of first entry where MSE(cached_M, M) < eps
        """
        self.entries: Deque[Tuple[int, torch.Tensor, torch.Tensor]] = deque()  # (P, M, value)
        self.eps = eps
        self.max_entries = max_entries

    def get(self, M: torch.Tensor, P: int) -> Optional[torch.Tensor]:
        """Search among entries with matching P for an M close enough to input M."""
        for cached_P, cached_M, cached_value in self.entries:
            if cached_P != P:
                continue
            if cached_M.shape != M.shape:
                continue
            mse = torch.max((cached_M - M) ** 2).item()
            if mse < self.eps:
                return cached_value.clone()
        return None

    def put(self, M: torch.Tensor, P: int, value: torch.Tensor) -> None:
        """Add a new (P, M, value) entry. Evict oldest if over capacity."""
        # Add new entry
        self.entries.append((P, M.detach().clone(), value.detach().clone()))
        # Enforce size limit (FIFO)
        while len(self.entries) > self.max_entries:
            self.entries.popleft()  # remove oldest

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        self.entries.clear()

class SE3ToSO4ElementCache:
    def __init__(self, max_size: int = 1024, eps: float = 1e-5, quantize_decimals: int = 5):
        self.cache = {}  # key (tuple) -> (X_ref, output)
        self.max_size = max_size
        self.eps = eps
        self.quantize_decimals = quantize_decimals
        self.scale = 10 ** quantize_decimals

    def _make_key(self, X: torch.Tensor) -> tuple:
        """Create a hashable key from quantized X. Keeps device/dtype agnostic."""
        # Work in float32 for stability
        if X.dtype in (torch.bfloat16, torch.float16):
            X = X.float()
        # Quantize in-place mathematically (no CPU transfer!)
        X_q = torch.round(X * self.scale) / self.scale
        # Convert to tuple of Python floats (hashable, fast for 16 elements)
        return tuple(X_q.flatten().tolist())

    def get_or_compute(self, X: torch.Tensor, compute_fn: callable) -> torch.Tensor:
        """X is (4, 4). Returns cached or computed SO(4) element."""
        key = self._make_key(X)

        # Fast path: key exists
        if key in self.cache:
            X_ref, output = self.cache[key]
            # Optional: skip distance check if quantization is trusted
            # But keep it for safety (in case quantization bins are too coarse)
            if X_ref.shape == X.shape:
                diff_norm = torch.norm(X_ref - X, p='fro').item()
                if diff_norm < self.eps:
                    return output.clone()

        # Compute
        with torch.no_grad():
            output = compute_fn(X.unsqueeze(0)).squeeze(0)  # (4,4)

        # Evict if full (FIFO via insertion order — Python 3.7+ dict is ordered)
        if len(self.cache) >= self.max_size:
            del self.cache[next(iter(self.cache))]

        self.cache[key] = (X.clone(), output.clone())
        return output