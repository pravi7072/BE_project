"""
Robust collate function for variable-length mels and dict-based dataset outputs.
Usage: DataLoader(..., collate_fn=collate_fn)
"""

from typing import List, Dict, Any
import torch


def collate_fn(batch: List[Dict[str, Any]]):
    if batch is None or len(batch) == 0:
        return None

    # Filter out invalid entries early
    batch = [b for b in batch if b is not None and isinstance(b, dict)]
    if len(batch) == 0:
        return None

    # Collect keys (use keys from first valid item)
    keys = list(batch[0].keys())
    out = {}

    for key in keys:
        vals = [b.get(key, None) for b in batch]
        # Drop items where this key is missing
        filtered = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(filtered) == 0:
            continue

        indices, values = zip(*filtered)

        # If tensors, try to pad them
        if all(torch.is_tensor(v) for v in values):
            tensors = list(values)
            # for mel: shape usually (n_mels, T)
            if all(t.dim() == 2 for t in tensors):
                # 🔥 FIX: use tensors directly (not batch again)
                max_t = max(t.shape[-1] for t in tensors)
                padded = [
                    torch.nn.functional.pad(t, (0, max_t - t.shape[-1]))
                    for t in tensors
                ]
                out[key] = torch.stack(padded)
            elif tensors[0].dim() == 1:
                out[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
            else:
                try:
                    out[key] = torch.stack(tensors)
                except Exception:
                    # 🔥 FIX: FORCE PAD + STACK instead of list
                    if tensors[0].dim() >= 2:
                        max_t = max(t.shape[-1] for t in tensors)
                        padded = [torch.nn.functional.pad(t, (0, max_t - t.shape[-1])) for t in tensors]
                        out[key] = torch.stack(padded)
                    else:
                        out[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        else:
            # Non-tensor metadata: return list (in original order)
            ordered = [None] * len(batch)
            for idx, v in zip(indices, values):
                ordered[idx] = v
            # compact (remove None)
            out[key] = [x for x in ordered if x is not None]

    # If no tensor keys found, return None to let trainer skip
    if not any(torch.is_tensor(v) for v in out.values()):
        return None

    return out
