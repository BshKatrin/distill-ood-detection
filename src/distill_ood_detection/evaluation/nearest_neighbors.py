"""Exact nearest-neighbor search with FAISS."""

from __future__ import annotations

import numpy as np
import torch


class FaissExactL2Index:
    """Reusable exact L2 index backed by FAISS on CPU or CUDA."""

    def __init__(
        self,
        references: torch.Tensor,
        device: torch.device,
        *,
        gpu_resources: object | None = None,
    ) -> None:
        import faiss

        if references.ndim != 2:
            raise ValueError("references must be two-dimensional")
        if references.shape[0] == 0:
            raise ValueError("references must not be empty")

        self.dimension = references.shape[1]
        self.reference_count = references.shape[0]
        self._gpu_resources: object | None = None
        cpu_index = faiss.IndexFlatL2(self.dimension)
        if device.type == "cuda":
            gpu_count = faiss.get_num_gpus()
            if gpu_count == 0:
                raise RuntimeError(
                    "CUDA was selected, but this FAISS installation has no GPU support"
                )
            gpu_id = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            if gpu_id >= gpu_count:
                raise ValueError(
                    f"FAISS GPU index {gpu_id} is unavailable; found {gpu_count} GPU(s)"
                )
            self._gpu_resources = (
                gpu_resources
                if gpu_resources is not None
                else faiss.StandardGpuResources()
            )
            self._index = faiss.index_cpu_to_gpu(
                self._gpu_resources,
                gpu_id,
                cpu_index,
            )
            self.device = f"cuda:{gpu_id}"
        else:
            self._index = cpu_index
            self.device = "cpu"

        self._index.add(_as_float32_array(references))

    def search(
        self,
        queries: torch.Tensor,
        *,
        k: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact Euclidean distances and reference indices."""

        if queries.ndim != 2:
            raise ValueError("queries must be two-dimensional")
        if queries.shape[1] != self.dimension:
            raise ValueError("queries and references must have the same dimension")
        if k <= 0:
            raise ValueError("k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        effective_k = min(k, self.reference_count)
        all_distances: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for start in range(0, queries.shape[0], batch_size):
            query_batch = _as_float32_array(queries[start : start + batch_size])
            squared_distances, indices = self._index.search(query_batch, effective_k)
            distances = np.sqrt(np.maximum(squared_distances, 0.0))
            all_distances.append(torch.from_numpy(distances))
            all_indices.append(torch.from_numpy(indices))
        if not all_distances:
            return (
                torch.empty((0, effective_k), dtype=torch.float32),
                torch.empty((0, effective_k), dtype=torch.int64),
            )
        return torch.cat(all_distances), torch.cat(all_indices)


class MaskedChannelExactL2Index:
    """Exact batched L2 search over unmasked channels of feature maps.

    References remain resident on ``device``. Search is chunked over queries
    and references, and only the current global top-k candidates are retained.
    """

    def __init__(self, references: torch.Tensor, device: torch.device) -> None:
        if references.ndim != 4:
            raise ValueError("references must have shape (N, C, H, W)")
        if references.shape[0] == 0:
            raise ValueError("references must not be empty")
        if not references.is_floating_point():
            raise ValueError("references must be floating point")

        self.feature_shape = tuple(references.shape[1:])
        self.reference_count = references.shape[0]
        self.device = device
        self._references = (
            references.detach()
            .to(
                device=device,
                dtype=torch.float32,
            )
            .contiguous()
        )
        self._flat_references = self._references.flatten(start_dim=1)
        self._channel_squared_norms = self._references.square().sum(dim=(-2, -1))

    def search(
        self,
        queries: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        k: int,
        query_batch_size: int,
        reference_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact masked squared-L2 distances and reference indices."""

        self._validate_search_inputs(
            queries,
            keep_mask,
            k=k,
            query_batch_size=query_batch_size,
            reference_chunk_size=reference_chunk_size,
        )
        effective_k = min(k, self.reference_count)
        all_distances: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for query_start in range(0, queries.shape[0], query_batch_size):
            query_stop = min(query_start + query_batch_size, queries.shape[0])
            query_batch = queries[query_start:query_stop].to(
                device=self.device,
                dtype=torch.float32,
            )
            mask_batch = keep_mask[query_start:query_stop].to(
                device=self.device,
                dtype=torch.float32,
            )
            batch_distances, batch_indices = self._search_batch(
                query_batch,
                mask_batch,
                k=effective_k,
                reference_chunk_size=reference_chunk_size,
            )
            all_distances.append(batch_distances)
            all_indices.append(batch_indices)
        if not all_distances:
            return (
                torch.empty((0, effective_k), dtype=torch.float32, device=self.device),
                torch.empty((0, effective_k), dtype=torch.int64, device=self.device),
            )
        return torch.cat(all_distances), torch.cat(all_indices)

    def reconstruct_hidden_channels(
        self,
        queries: torch.Tensor,
        keep_mask: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Mean neighbor maps on hidden channels and preserve visible channels."""

        if tuple(queries.shape[1:]) != self.feature_shape:
            raise ValueError("queries do not match the indexed feature shape")
        if keep_mask.shape != (queries.shape[0], self.feature_shape[0], 1, 1):
            raise ValueError("keep_mask must have shape (B, C, 1, 1)")
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != queries.shape[0]:
            raise ValueError("neighbor_indices must have shape (B, k)")
        if neighbor_indices.shape[1] == 0:
            raise ValueError("neighbor_indices must contain at least one neighbor")
        if (
            int(neighbor_indices.min()) < 0
            or int(neighbor_indices.max()) >= self.reference_count
        ):
            raise ValueError("neighbor_indices contain an out-of-range reference")

        device_indices = neighbor_indices.to(device=self.device, dtype=torch.long)
        neighbor_mean = self._references[device_indices].mean(dim=1)
        device_queries = queries.to(device=self.device, dtype=torch.float32)
        device_mask = keep_mask.to(device=self.device, dtype=torch.float32)
        return device_queries * device_mask + neighbor_mean * (1.0 - device_mask)

    def _search_batch(
        self,
        queries: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        k: int,
        reference_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_mask = keep_mask.expand_as(queries)
        masked_queries = (queries * expanded_mask).flatten(start_dim=1)
        query_squared_norms = masked_queries.square().sum(dim=1, keepdim=True)
        channel_mask = keep_mask[:, :, 0, 0]
        best_distances: torch.Tensor | None = None
        best_indices: torch.Tensor | None = None

        for reference_start in range(0, self.reference_count, reference_chunk_size):
            reference_stop = min(
                reference_start + reference_chunk_size,
                self.reference_count,
            )
            reference_vectors = self._flat_references[reference_start:reference_stop]
            reference_squared_norms = self._channel_squared_norms[
                reference_start:reference_stop
            ]
            distances = (
                query_squared_norms
                + channel_mask @ reference_squared_norms.T
                - 2.0 * (masked_queries @ reference_vectors.T)
            ).clamp_min_(0.0)
            chunk_k = min(k, reference_stop - reference_start)
            chunk_distances, chunk_indices = torch.topk(
                distances,
                k=chunk_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            chunk_indices += reference_start
            if best_distances is None:
                best_distances = chunk_distances
                best_indices = chunk_indices
                continue
            assert best_indices is not None
            candidate_distances = torch.cat((best_distances, chunk_distances), dim=1)
            candidate_indices = torch.cat((best_indices, chunk_indices), dim=1)
            merge_k = min(k, candidate_distances.shape[1])
            best_distances, selected = torch.topk(
                candidate_distances,
                k=merge_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(1, selected)

        if best_distances is None or best_indices is None:
            raise RuntimeError("Masked k-NN search did not examine any references")
        return best_distances, best_indices

    def _validate_search_inputs(
        self,
        queries: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        k: int,
        query_batch_size: int,
        reference_chunk_size: int,
    ) -> None:
        if queries.ndim != 4 or tuple(queries.shape[1:]) != self.feature_shape:
            raise ValueError("queries must have shape (B, C, H, W) matching references")
        if keep_mask.shape != (queries.shape[0], self.feature_shape[0], 1, 1):
            raise ValueError("keep_mask must have shape (B, C, 1, 1)")
        if torch.any(keep_mask.sum(dim=(1, 2, 3)) <= 0):
            raise ValueError("every query must contain at least one unmasked channel")
        if k <= 0:
            raise ValueError("k must be positive")
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        if reference_chunk_size <= 0:
            raise ValueError("reference_chunk_size must be positive")


def _as_float32_array(tensor: torch.Tensor) -> np.ndarray:
    """Return a C-contiguous CPU float32 array accepted by FAISS."""

    return np.ascontiguousarray(tensor.detach().cpu().float().numpy())
