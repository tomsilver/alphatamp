"""PyTorch Dataset for bilevel planning skeleton reordering.

On-disk format: HDF5 with two-level CSR encoding for variable-length
operator sequences. Loadable with only h5py + numpy — no kinder/dill bootstrap.

Public API
----------
write_skeleton_dataset(output_path, dataset_dict, ...)
    Convert a build_dataset() output dict to HDF5.

SkeletonDataset(hdf5_path, *, preload=False)
    torch.utils.data.Dataset indexed over N problem instances.
    __getitem__(i) -> SkeletonItem  (per-instance (M,) tensors)
    Vocab-level data (op_sequences, skeleton_lengths, ...) are Dataset attributes.

skeleton_collate_fn(batch) -> SkeletonBatch
    Pass to DataLoader(collate_fn=skeleton_collate_fn).

HDF5 layout
-----------
/ (root)
├── attrs: format_version=1, N, M, created_at, source_description
├── meta/
│     op_type_vocab  (V_op,)   str
│     obj_vocab      (V_obj,)  str
│     type_vocab     (V_type,) str
├── vocab/
│     skeleton_lengths  (M,)   int16
│     op_offsets        (M+1,) int64   CSR: skeleton j owns ops[op_offsets[j]:op_offsets[j+1]]
│     op_type_ids       (T,)   int32   T = sum(L_j)
│     param_offsets     (T+1,) int64   CSR: op k owns params[param_offsets[k]:param_offsets[k+1]]
│     obj_ids_flat      (P,)   int32   P = total params
│     type_ids_flat     (P,)   int32
└── instances/
      seed_ids                   (N,)    int64
      applicability              (N, M)  float32
      success                    (N, M)  float32
      steps_completed_fraction   (N, M)  float32
      refinement_time            (N, M)  float32
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.utils.data

__all__ = [
    "OpSequenceTokens",
    "SkeletonItem",
    "SkeletonBatch",
    "SkeletonDataset",
    "skeleton_collate_fn",
    "write_skeleton_dataset",
]

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpSequenceTokens:
    """Tokenized representation of one skeleton's operator sequence.

    Attributes
    ----------
    op_type_ids:
        Integer op-type token for each operator step. Shape (L_j,), dtype int32.
    obj_ids:
        Grounded object ID for each parameter of each operator. Shape
        (L_j, max_params), dtype int32, padded with -1 for shorter operators.
    type_ids:
        Type ID for each parameter position (same shape as obj_ids), padded -1.
    length:
        Unpadded sequence length L_j.
    """

    op_type_ids: torch.Tensor  # int32 (L_j,)
    obj_ids: torch.Tensor  # int32 (L_j, max_params), -1 padded
    type_ids: torch.Tensor  # int32 (L_j, max_params), -1 padded
    length: int


@dataclass(frozen=True)
class SkeletonItem:
    """Single problem instance returned by SkeletonDataset.__getitem__(i).

    All tensor fields have shape (M,) where M is the vocabulary size.
    Vocabulary-level data (op_sequences, skeleton_lengths, vocabs) lives
    as Dataset *attributes* and is shared across all items.
    """

    seed_id: int
    applicability: torch.Tensor  # float32 (M,) binary {0,1}
    success: torch.Tensor  # float32 (M,) binary {0,1}
    steps_completed_fraction: torch.Tensor  # float32 (M,) ∈ [0,1]
    refinement_time: torch.Tensor  # float32 (M,) ≥ 0


@dataclass(frozen=True)
class SkeletonBatch:
    """Batched output from skeleton_collate_fn.

    All tensor fields have shape (B, M).
    """

    seed_ids: list[int]  # length B
    applicability: torch.Tensor  # float32 (B, M)
    success: torch.Tensor  # float32 (B, M)
    steps_completed_fraction: torch.Tensor  # float32 (B, M)
    refinement_time: torch.Tensor  # float32 (B, M)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_string_vocabs(
    op_sequence_vocab: list[Any],
) -> tuple[list[str], dict[str, int], list[str], dict[str, int], list[str], dict[str, int]]:
    """Collect unique op-type names, object names, and type names from the vocabulary.

    Accesses GroundOperator via duck-typing: op.name, op.parameters, p.name, p.type.name.
    Also works with any duck-typed substitute (e.g. _SyntheticOp).

    Returns
    -------
    (op_type_vocab, op_type_to_id, obj_vocab, obj_to_id, type_vocab, type_to_id)
    All vocab lists are sorted for determinism.
    """
    op_type_names: set[str] = set()
    obj_names: set[str] = set()
    type_names: set[str] = set()

    for seq in op_sequence_vocab:
        for op in seq:
            op_type_names.add(op.name)
            for p in op.parameters:
                obj_names.add(p.name)
                type_names.add(p.type.name)

    op_type_vocab = sorted(op_type_names)
    obj_vocab = sorted(obj_names)
    type_vocab = sorted(type_names)

    op_type_to_id = {n: i for i, n in enumerate(op_type_vocab)}
    obj_to_id = {n: i for i, n in enumerate(obj_vocab)}
    type_to_id = {n: i for i, n in enumerate(type_vocab)}

    return op_type_vocab, op_type_to_id, obj_vocab, obj_to_id, type_vocab, type_to_id


def _build_csr_arrays(
    op_sequence_vocab: list[Any],
    op_type_to_id: dict[str, int],
    obj_to_id: dict[str, int],
    type_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two-level CSR arrays encoding the ragged op sequences.

    Level 1: skeleton j → its operators
      op_offsets:  (M+1,) int64  skeleton j owns op_type_ids[op_offsets[j]:op_offsets[j+1]]
      op_type_ids: (T,)   int32  T = sum(L_j)

    Level 2: op k → its parameters
      param_offsets: (T+1,) int64  op k owns obj_ids_flat[param_offsets[k]:param_offsets[k+1]]
      obj_ids_flat:  (P,)   int32  P = total params
      type_ids_flat: (P,)   int32
    """
    M = len(op_sequence_vocab)

    op_offsets_list: list[int] = [0]
    op_type_ids_list: list[int] = []
    param_offsets_list: list[int] = [0]
    obj_ids_flat_list: list[int] = []
    type_ids_flat_list: list[int] = []

    for seq in op_sequence_vocab:
        for op in seq:
            op_type_ids_list.append(op_type_to_id[op.name])
            for p in op.parameters:
                obj_ids_flat_list.append(obj_to_id[p.name])
                type_ids_flat_list.append(type_to_id[p.type.name])
            param_offsets_list.append(len(obj_ids_flat_list))
        op_offsets_list.append(len(op_type_ids_list))

    op_offsets = np.array(op_offsets_list, dtype=np.int64)
    assert op_offsets.shape == (M + 1,)

    op_type_ids = np.array(op_type_ids_list, dtype=np.int32)
    param_offsets = np.array(param_offsets_list, dtype=np.int64)
    obj_ids_flat = np.array(obj_ids_flat_list, dtype=np.int32)
    type_ids_flat = np.array(type_ids_flat_list, dtype=np.int32)

    return op_offsets, op_type_ids, param_offsets, obj_ids_flat, type_ids_flat


def _decode_op_sequences(
    M: int,
    op_offsets: np.ndarray,
    op_type_ids: np.ndarray,
    param_offsets: np.ndarray,
    obj_ids_flat: np.ndarray,
    type_ids_flat: np.ndarray,
) -> list[OpSequenceTokens]:
    """Decode CSR arrays back into a list of OpSequenceTokens (one per skeleton).

    obj_ids and type_ids are padded with -1 to shape (L_j, max_params_j) where
    max_params_j is the max number of parameters across all operators in skeleton j.
    If L_j == 0 (empty skeleton), returns empty tensors of shape (0, 0).
    """
    result: list[OpSequenceTokens] = []

    for j in range(M):
        op_start = int(op_offsets[j])
        op_end = int(op_offsets[j + 1])
        L_j = op_end - op_start

        if L_j == 0:
            result.append(
                OpSequenceTokens(
                    op_type_ids=torch.zeros(0, dtype=torch.int32),
                    obj_ids=torch.zeros((0, 0), dtype=torch.int32),
                    type_ids=torch.zeros((0, 0), dtype=torch.int32),
                    length=0,
                )
            )
            continue

        op_type_ids_j = torch.from_numpy(op_type_ids[op_start:op_end].copy())

        # Find max arity for padding
        arities = [
            int(param_offsets[op_start + k + 1]) - int(param_offsets[op_start + k])
            for k in range(L_j)
        ]
        max_arity = max(arities) if arities else 0

        obj_ids_j = torch.full((L_j, max(max_arity, 1)), -1, dtype=torch.int32)
        type_ids_j = torch.full((L_j, max(max_arity, 1)), -1, dtype=torch.int32)

        for k in range(L_j):
            global_k = op_start + k
            p_start = int(param_offsets[global_k])
            p_end = int(param_offsets[global_k + 1])
            arity = p_end - p_start
            if arity > 0:
                obj_ids_j[k, :arity] = torch.from_numpy(obj_ids_flat[p_start:p_end].copy())
                type_ids_j[k, :arity] = torch.from_numpy(type_ids_flat[p_start:p_end].copy())

        result.append(
            OpSequenceTokens(
                op_type_ids=op_type_ids_j,
                obj_ids=obj_ids_j,
                type_ids=type_ids_j,
                length=L_j,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_skeleton_dataset(
    output_path: str | Path,
    dataset_dict: dict[str, Any],
    *,
    source_description: str = "",
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """Convert a build_dataset() output dict to an HDF5 file.

    Parameters
    ----------
    output_path:
        Path to the .h5 file to create. Overwritten if it already exists.
    dataset_dict:
        Dict as returned by EncoderApproach.build_dataset(). Required keys:
        seed_ids, op_sequence_vocab, applicability, success, refinement_time,
        steps_completed_fraction. Optional: skeleton_lengths (computed if absent).
    source_description:
        Free-text provenance string stored in root attrs.
    compression:
        HDF5 compression filter ('gzip', 'lzf', or None to disable).
    compression_opts:
        Compression level (1–9 for gzip, ignored for other filters).
    """
    op_sequence_vocab: list[Any] = dataset_dict["op_sequence_vocab"]
    seed_ids: list[int] = list(dataset_dict["seed_ids"])
    applicability: np.ndarray = np.asarray(dataset_dict["applicability"], dtype=np.float32)
    success: np.ndarray = np.asarray(dataset_dict["success"], dtype=np.float32)
    refinement_time: np.ndarray = np.asarray(dataset_dict["refinement_time"], dtype=np.float32)
    steps_completed_fraction: np.ndarray = np.asarray(
        dataset_dict["steps_completed_fraction"], dtype=np.float32
    )

    N = len(seed_ids)
    M = len(op_sequence_vocab)

    # Compute skeleton_lengths if not present
    if "skeleton_lengths" in dataset_dict:
        skeleton_lengths = np.asarray(dataset_dict["skeleton_lengths"], dtype=np.int16)
    else:
        skeleton_lengths = np.array([len(seq) for seq in op_sequence_vocab], dtype=np.int16)

    # Build string vocabularies and CSR arrays
    (
        op_type_vocab,
        op_type_to_id,
        obj_vocab,
        obj_to_id,
        type_vocab,
        type_to_id,
    ) = _build_string_vocabs(op_sequence_vocab)

    op_offsets, op_type_ids_arr, param_offsets, obj_ids_flat, type_ids_flat = _build_csr_arrays(
        op_sequence_vocab, op_type_to_id, obj_to_id, type_to_id
    )

    # Compress kwargs
    ckw: dict[str, Any] = {}
    if compression is not None:
        ckw["compression"] = compression
        if compression == "gzip":
            ckw["compression_opts"] = compression_opts

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype()

    with h5py.File(output_path, "w") as hf:
        # Root attrs
        hf.attrs["format_version"] = FORMAT_VERSION
        hf.attrs["N"] = N
        hf.attrs["M"] = M
        hf.attrs["created_at"] = datetime.datetime.utcnow().isoformat()
        hf.attrs["source_description"] = source_description

        # meta/ — string vocabularies
        mg = hf.create_group("meta")
        mg.create_dataset(
            "op_type_vocab",
            data=np.array(op_type_vocab, dtype=object),
            dtype=str_dtype,
        )
        mg.create_dataset(
            "obj_vocab",
            data=np.array(obj_vocab, dtype=object),
            dtype=str_dtype,
        )
        mg.create_dataset(
            "type_vocab",
            data=np.array(type_vocab, dtype=object),
            dtype=str_dtype,
        )

        # vocab/ — shared skeleton data
        vg = hf.create_group("vocab")
        vg.create_dataset("skeleton_lengths", data=skeleton_lengths, **ckw)
        vg.create_dataset("op_offsets", data=op_offsets, **ckw)
        vg.create_dataset("op_type_ids", data=op_type_ids_arr, **ckw)
        vg.create_dataset("param_offsets", data=param_offsets, **ckw)
        vg.create_dataset("obj_ids_flat", data=obj_ids_flat, **ckw)
        vg.create_dataset("type_ids_flat", data=type_ids_flat, **ckw)

        # instances/ — per-instance data
        ig = hf.create_group("instances")
        ig.create_dataset("seed_ids", data=np.array(seed_ids, dtype=np.int64), **ckw)
        ig.create_dataset("applicability", data=applicability, **ckw)
        ig.create_dataset("success", data=success, **ckw)
        ig.create_dataset(
            "steps_completed_fraction", data=steps_completed_fraction, **ckw
        )
        ig.create_dataset("refinement_time", data=refinement_time, **ckw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SkeletonDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for bilevel planning skeleton reordering.

    Each item corresponds to one problem instance (one row in the (N, M) matrices).

    Parameters
    ----------
    hdf5_path:
        Path to the .h5 file written by write_skeleton_dataset().
    preload:
        If True, load all instance arrays into RAM at init.
        Recommended for datasets that fit in memory and for DataLoader
        with num_workers > 0 (avoids per-item HDF5 I/O in worker processes).
        If False, arrays are read lazily from HDF5; the file handle is
        re-opened per worker process to support multiprocessing.

    Attributes
    ----------
    N : int
        Number of problem instances.
    M : int
        Vocabulary size.
    skeleton_lengths : torch.Tensor
        Shape (M,), dtype int32. Number of operators in each skeleton.
    op_sequences : list[OpSequenceTokens]
        Length M. Decoded token representations of each skeleton's op sequence.
    op_type_vocab : list[str]
        Operator type name strings indexed by op_type_id.
    obj_vocab : list[str]
        Object name strings indexed by obj_id.
    type_vocab : list[str]
        Type name strings indexed by type_id.
    """

    def __init__(self, hdf5_path: str | Path, *, preload: bool = False) -> None:
        self._hdf5_path = Path(hdf5_path)
        self._preload = preload
        self._file: h5py.File | None = None

        # Eagerly load all vocab-level data
        with h5py.File(self._hdf5_path, "r") as hf:
            self.N: int = int(hf.attrs["N"])
            self.M: int = int(hf.attrs["M"])

            # String vocabs
            self.op_type_vocab: list[str] = [
                s.decode() if isinstance(s, bytes) else s
                for s in hf["meta/op_type_vocab"][:]
            ]
            self.obj_vocab: list[str] = [
                s.decode() if isinstance(s, bytes) else s
                for s in hf["meta/obj_vocab"][:]
            ]
            self.type_vocab: list[str] = [
                s.decode() if isinstance(s, bytes) else s
                for s in hf["meta/type_vocab"][:]
            ]

            # CSR arrays for op sequences
            skeleton_lengths_np = hf["vocab/skeleton_lengths"][:]
            op_offsets = hf["vocab/op_offsets"][:]
            op_type_ids_arr = hf["vocab/op_type_ids"][:]
            param_offsets = hf["vocab/param_offsets"][:]
            obj_ids_flat = hf["vocab/obj_ids_flat"][:]
            type_ids_flat = hf["vocab/type_ids_flat"][:]

            self.skeleton_lengths: torch.Tensor = torch.from_numpy(
                skeleton_lengths_np.astype(np.int32)
            )

            self.op_sequences: list[OpSequenceTokens] = _decode_op_sequences(
                self.M,
                op_offsets,
                op_type_ids_arr,
                param_offsets,
                obj_ids_flat,
                type_ids_flat,
            )

            # Optional preload of instance data
            if preload:
                self._seed_ids = torch.from_numpy(hf["instances/seed_ids"][:].astype(np.int64))
                self._applicability = torch.from_numpy(hf["instances/applicability"][:])
                self._success = torch.from_numpy(hf["instances/success"][:])
                self._steps = torch.from_numpy(hf["instances/steps_completed_fraction"][:])
                self._time = torch.from_numpy(hf["instances/refinement_time"][:])
            else:
                self._seed_ids = None
                self._applicability = None
                self._success = None
                self._steps = None
                self._time = None

    def _get_file(self) -> h5py.File:
        """Return an open HDF5 file handle, opening lazily (safe per-process)."""
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index: int) -> SkeletonItem:
        if self._preload:
            return SkeletonItem(
                seed_id=int(self._seed_ids[index].item()),
                applicability=self._applicability[index],
                success=self._success[index],
                steps_completed_fraction=self._steps[index],
                refinement_time=self._time[index],
            )

        hf = self._get_file()
        return SkeletonItem(
            seed_id=int(hf["instances/seed_ids"][index]),
            applicability=torch.from_numpy(hf["instances/applicability"][index]),
            success=torch.from_numpy(hf["instances/success"][index]),
            steps_completed_fraction=torch.from_numpy(
                hf["instances/steps_completed_fraction"][index]
            ),
            refinement_time=torch.from_numpy(hf["instances/refinement_time"][index]),
        )

    def __getstate__(self) -> dict[str, Any]:
        """Close HDF5 handle before pickling (for DataLoader multiprocessing)."""
        state = self.__dict__.copy()
        if state["_file"] is not None:
            state["_file"].close()
        state["_file"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def skeleton_collate_fn(batch: list[SkeletonItem]) -> SkeletonBatch:
    """Collate a list of SkeletonItems into a SkeletonBatch.

    Pass to torch.utils.data.DataLoader as collate_fn=skeleton_collate_fn.
    All tensor fields are stacked along dim=0 to produce (B, M) tensors.
    seed_ids is kept as a Python list (identifier, not computation input).
    """
    return SkeletonBatch(
        seed_ids=[item.seed_id for item in batch],
        applicability=torch.stack([item.applicability for item in batch]),
        success=torch.stack([item.success for item in batch]),
        steps_completed_fraction=torch.stack(
            [item.steps_completed_fraction for item in batch]
        ),
        refinement_time=torch.stack([item.refinement_time for item in batch]),
    )
