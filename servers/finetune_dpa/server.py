from typing import Optional, List, Union, Dict
from pathlib import Path
import logging
import os
import glob
import subprocess
import numpy as np
import dpdata
from dp.agent.server import CalculationMCPServer
from typing_extensions import TypedDict
import csv

import random
import shutil

from pymatgen.core import Composition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize MCP server
mcp = CalculationMCPServer(
    "DPAFinetune",
    host="0.0.0.0",
    port=50003
)


#=========Function to revise dpa3 input.json=======
import json
from typing import Any, Dict, Mapping, Optional, Tuple, Literal

# ------------- helpers -------------

def _set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

def _get_section(d: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    cur = d
    for k in path:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur

def _merge_whitelisted(
    data: Dict[str, Any],
    base_path: Tuple[str, ...],
    incoming: Optional[Mapping[str, Any]],
    allowed_keys: Optional[set] = None,
) -> None:
    if not incoming:
        return
    section = _get_section(data, base_path)
    if allowed_keys is None:
        # merge all keys
        for k, v in incoming.items():
            section[k] = v
    else:
        for k, v in incoming.items():
            if k in allowed_keys:
                section[k] = v

# ------------- allowed key sets -------------

DPA3_REPFLOW_KEYS = {
    "n_dim","e_dim","a_dim","nlayers",
    "e_rcut","e_rcut_smth","e_sel",
    "a_rcut","a_rcut_smth","a_sel",
    "axis_neuron","fix_stat_std",
    "a_compress_rate","a_compress_e_rate","a_compress_use_split",
    "update_angle","smooth_edge_update","use_dynamic_sel",
    "sel_reduce_factor","use_exp_switch",
    "update_style","update_residual","update_residual_init",
}

DPA3_TOP_KEYS = {"activation_function","use_tebd_bias","precision","concat_output_tebd"}

DPA3_FITTING_NET_KEYS = {"neuron","dim_case_embd","resnet_dt","precision","activation_function","seed"}

DPA2_REPINIT_KEYS = {
    "tebd_dim","rcut","rcut_smth","nsel","neuron",
    "axis_neuron","three_body_neuron","activation_function",
    "three_body_sel","three_body_rcut","three_body_rcut_smth","use_three_body",
}

DPA2_REPFORMER_KEYS = {
    "rcut","rcut_smth","nsel","nlayers",
    "g1_dim","g2_dim","attn2_hidden","attn2_nhead","attn1_hidden","attn1_nhead",
    "axis_neuron",
    "update_h2","update_g1_has_conv","update_g1_has_grrg","update_g1_has_drrd","update_g1_has_attn",
    "update_g2_has_g1g1","update_g2_has_attn",
    "update_style","update_residual","update_residual_init",
    "attn2_has_gate","use_sqrt_nnei","g1_out_conv","g1_out_mlp",
    "activation_function",
}

DPA2_FITTING_NET_KEYS = {"neuron","activation_function","resnet_dt","precision","dim_case_embd","seed","_comment"}

DPA2_TOP_KEYS = {"use_tebd_bias","precision","add_tebd_to_repinit_out"}

# ------------- main -------------

def update_dpa_train_json(
    template_path: str,
    version: Literal["dpa2","dpa3"],
    # global switch
    default_model: bool = True,
    # -------- common knobs (always allowed) --------
    lr_type: Optional[str] = None,
    decay_steps: Optional[int] = None,
    start_lr: Optional[float] = None,
    stop_lr: Optional[float] = None,
    loss_type: Optional[str] = None,
    start_pref_e: Optional[float] = None,
    limit_pref_e: Optional[float] = None,
    start_pref_f: Optional[float] = None,
    limit_pref_f: Optional[float] = None,
    start_pref_v: Optional[float] = None,
    limit_pref_v: Optional[float] = None,
    numb_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    # -------- DPA-3 extras (used only when default_model=False) --------
    dpa3_repflow: Optional[Mapping[str, Any]] = None,
    dpa3_top: Optional[Mapping[str, Any]] = None,        # keys in DPA3_TOP_KEYS
    dpa3_fitting_net: Optional[Mapping[str, Any]] = None, # keys in DPA3_FITTING_NET_KEYS
    # -------- DPA-2 extras (used only when default_model=False) --------
    dpa2_repinit: Optional[Mapping[str, Any]] = None,
    dpa2_repformer: Optional[Mapping[str, Any]] = None,
    dpa2_fitting_net: Optional[Mapping[str, Any]] = None,
    dpa2_top: Optional[Mapping[str, Any]] = None,        # keys in DPA2_TOP_KEYS
    # output
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads input JSON (DPA-2 or DPA-3), applies updates, writes output, returns dict.

    Behavior:
      - default_model=True  -> only LR/Loss/Schedule will be updated.
      - default_model=False -> also allow the exact section/keys per your spec:
            DPA-3: model.descriptor.repflow, model.{activation_function,use_tebd_bias,precision,concat_output_tebd},
                   model.fitting_net (selected keys)
            DPA-2: model.descriptor.repinit, model.descriptor.repformer,
                   model.fitting_net (selected keys), top {use_tebd_bias,precision,add_tebd_to_repinit_out}
    """
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- always-allowed (both versions) ----
    if lr_type is not None: _set(data, ("learning_rate","type"), lr_type)
    if decay_steps is not None: _set(data, ("learning_rate","decay_steps"), decay_steps)
    if start_lr is not None: _set(data, ("learning_rate","start_lr"), start_lr)
    if stop_lr is not None: _set(data, ("learning_rate","stop_lr"), stop_lr)

    if loss_type is not None: _set(data, ("loss","type"), loss_type)
    if start_pref_e is not None: _set(data, ("loss","start_pref_e"), start_pref_e)
    if limit_pref_e is not None: _set(data, ("loss","limit_pref_e"), limit_pref_e)
    if start_pref_f is not None: _set(data, ("loss","start_pref_f"), start_pref_f)
    if limit_pref_f is not None: _set(data, ("loss","limit_pref_f"), limit_pref_f)
    if start_pref_v is not None: _set(data, ("loss","start_pref_v"), start_pref_v)
    if limit_pref_v is not None: _set(data, ("loss","limit_pref_v"), limit_pref_v)

    if numb_steps is not None: _set(data, ("training","numb_steps"), numb_steps)
    if warmup_steps is not None: _set(data, ("training","warmup_steps"), warmup_steps)

    # ---- extended edits only when default_model=False ----
    if not default_model:
        if version == "dpa3":
            # repflow
            _merge_whitelisted(
                data, ("model","descriptor","repflow"), dpa3_repflow, DPA3_REPFLOW_KEYS
            )
            # top-level model extras
            if dpa3_top:
                for k, v in dpa3_top.items():
                    if k in DPA3_TOP_KEYS:
                        _set(data, ("model", k), v)
            # fitting_net
            _merge_whitelisted(
                data, ("model","fitting_net"), dpa3_fitting_net, DPA3_FITTING_NET_KEYS
            )

        elif version == "dpa2":
            _merge_whitelisted(
                data, ("model","descriptor","repinit"), dpa2_repinit, DPA2_REPINIT_KEYS
            )
            _merge_whitelisted(
                data, ("model","descriptor","repformer"), dpa2_repformer, DPA2_REPFORMER_KEYS
            )
            _merge_whitelisted(
                data, ("model","fitting_net"), dpa2_fitting_net, DPA2_FITTING_NET_KEYS
            )
            if dpa2_top:
                for k, v in dpa2_top.items():
                    if k in DPA2_TOP_KEYS:
                        _set(data, (k,), v)
        else:
            raise ValueError("version must be 'dpa2' or 'dpa3'")

    # ---- write out ----
    if output_path is None:
        output_path = "train_dpa3.json" if version == "dpa3" else "train_dpa2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

def split_train_valid(
    input_path: str,
    valid_ratio: float = 0.10
    ):
  """
    Split dpdata into ./data_train and ./data_valid with given valid_ratio for further model fine tune operations.
    Args:
       input_path (str): Path to the whol dpdata (can be a directory or a compressed archive)
       valid_ratio (float): validation data set ratio, and the default setting is 0.10
  """
  import os, shutil, tempfile, random
  import dpdata

  def _is_archive(p: str) -> bool:
      pl = p.lower()
      return pl.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))

  print(f"input_path={input_path}")

  # load MultiSystems (auto-extract if needed)
  if _is_archive(input_path):
      print(f"_is_archive(input_path)= {_is_archive(input_path)}")
      with tempfile.TemporaryDirectory(prefix="dpdata_extract_") as tmpd:
          shutil.unpack_archive(input_path, tmpd)
          # if a single top-level directory exists after extraction, use it
          entries = [os.path.join(tmpd, e) for e in os.listdir(tmpd) if not e.startswith("__MACOSX")]
          if len(entries) == 1 and os.path.isdir(entries[0]):
              root = entries[0]
          else:
              root = tmpd
          ms = dpdata.MultiSystems()
          p = Path(root)
          has_real = next(p.rglob("real_atom_types.npy"), None) is not None
          fmt = "deepmd/npy/mixed" if has_real else "deepmd/npy"
          if has_real:
             ms = dpdata.MultiSystems().load_systems_from_file(root,fmt=fmt)
          else: 
             parent_dirs = sorted({p.parent.resolve() for p in Path(root).rglob("type.raw")})
             for d in parent_dirs:
                 try:
                     sys = dpdata.LabeledSystem(str(d), fmt="deepmd/npy")
                 except Exception as e:
                     print(f"[WARN] Skipping {d}: {e}")
                     continue
                 ms.append(sys)



          # collect all systems
          systems = []
          count = 0
          for key, sys_list in ms.systems.items():
              for s in sys_list:
                  systems.append((key, s))
                  count += 1

          print(count)

          # random select train and validation set based on given ratio
          random.seed(123)
          random.shuffle(systems)
          split_idx = int(valid_ratio * len(systems))
          valid_set = systems[:split_idx]
          train_set = systems[split_idx:]

          ms_train = dpdata.MultiSystems()
          ms_valid = dpdata.MultiSystems()

          for k, s in train_set:
              ms_train.append(s)  # preserve label

          for k, s in valid_set:
              ms_valid.append(s)  # preserve label

          ms_valid.to_deepmd_npy_mixed("./data_valid")
          ms_train.to_deepmd_npy_mixed("./data_train")
          return  # important: exit before tmp dir is cleaned up
  else:
      # non-archive path
      ms = dpdata.MultiSystems()
      p = Path(input_path)
      has_real = next(p.rglob("real_atom_types.npy"), None) is not None
      fmt = "deepmd/npy/mixed" if has_real else "deepmd/npy"
      ms = dpdata.MultiSystems().load_systems_from_file(input_path,fmt=fmt)
      
      # collect all systems
      systems = []
      count = 0
      for key, sys_list in ms.systems.items():
          for s in sys_list:
              systems.append((key, s))
              count += 1
              print(count)
      
      # random select train and validation set based on given ratio
      random.seed(123)
      random.shuffle(systems)
      split_idx = int(valid_ratio * len(systems))
      valid_set = systems[:split_idx]
      train_set = systems[split_idx:]
      
      ms_train = dpdata.MultiSystems()
      ms_valid = dpdata.MultiSystems()
      
      for k, s in train_set:
          ms_train.append(s, k)  # preserve label
      
      for k, s in valid_set:
          ms_valid.append(s, k)  # preserve label
      
      ms_valid.to_deepmd_npy_mixed("./data_valid")
      ms_train.to_deepmd_npy_mixed("./data_train")

import logging
import re
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    """Base enum with case-insensitive matching."""

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            v = value.strip().lower()
            for item in cls:
                if item.value.lower() == v:
                    return item
        return None


class ModelType(StrEnum):
    DPA2 = "dpa2"
    DPA3 = "dpa3"


class LRType(StrEnum):
    EXP = "exp"

    # Must be a dict[str, str]
    _ALIASES = {
        "exponential": "exp",
        "expdecay": "exp",
        "exponential_decay": "exp",
        "cosine": "exp",  # optional: treat cosine as exp fallback
    }

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            v = value.strip().lower()

            aliases = getattr(cls, "_ALIASES", None)
            # Robust guard: if aliases is not a dict, ignore it
            if isinstance(aliases, dict) and v in aliases:
                mapped = aliases[v]
                try:
                    return cls(mapped)
                except ValueError:
                    return None

        return super()._missing_(value)


class LossType(StrEnum):
    ENER = "ener"
    ENER_SPIN = "ener_spin"
    DOS = "dos"
    PROPERTY = "property"
    TENSOR = "tensor"


class PrecisionType(StrEnum):
    FLOAT32 = "float32"


class ActivationType(StrEnum):
    TANH = "tanh"
    SILUT_3 = "silut:3.0"


class UpdateStyleType(StrEnum):
    RES_RESIDUAL = "res_residual"


class ResidualInitType(StrEnum):
    CONST = "const"
    NORM = "norm"


def coerce_enum(
    enum_cls: type[StrEnum],
    raw_value: Any,
    default: StrEnum,
    param_name: str,
) -> StrEnum:
    if raw_value is None or raw_value == "":
        return default

    if isinstance(raw_value, enum_cls):
        return raw_value

    if isinstance(raw_value, str):
        try:
            v = enum_cls(raw_value)
            if v is not None:
                return v
        except ValueError:
            pass

    logging.warning(
        "Invalid value for %s=%r, falling back to default %r",
        param_name,
        raw_value,
        default.value,
    )
    return default


def coerce_float(raw_value: Any, default: float, param_name: str) -> float:
    if raw_value is None:
        return default

    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    if isinstance(raw_value, str):
        try:
            return float(raw_value.strip())
        except ValueError:
            pass

    logging.warning(
        "Invalid float for %s=%r, falling back to default %r",
        param_name,
        raw_value,
        default,
    )
    return default


def coerce_int(raw_value: Any, default: int, param_name: str) -> int:
    """
    Enhanced int parser for LLM/agent-produced values.
    Examples that will all parse to 3000:
      "3000", " 3000 ", "3,000", "3000 steps",
      "train 3000 steps", "numb_steps = 3000"
    """
    if raw_value is None:
        return default

    if isinstance(raw_value, int):
        return raw_value

    if isinstance(raw_value, float):
        return int(round(raw_value))

    if isinstance(raw_value, str):
        cleaned = raw_value.replace(",", "")
        m = re.search(r"[-+]?\d+", cleaned)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass

    logging.warning(
        "Invalid int for %s=%r, falling back to default %r",
        param_name,
        raw_value,
        default,
    )
    return default
def coerce_positive_float(raw_value: Any, default: float, param_name: str, eps: float = 1e-12) -> float:
    """
    Like coerce_float(), but additionally enforces value > 0 (strictly).
    If value is <= eps, fall back to default.
    """
    v = coerce_float(raw_value, default, param_name)
    if v <= eps:
        logging.warning(
            "Invalid (non-positive) value for %s=%r, falling back to default %r",
            param_name,
            raw_value,
            default,
        )
        return default
    return v

def is_user_provided_model_path(p: Optional[Path]) -> bool:
    """
    Treat None / "" / "none" / "null" as NOT provided.
    Only return True when user explicitly provides a real path/uri.
    """
    if p is None:
        return False
    s = str(p).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return False
    return True

import inspect

def reset_params_to_signature_defaults(func, local_vars: dict, prefixes: tuple[str, ...]) -> None:
    """
    For any parameter whose name starts with one of `prefixes`,
    reset it to the default value in function signature.
    """
    sig = inspect.signature(func)
    for name, p in sig.parameters.items():
        if any(name.startswith(pref) for pref in prefixes):
            if p.default is not inspect._empty:
                local_vars[name] = p.default

class Finetuned_model(TypedDict):
    results: Path
    message: str

@mcp.tool()
def finetune_dpa_model(
    input_path: Path,
    model_type: str = "dpa3",
    model_path: Optional[Path] = None,
    valid_ratio: float = 0.1,
    # learning rate
    lr_type: str = "exp",
    decay_steps: int = 5000,
    start_lr: float = 0.001,
    stop_lr: float = 3e-5,
    # loss
    loss_type: str = "ener",
    start_pref_e: float = 0.2,
    limit_pref_e: float = 20.0,
    start_pref_f: float = 100.0,
    limit_pref_f: float = 60.0,
    start_pref_v: float = 0.02,
    limit_pref_v: float = 1.0,
    # training schedule
    numb_steps: int = 3000,
    warmup_steps: int = 2000,
    # ---------- dpa3: repflow ----------
    dpa3_repflow_n_dim: int = 128,
    dpa3_repflow_e_dim: int = 64,
    dpa3_repflow_a_dim: int = 32,
    dpa3_repflow_nlayers: int = 16,
    dpa3_repflow_e_rcut: float = 6.0,
    dpa3_repflow_e_rcut_smth: float = 3.5,
    dpa3_repflow_e_sel: int = 1200,
    dpa3_repflow_a_rcut: float = 4.0,
    dpa3_repflow_a_rcut_smth: float = 3.5,
    dpa3_repflow_a_sel: int = 300,
    dpa3_repflow_axis_neuron: int = 4,
    dpa3_repflow_fix_stat_std: float = 0.3,
    dpa3_repflow_a_compress_rate: int = 1,
    dpa3_repflow_a_compress_e_rate: int = 2,
    dpa3_repflow_a_compress_use_split: bool = True,
    dpa3_repflow_update_angle: bool = True,
    dpa3_repflow_smooth_edge_update: bool = True,
    dpa3_repflow_use_dynamic_sel: bool = True,
    dpa3_repflow_sel_reduce_factor: float = 10.0,
    dpa3_repflow_use_exp_switch: bool = True,
    dpa3_repflow_update_style: str = "res_residual",
    dpa3_repflow_update_residual: float = 0.1,
    dpa3_repflow_update_residual_init: str = "const",
    # ---------- dpa3: top ----------
    dpa3_top_activation_function: str = "silut:3.0",
    dpa3_top_use_tebd_bias: bool = False,
    dpa3_top_precision: str = "float32",
    dpa3_top_concat_output_tebd: bool = False,
    # ---------- dpa3: fitting_net ----------
    dpa3_fitting_net_neuron: Optional[List[int]] = None,  # e.g. [240, 240, 240]
    dpa3_fitting_net_dim_case_embd: int = 31,
    dpa3_fitting_net_resnet_dt: bool = True,
    dpa3_fitting_net_precision: str = "float32",
    dpa3_fitting_net_activation_function: str = "tanh",
    dpa3_fitting_net_seed: int = 1,
    # ---------- dpa2: repinit ----------
    dpa2_repinit_tebd_dim: int = 80,
    dpa2_repinit_rcut: float = 4.0,
    dpa2_repinit_rcut_smth: float = 3.5,
    dpa2_repinit_nsel: int = 40,
    dpa2_repinit_neuron: int = 32,
    dpa2_repinit_axis_neuron: int = 4,
    dpa2_repinit_three_body_neuron: int = 32,
    dpa2_repinit_activation_function: str = "tanh",
    dpa2_repinit_three_body_sel: int = 40,
    dpa2_repinit_three_body_rcut: float = 4.0,
    dpa2_repinit_three_body_rcut_smt: float = 3.5,
    dpa2_repinit_use_three_body: bool = True,
    # ---------- dpa2: repformer ----------
    dpa2_repformer_rcut: float = 4.0,
    dpa2_repformer_rcut_smth: float = 3.5,
    dpa2_repformer_nsel: int = 40,
    dpa2_repformer_nlayers: int = 6,
    dpa2_repformer_g1_dim: int = 384,
    dpa2_repformer_g2_dim: int = 96,
    dpa2_repformer_attn2_hidden: int = 24,
    dpa2_repformer_attn2_nhead: int = 4,
    dpa2_repformer_attn1_hidden: int = 128,
    dpa2_repformer_attn1_nhead: int = 4,
    dpa2_repformer_axis_neuron: int = 4,
    dpa2_repformer_update_h2: bool = False,
    dpa2_repformer_update_g1_has_conv: bool = True,
    dpa2_repformer_update_g1_has_grrg: bool = True,
    dpa2_repformer_update_g1_has_drrd: bool = True,
    dpa2_repformer_update_g1_has_attn: bool = False,
    dpa2_repformer_update_g2_has_g1g1: bool = False,
    dpa2_repformer_update_g2_has_attn: bool = True,
    dpa2_repformer_update_style: str = "res_residual",
    dpa2_repformer_update_residual: float = 0.01,
    dpa2_repformer_update_residual_init: str = "norm",
    dpa2_repformer_attn2_has_gate: bool = True,
    dpa2_repformer_use_sqrt_nnei: bool = True,
    dpa2_repformer_g1_out_conv: bool = True,
    dpa2_repformer_g1_out_mlp: bool = True,
    dpa2_repformer_activation_function: str = "tanh",
    # ---------- dpa2: fitting_net ----------
    dpa2_fitting_net_neuron: Optional[List[int]] = None,  # e.g. [240, 240, 240]
    dpa2_fitting_net_activation_function: str = "tanh",
    dpa2_fitting_net_resnet_dt: bool = True,
    dpa2_fitting_net_precision: str = "float32",
    dpa2_fitting_net_dim_case_embd: int = 37,
    dpa2_fitting_net_seed: int = 1,
    # ---------- dpa2: top ----------
    dpa2_top_use_tebd_bias: bool = False,
    dpa2_top_precision: str = "float32",
    dpa2_top_add_tebd_to_repinit_out: bool = False,
    # ---- NEW: add at the very end ----
    user_text: Optional[str] = None,
    training_steps: Optional[str] = None,
    total_steps: Optional[str] = None,
    steps: Optional[str] = None,
) -> Finetuned_model:

    """
    Finetune the DPA3 model based on user requirment. If use do not provide model, please just use our dpa2 and dpa3 default model to be fined tuned.
    Args:
        --input_path (Path): the path to fine tune model data
        --valid_ratio (float): the ratio to split data into train and validation sets
        --model_type (str): the version of DPA model, could be dpa2 or dpa3.
        --model_path (Path): the path to model which need to be finetuned further. If user did not provide model, just use dpa2 or dpa3 default model
        --lr_type (str): The type of the learning rate
        --decay_steps (int): The learning rate is decaying every this number of training steps.
        --start_lr float: The learning rate at the start of the training
        --stop_lr (float):The desired learning rate at the end of the training. 
        --loss_type (str): The type of the loss. possible choices: ener, ener_spin, dos, property, tensor
        --start_pref_e (float):The prefactor of energy loss at the start of the training. 
        --limit_pref_e (float): The prefactor of energy loss at the limit of the training. Should be larger than or equal to 0.
        --start_pref_f (float): The prefactor of force loss at the start of the training. Should be larger than or equal to 0
        --limit_pref_f (float): The prefactor of force loss at the limit of the training. Should be larger than or equal to 0.
        --start_pref_v (float): The prefactor of virial loss at the start of the training. Should be larger than or equal to 0.
        --limit_pref_v (float): The prefactor of virial loss at the limit of the training. Should be larger than or equal to 0.
        --numb_steps (int): Number of training batch. 
        --warmup_steps (int): The number of steps for learning rate warmup. 
        --dpa3_repflow (Mapping[str,Any]): For dpa3 model type, if user provide with model, it could be defined by user about repflow part
        --dpa3_top (Mapping[str,Any]): For dpa3 model type, if user provide with model, it could be defined by user about top part
        --dpa3_fitting_net (Mapping[str,Any]): For dpa3 model type, if user provide with model, it could be defined by user about fitting_net part
        --dpa2_repinit (Mapping[str,Any]): For dpa2 model type, if user provide with model, it could be defined by user about repinit part
        --dpa2_repformer (Mapping[str,Any]): For dpa2 model type, if user provide with model, it could be defined by user about repformer part
        --dpa2_top (Mapping[str,Any]): For dpa2 model type, if user provide with model, it could be defined by user about top part
        --dpa2_fitting_net (Mapping[str,Any]): For dpa2 model type, if user provide with model, it could be defined by user about fitting_net part
    Return:
        Finetuned_model with keys:
        --results (Path): Path to finetuned model
        --message: Message about operation results

    """
    import re
    from typing import Optional
    
    def extract_training_steps(user_text: str) -> Optional[int]:
        if not user_text:
            return None
        text = user_text.replace(",", "")
    
        # High-confidence patterns (prefer these)
        patterns = [
            r"(?:training\s*)?(?:total\s*)?steps\s*[:=]?\s*(\d+)",
            r"(?:num(?:b)?_steps|max_steps|steps)\s*[:=]?\s*(\d+)",
            r"(?:训练总步数|总步数|训练步数|训练步)\s*[:=：]?\s*(\d+)",
            r"(?:跑|训练)\s*(\d+)\s*(?:步|steps?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    
        return None

    # ====== Normalize and debug-log all important hyper-parameters ======

    # Remember raw values for debugging
    raw_model_type = model_type
    raw_lr_type = lr_type
    raw_loss_type = loss_type
    raw_numb_steps = numb_steps
    raw_warmup_steps = warmup_steps

    model_type = coerce_enum(
        ModelType, model_type, ModelType.DPA3, "model_type"
    ).value

    lr_type = coerce_enum(
        LRType, lr_type, LRType.EXP, "lr_type"
    ).value

    loss_type = coerce_enum(
        LossType, loss_type, LossType.ENER, "loss_type"
    ).value

    # dpa3 repflow
    dpa3_repflow_update_style = coerce_enum(
        UpdateStyleType,
        dpa3_repflow_update_style,
        UpdateStyleType.RES_RESIDUAL,
        "dpa3_repflow_update_style",
    ).value

    dpa3_repflow_update_residual_init = coerce_enum(
        ResidualInitType,
        dpa3_repflow_update_residual_init,
        ResidualInitType.CONST,
        "dpa3_repflow_update_residual_init",
    ).value

    # dpa3 top / fitting
    dpa3_top_activation_function = coerce_enum(
        ActivationType,
        dpa3_top_activation_function,
        ActivationType.SILUT_3,
        "dpa3_top_activation_function",
    ).value

    dpa3_top_precision = coerce_enum(
        PrecisionType,
        dpa3_top_precision,
        PrecisionType.FLOAT32,
        "dpa3_top_precision",
    ).value

    dpa3_fitting_net_precision = coerce_enum(
        PrecisionType,
        dpa3_fitting_net_precision,
        PrecisionType.FLOAT32,
        "dpa3_fitting_net_precision",
    ).value

    dpa3_fitting_net_activation_function = coerce_enum(
        ActivationType,
        dpa3_fitting_net_activation_function,
        ActivationType.TANH,
        "dpa3_fitting_net_activation_function",
    ).value

    # dpa2 repinit / repformer / fitting / top
    dpa2_repinit_activation_function = coerce_enum(
        ActivationType,
        dpa2_repinit_activation_function,
        ActivationType.TANH,
        "dpa2_repinit_activation_function",
    ).value

    dpa2_repformer_update_style = coerce_enum(
        UpdateStyleType,
        dpa2_repformer_update_style,
        UpdateStyleType.RES_RESIDUAL,
        "dpa2_repformer_update_style",
    ).value

    dpa2_repformer_update_residual_init = coerce_enum(
        ResidualInitType,
        dpa2_repformer_update_residual_init,
        ResidualInitType.NORM,
        "dpa2_repformer_update_residual_init",
    ).value

    dpa2_repformer_activation_function = coerce_enum(
        ActivationType,
        dpa2_repformer_activation_function,
        ActivationType.TANH,
        "dpa2_repformer_activation_function",
    ).value

    dpa2_fitting_net_activation_function = coerce_enum(
        ActivationType,
        dpa2_fitting_net_activation_function,
        ActivationType.TANH,
        "dpa2_fitting_net_activation_function",
    ).value

    dpa2_fitting_net_precision = coerce_enum(
        PrecisionType,
        dpa2_fitting_net_precision,
        PrecisionType.FLOAT32,
        "dpa2_fitting_net_precision",
    ).value

    dpa2_top_precision = coerce_enum(
        PrecisionType,
        dpa2_top_precision,
        PrecisionType.FLOAT32,
        "dpa2_top_precision",
    ).value

    # numeric hyper-parameters
    valid_ratio = coerce_float(valid_ratio, 0.1, "valid_ratio")
    decay_steps = coerce_int(decay_steps, 5000, "decay_steps")
    start_lr = coerce_float(start_lr, 0.001, "start_lr")
    stop_lr = coerce_float(stop_lr, 3e-5, "stop_lr")

    # Prefactors MUST be > 0; agent sometimes injects 0 which kills gradients.
    start_pref_e = coerce_positive_float(start_pref_e, 0.2, "start_pref_e")
    start_pref_f = coerce_positive_float(start_pref_f, 100.0, "start_pref_f")
    start_pref_v = coerce_positive_float(start_pref_v, 0.02, "start_pref_v")
    
    # Limits should also be > 0 (and usually >= start)
    limit_pref_e = coerce_positive_float(limit_pref_e, 20.0, "limit_pref_e")
    limit_pref_f = coerce_positive_float(limit_pref_f, 60.0, "limit_pref_f")
    limit_pref_v = coerce_positive_float(limit_pref_v, 1.0, "limit_pref_v")
    
    # Ensure limit >= start (safety)
    limit_pref_e = max(limit_pref_e, start_pref_e)
    limit_pref_f = max(limit_pref_f, start_pref_f)
    limit_pref_v = max(limit_pref_v, start_pref_v)

    if (start_pref_e <= 1e-12) and (start_pref_f <= 1e-12) and (start_pref_v <= 1e-12):
        logging.warning("All start_pref_* are ~0. Resetting to safe defaults.")
        start_pref_e, start_pref_f, start_pref_v = 0.2, 100.0, 0.02


    # ====== Solve "could not read steps" problem ======
    DEFAULT_NUMB_STEPS = 3000
    DEFAULT_WARMUP_STEPS = 2000
    
    raw_numb_steps = numb_steps
    raw_warmup_steps = warmup_steps
    
    # Normalize both (handles "2000 steps", "3,000", etc.)
    numb_steps = coerce_int(numb_steps, DEFAULT_NUMB_STEPS, "numb_steps")
    warmup_steps = coerce_int(warmup_steps, DEFAULT_WARMUP_STEPS, "warmup_steps")
    
    # If user_text contains an explicit total steps, let it override (highest priority)
    parsed_total = extract_training_steps(user_text) if user_text else None
    if parsed_total is not None:
        if parsed_total != numb_steps:
            logging.info(
                "DEBUG steps override: user_text specifies total_steps=%d, overriding agent numb_steps=%d",
                parsed_total,
                numb_steps,
            )
        numb_steps = parsed_total
    
    # --------- Critical safety clamp for warmup_steps ----------
    # DeepMD requires: warmup_steps < numb_steps OR warmup_steps == 0
    if numb_steps <= 0:
        # Defensive: never allow non-positive total steps
        logging.warning("Invalid numb_steps=%d; forcing to 1 and warmup_steps=0", numb_steps)
        numb_steps = 1
        warmup_steps = 0
    else:
        if warmup_steps < 0:
            logging.warning("Invalid warmup_steps=%d; forcing to 0", warmup_steps)
            warmup_steps = 0
    
        if warmup_steps >= numb_steps:
            # Strategy: keep warmup small and valid
            # Option A (recommended): set warmup to 0 when total steps is small
            new_warmup = 0 if numb_steps < 50 else max(1, int(0.1 * numb_steps))
            # Ensure strictly less than numb_steps
            new_warmup = min(new_warmup, numb_steps - 1)
            logging.warning(
                "Warmup steps (%d) >= total steps (%d). Adjusting warmup_steps -> %d",
                warmup_steps,
                numb_steps,
                new_warmup,
            )
            warmup_steps = new_warmup
    
    logging.info(
        "DEBUG final schedule: numb_steps raw=%r -> %d, warmup_steps raw=%r -> %d",
        raw_numb_steps, numb_steps, raw_warmup_steps, warmup_steps
    )

    # debug log: see what actually happened
    logging.info(
        "DEBUG hyper: model_type raw=%r final=%r, "
        "lr_type raw=%r final=%r, "
        "loss_type raw=%r final=%r, "
        "numb_steps raw=%r final=%r, "
        "warmup_steps raw=%r final=%r",
        raw_model_type,
        model_type,
        raw_lr_type,
        lr_type,
        raw_loss_type,
        loss_type,
        raw_numb_steps,
        numb_steps,
        raw_warmup_steps,
        warmup_steps,
    )

    user_provided_model = is_user_provided_model_path(model_path)
    
    if not user_provided_model:
        default_model = True
        if model_type == "dpa3":
            input_json = "/opt/agents/dpa_finetune/input_dpa3.json"
            model_path = Path("/opt/agents/dpa_finetune/models/dpa3.pt")
        else:
            input_json = "/opt/agents/dpa_finetune/input_dpa2.json"
            model_path = Path("/opt/agents/dpa_finetune/dpa2.pt")
    else:
        default_model = False
        if model_type == "dpa3":
            input_json = "/opt/agents/dpa_finetune/input_dpa3.json"
        else:
            input_json = "/opt/agents/dpa_finetune/input_dpa2.json"

    # Lock architecture-related params when using built-in default model
    if default_model:
        reset_params_to_signature_defaults(
            finetune_dpa_model,
            locals(),
            prefixes=("dpa3_", "dpa2_"),
        )
        logging.info("DEBUG: default_model=True -> locked all dpa3_* / dpa2_* params to signature defaults.")

     
    dpa3_repflow = {
            "n_dim": dpa3_repflow_n_dim,
            "e_dim": dpa3_repflow_e_dim,
            "a_dim": dpa3_repflow_a_dim,
            "nlayers": dpa3_repflow_nlayers,
            "e_rcut": dpa3_repflow_e_rcut,
            "e_rcut_smth": dpa3_repflow_e_rcut_smth,
            "e_sel": dpa3_repflow_e_sel,
            "a_rcut": dpa3_repflow_a_rcut,
            "a_rcut_smth": dpa3_repflow_a_rcut_smth,
            "a_sel": dpa3_repflow_a_sel,
            "axis_neuron": dpa3_repflow_axis_neuron,
            "fix_stat_std": dpa3_repflow_fix_stat_std,
            "a_compress_rate": dpa3_repflow_a_compress_rate,
            "a_compress_e_rate": dpa3_repflow_a_compress_e_rate,
            "a_compress_use_split": dpa3_repflow_a_compress_use_split,
            "update_angle": dpa3_repflow_update_angle,
            "smooth_edge_update": dpa3_repflow_smooth_edge_update,
            "use_dynamic_sel": dpa3_repflow_use_dynamic_sel,
            "sel_reduce_factor": dpa3_repflow_sel_reduce_factor,
            "use_exp_switch": dpa3_repflow_use_exp_switch,
            "update_style": dpa3_repflow_update_style,
            "update_residual": dpa3_repflow_update_residual,
            "update_residual_init": dpa3_repflow_update_residual_init,
            }
    dpa3_top = { 
            "activation_function": dpa3_top_activation_function,
            "use_tebd_bias": dpa3_top_use_tebd_bias,
            "precision": dpa3_top_precision,
            "concat_output_tebd": dpa3_top_concat_output_tebd,
                                 }
    dpa3_fitting_net = { 
            "neuron":dpa3_fitting_net_neuron,
            "dim_case_embd":dpa3_fitting_net_dim_case_embd,
            "resnet_dt": dpa3_fitting_net_resnet_dt,
            "precision":dpa3_fitting_net_precision,
            "activation_function":dpa3_fitting_net_activation_function,
            "seed":dpa3_fitting_net_seed,
            }
    dpa2_repinit = {
            "tebd_dim":dpa2_repinit_tebd_dim, 
            "rcut":dpa2_repinit_rcut,
            "rcut_smth":dpa2_repinit_rcut_smth,
            "nsel":dpa2_repinit_nsel,
            "neuron":dpa2_repinit_neuron,
            "axis_neuron":dpa2_repinit_axis_neuron,
            "three_body_neuron":dpa2_repinit_three_body_neuron,
            "activation_function":dpa2_repinit_activation_function,
            "three_body_sel":dpa2_repinit_three_body_sel,
            "three_body_rcut":dpa2_repinit_three_body_rcut,
            "three_body_rcut_smth":dpa2_repinit_three_body_rcut_smt,
            "use_three_body": dpa2_repinit_use_three_body,
                                      }
    dpa2_repformer  = {
            "rcut":dpa2_repformer_rcut, 
            "rcut_smth":dpa2_repformer_rcut_smth,
            "nsel":dpa2_repformer_nsel,
            "nlayers": dpa2_repformer_nlayers,
            "g1_dim": dpa2_repformer_g1_dim,
            "g2_dim": dpa2_repformer_g2_dim,
            "attn2_hidden": dpa2_repformer_attn2_hidden,
            "attn2_nhead": dpa2_repformer_attn2_nhead,
            "attn1_hidden": dpa2_repformer_attn1_hidden,
            "attn1_nhead": dpa2_repformer_attn1_nhead,
            "axis_neuron": dpa2_repformer_axis_neuron,
            "update_h2":  dpa2_repformer_update_h2,
            "update_g1_has_conv": dpa2_repformer_update_g1_has_conv,
            "update_g1_has_grrg": dpa2_repformer_update_g1_has_grrg,
            "update_g1_has_drrd": dpa2_repformer_update_g1_has_drrd,
            "update_g1_has_attn": dpa2_repformer_update_g1_has_attn,
            "update_g2_has_g1g1": dpa2_repformer_update_g2_has_g1g1,
            "update_g2_has_attn": dpa2_repformer_update_g2_has_attn,
            "update_style":dpa2_repformer_update_style,
            "update_residual":dpa2_repformer_update_residual,
            "update_residual_init":dpa2_repformer_update_residual_init,
            "attn2_has_gate": dpa2_repformer_attn2_has_gate,
            "use_sqrt_nnei": dpa2_repformer_use_sqrt_nnei,
            "g1_out_conv": dpa2_repformer_g1_out_conv,
            "g1_out_mlp": dpa2_repformer_g1_out_mlp,
            "activation_function":dpa2_repformer_activation_function,
                                        }
    dpa2_fitting_net = {
            "neuron": dpa2_fitting_net_neuron,
            "activation_function": dpa2_fitting_net_activation_function,
            "resnet_dt": dpa2_fitting_net_resnet_dt,
            "precision": dpa2_fitting_net_precision,
            "dim_case_embd": dpa2_fitting_net_dim_case_embd,
            "seed": dpa2_fitting_net_seed,
            }
    dpa2_top = {
            "use_tebd_bias": dpa2_top_use_tebd_bias,
            "precision": dpa2_top_precision,
            "add_tebd_to_repinit_out": dpa2_top_add_tebd_to_repinit_out,
                                  }
    #input_json = "bohrium://13756/501205/store/upload/523748fd-7d2e-4e94-9304-8d49e710e8ba/input_dpa3.json"
    update_dpa_train_json(
                      input_json,
                      model_type,
                      default_model,
                      lr_type,
                      decay_steps,
                      start_lr,
                      stop_lr,
                      loss_type,
                      start_pref_e,
                      limit_pref_e,
                      start_pref_f,
                      limit_pref_f,
                      start_pref_v,
                      limit_pref_v,
                      numb_steps,
                      warmup_steps,
                      dpa3_repflow,
                      dpa3_top,
                      dpa3_fitting_net,
                      dpa2_repinit,
                      dpa2_repformer,
                      dpa2_fitting_net,
                      dpa2_top,
                      "train.json"
                    )
    print(f"checkcheck lr_type={lr_type}")
#Split dpdata into train and valid two sets
    split_train_valid(str(input_path), valid_ratio)
    #model_path = "bohrium://13756/501205/store/upload/6ea62778-1c8a-4ff6-9b12-85288b1912aa/dpa3.pt"
    if model_type == "dpa3":
       cmd = [ "dp",
               "--pt",
               "train", 
               "train.json", 
               "--finetune", 
               str(model_path),
               ]
    else:
       cmd = [ "dp",
               "--pt",
               "train", 
               "train.json", 
               "--finetune", 
               str(model_path),
               "--model-branch",
               "Omat24"
               ]

    subprocess.run(cmd, check=True)


    finetuned_model = './model.ckpt.pt'
    finetuned_model_path = Path(finetuned_model)
    return{
      "results": finetuned_model_path,
      "message": "Fine tune model successfully!"
    }

# ====== Run Server ======

if __name__ == "__main__":
    logging.info("Starting FinetuneDPAServer on port 50003...")
    mcp.run(transport="sse")

