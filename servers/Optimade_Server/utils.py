import os
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Iterable, Optional
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.groups import SpaceGroup

import re

from urllib.parse import urlparse
import oss2
from dotenv import load_dotenv
from oss2.credentials import EnvironmentVariableCredentialsProvider

# === LOAD ENV ===
load_dotenv()


DEFAULT_PROVIDERS = {
    # "aflow",
    "alexandria",
    # "aiida",
    # "ccdc",
    # "ccpnc",
    "cmr",
    "cod",
    # "httk",
    # "jarvis",
    "mcloud",
    "mcloudarchive",
    "mp",
    "mpdd",
    "mpds",
    # "mpod",
    "nmd",
    "odbx",
    "omdb",
    "oqmd",
    # "optimade",
    # "optimake",
    # "pcod",
    # "psdi",
    "tcod",
    "twodmatpedia",
}

DEFAULT_SPG_PROVIDERS = {
    "alexandria",
    "cod",
    "mpdd",
    "nmd",
    "odbx",
    "oqmd",
    "tcod",
}

DEFAULT_BG_PROVIDERS = {
    "alexandria",
    "odbx",
    "oqmd",
    "mcloudarchive",
    "twodmatpedia",
}

URLS_FROM_PROVIDERS = {
    "aflow": ["https://aflow.org/API/optimade/"],
    "alexandria": [
        "https://alexandria.icams.rub.de/pbe",
        "https://alexandria.icams.rub.de/pbesol"
    ],
    "cod": ["https://www.crystallography.net/cod/optimade"],
    "cmr": ["https://cmr-optimade.fysik.dtu.dk/"],
    "mcloud": [
        "https://optimade.materialscloud.io/main/mc3d-pbe-v1",
        "https://optimade.materialscloud.io/main/mc2d",
        "https://optimade.materialscloud.io/main/2dtopo",
        "https://optimade.materialscloud.io/main/tc-applicability",
        "https://optimade.materialscloud.io/main/pyrene-mofs",
        "https://optimade.materialscloud.io/main/curated-cofs",
        "https://optimade.materialscloud.io/main/stoceriaitf",
        "https://optimade.materialscloud.io/main/autowannier",
        "https://optimade.materialscloud.io/main/tin-antimony-sulfoiodide"
    ],
    "mcloudarchive": [
        "https://optimade.materialscloud.org/archive/zk-gc",
        "https://optimade.materialscloud.org/archive/c8-gy",
        "https://optimade.materialscloud.org/archive/5p-vq",
        "https://optimade.materialscloud.org/archive/vg-ya"
    ],
    "mp": ["https://optimade.materialsproject.org/"],
    "mpdd": ["http://mpddoptimade.phaseslab.org/"],
    "mpds": ["https://api.mpds.io/"],
    "mpod": ["http://mpod_optimade.cimav.edu.mx/"],
    "nmd": ["https://nomad-lab.eu/prod/rae/optimade/"],
    "odbx": [
        "https://optimade.odbx.science/",
        "https://optimade-misc.odbx.science/",
        "https://optimade-gnome.odbx.science/"
    ],
    "omdb": ["http://optimade.openmaterialsdb.se/"],
    "oqmd": ["https://oqmd.org/optimade/"],
    "jarvis": ["https://jarvis.nist.gov/optimade/jarvisdft"],
    "tcod": ["https://www.crystallography.net/tcod/optimade"],
    "twodmatpedia": ["http://optimade.2dmatpedia.org/"]
}

DROP_ATTRS = {
    "cartesian_site_positions",
    "species_at_sites",
    "species",
    "immutable_id",
    "_alexandria_charges",
    "_alexandria_magnetic_moments",
    "_alexandria_forces",
    "_alexandria_scan_forces",
    "_alexandria_scan_charges",
    "_alexandria_scan_magnetic_moments",
    "_nmd_dft_quantities",
    "_nmd_files",
    "_nmd_dft_geometries",
    "_mpdd_descriptors",
    "_mpdd_poscar",
}

# === UTILS ===
def hill_formula_filter(formula: str) -> str:
    hill_formula = Composition(formula).hill_formula.replace(' ', '')
    return f'chemical_formula_reduced="{hill_formula}"'

# regex for chemical_formula_reduced="..."/'...'
_CFR_EQ = re.compile(r'(?i)\bchemical_formula_reduced\b\s*=\s*([\'"])(.+?)\1')

def normalize_cfr_in_filter(filter_str: str) -> str:
    """Normalize all chemical_formula_reduced=... clauses (0, 1, many)."""
    if not filter_str:
        return filter_str

    def repl(m):
        raw = m.group(2)  # the formula
        return hill_formula_filter(raw)

    return _CFR_EQ.sub(repl, filter_str)

# === Saver ===
def _provider_name_from_url(url: str) -> str:
    """Turn provider URL into a filesystem-safe name."""
    parsed = urlparse(url)
    netloc = parsed.netloc.replace('.', '_')
    path = parsed.path.strip('/').replace('/', '_')
    name = f"{netloc}_{path}" if path else netloc
    return name.strip('_') or "provider"

def shorten_id(orig_id: str, head: int = 6, tail: int = 3, min_len: int = 12) -> str:
    """
    Shorten a long ID for display.

    Args:
        orig_id: the original string ID
        head: number of characters to keep at the start
        tail: number of characters to keep at the end
        min_len: minimum length before shortening is applied

    Returns:
        A shortened ID string like 'abcdef...xyz' if longer than min_len,
        otherwise the original ID unchanged.
    """
    if not orig_id:
        return orig_id
    if len(orig_id) > min_len:
        return f"{orig_id[:head]}...{orig_id[-tail:]}"
    return orig_id

def save_structures(results: Dict, output_folder: Path, max_results: int, as_cif: bool):
    """
    Walk OPTIMADE aggregated results and write per-provider files.
    Returns files list, warnings list, providers_seen list.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    warnings: List[str] = []
    providers_seen: List[str] = []
    cleaned_structures: List[dict] = []

    seen_ids: set[str] = set()

    structures_by_filter = results.get("structures", {})
    structures_by_url = list(structures_by_filter.values())[0]
    for provider_url, content in structures_by_url.items():
        provider_name = _provider_name_from_url(provider_url)
        providers_seen.append(provider_name)
        data_list = content.get("data", [])
        logging.info(f"[save] {provider_name}: {len(data_list)} candidates")

        saved = 0
        for structure_data in data_list:
            if saved >= max_results:
                break

            orig_id = str(structure_data.get("id", ""))
            if orig_id in seen_ids:
                logging.warning(f"[save] duplicate skipped: {orig_id}")
                continue
            seen_ids.add(orig_id)

            # ---------- file write ----------
            suffix = "cif" if as_cif else "json"
            filename = f"{provider_name}_{orig_id}_{saved}.{suffix}"
            file_path = output_folder / filename

            try:
                if as_cif:
                    cif_content = Structure(
                        lattice=structure_data['attributes']['lattice_vectors'], 
                        species=structure_data['attributes']['species_at_sites'], 
                        coords=structure_data['attributes']['cartesian_site_positions'], 
                        coords_are_cartesian=True,
                    ).to(fmt='cif')
                    if not cif_content or not cif_content.strip():
                        raise ValueError("CIF content is empty")
                    file_path.write_text(cif_content)
                else:
                    file_path.write_text(json.dumps(structure_data, indent=2))

                logging.debug(f"[save] wrote {file_path}")
                files.append(str(file_path))
            except Exception as e:
                msg = f"Failed to save structure from {provider_name} #{orig_id}: {e}"
                logging.warning(msg)
                warnings.append(msg)

            # ---------- cleaned copy ----------
            try:
                sd = dict(structure_data)
                attrs = dict(sd.get("attributes", {}) or {})
                for k in DROP_ATTRS:
                    attrs.pop(k, None)
                sd["attributes"] = attrs
                sd["provider_url"] = provider_url
                # # overwrite id with short display form (first 6 + '...' + last 3)
                # orig_id = str(sd.get("id", ""))
                # sd["id"] = shorten_id(orig_id, head=6, tail=3, min_len=12)

                cleaned_structures.append(sd)
                
            except Exception as e:
                logging.warning(f"[save] clean-copy failed for {provider_name} #{orig_id}: {e}")

            saved += 1

    return files, warnings, providers_seen, cleaned_structures


def filter_to_tag(filter_str: str, max_len: int = 30) -> str:
    """
    Convert an OPTIMADE filter string into a short, filesystem-safe tag.

    Parameters
    ----------
    filter_str : str
        The original OPTIMADE filter string.
    max_len : int, optional
        Maximum length of the resulting tag (default: 30).

    Returns
    -------
    str
        A short, sanitized tag derived from the filter.
    """
    tag = filter_str.strip().replace('"', '').replace("'", "")
    tag = tag.replace(" ", "_").replace(",", "-").replace("=", "")
    tag = "".join(c for c in tag if c.isalnum() or c in "_-")

    # Limit length
    if len(tag) > max_len:
        tag = tag[:max_len]

    # Fallback if everything gets stripped
    return tag or "filter"



def _hm_symbol_from_number(spg_number: int) -> Optional[str]:
    """Return the short Hermann–Mauguin symbol (e.g. 'Im-3m') for a space-group number."""
    try:
        return SpaceGroup.from_int_number(spg_number).symbol
    except Exception as e:
        logging.warning(f"[spg] cannot map number {spg_number} to H–M symbol: {e}")
        return None

def _to_tcod_format(hm: str) -> str:
    """
    Convert a short Hermann–Mauguin symbol to TCOD spacing.
    """
    s = hm.strip()
    s = re.sub(r'/([A-Za-z]+)', lambda m: '/' + ' '.join(m.group(1)), s)
    s = re.sub(r'(?<=[A-Za-z])(?=[A-Za-z])', ' ', s)
    s = re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])', ' ', s)
    s = re.sub(r'\s*-\s*(?=\d)', ' -', s)
    return ' '.join(s.split())

def get_spg_filter_map(spg_number: int, providers: Iterable[str]) -> Dict[str, str]:
    """
    Map provider name → space-group filter clause for that provider.
    Handles alexandria, nmd, mpdd, odbx, oqmd, tcod, cod.
    """
    hm = _hm_symbol_from_number(spg_number)

    name_map = {
        "alexandria": lambda: f"_alexandria_space_group={spg_number}",
        "nmd":        lambda: f"_nmd_dft_spacegroup={spg_number}",
        "mpdd":       lambda: f"_mpdd_spacegroupn={spg_number}",
        "odbx":       lambda: f"_gnome_space_group_it_number={spg_number}",
        "oqmd":       lambda: f'_oqmd_spacegroup="{hm}"' if hm else "",
        "tcod":       lambda: f'_tcod_sg="{_to_tcod_format(hm)}"' if hm else "",
        "cod":        lambda: f'_cod_sg="{_to_tcod_format(hm)}"' if hm else "",
    }

    out: Dict[str, str] = {}
    for p in providers:
        if p in name_map:
            clause = name_map[p]()
            if clause:
                out[p] = clause
    return out


def _range_clause(prop: str, min_bg: Optional[float], max_bg: Optional[float]) -> str:
    """Return OPTIMADE range clause like: prop>=a AND prop<=b (handles open ends)."""
    parts = []
    if min_bg is not None:
        parts.append(f"{prop}>={min_bg}")
    if max_bg is not None:
        parts.append(f"{prop}<={max_bg}")
    return " AND ".join(parts) if parts else ""  # empty means 'no constraint'

def get_bandgap_filter_map(
    min_bg: Optional[float],
    max_bg: Optional[float],
    providers: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """
    Map provider name → band-gap clause using provider-specific property names.
    Providers without a known property are omitted.
    If providers is None, uses DEFAULT_BG_PROVIDERS.
    """
    providers = set(providers) if providers else DEFAULT_BG_PROVIDERS

    name_map = {
        "alexandria": "_alexandria_band_gap",
        "odbx": "_gnome_bandgap",             
        "oqmd": "_oqmd_band_gap",
        "mcloudarchive": "_mcloudarchive_band_gap",
        "twodmatpedia": "_twodmatpedia_band_gap",
    }

    out: Dict[str, str] = {}
    for p in providers:
        prop = name_map.get(p)
        if not prop:
            continue
        clause = _range_clause(prop, min_bg, max_bg)
        if clause:
            out[p] = clause
    return out

def build_provider_filters(base: Optional[str], provider_map: Dict[str, str]) -> Dict[str, str]:
    """
    Combine a base OPTIMADE filter with per-provider clauses.

    Parameters
    ----------
    base : str, optional
        Common OPTIMADE filter applied to all providers (can be empty/None).
    provider_map : dict
        {provider: specific_clause} mapping for each provider.

    Returns
    -------
    dict
        {provider: combined_clause}
    """
    b = (base or "").strip()
    return {
        p: f"({b}) AND ({c.strip()})" if b and c.strip() else (b or c.strip())
        for p, c in provider_map.items()
        if c and c.strip()  # skip empty clauses
    }


def get_base_urls() -> List[str]:

    try:
        from optimade.utils import get_all_databases

        base_urls = list(get_all_databases())
        return base_urls

    except ImportError:
        print("Warning: optimade.utils not available")
        return []
    except Exception as e:
        print(f"Error getting base URLs: {e}")
        return []


if __name__ == "__main__":
    urls = get_base_urls()
    print(urls)
    print(len(urls))