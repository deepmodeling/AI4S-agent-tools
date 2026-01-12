import argparse
import logging
import json
from typing import List, Optional, TypedDict, Literal
from pathlib import Path
from datetime import datetime
import hashlib
from anyio import to_thread
import asyncio

from optimade.client import OptimadeClient
from dp.agent.server import CalculationMCPServer

# Pull all helpers + provider sets from your utils.py
from utils import *

# === CONFIG ===
BASE_OUTPUT_DIR = Path("materials_data")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETURNED_STRUCTS = 100

# === ARG PARSING ===
def parse_args():
    parser = argparse.ArgumentParser(description="OPTIMADE Materials Data MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    try:
        return parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        return Args()


# === RESULT TYPE (what each tool returns) ===
Format = Literal["cif", "json"]

class FetchResult(TypedDict):
    output_dir: Path             # folder where results are saved
    cleaned_structures: List[dict]  # list of cleaned structures
    n_found: int                    # number of structures found (0 if none)


# === MCP SERVER ===
args = parse_args()
logging.basicConfig(level=args.log_level)
mcp = CalculationMCPServer("OptimadeServer", port=args.port, host=args.host)


# === TOOL 1: RAW OPTIMADE FILTER ===
@mcp.tool()
async def fetch_structures_with_filter(
    filter: str,
    as_format: Format = "cif",
    n_results: int = 2,
    providers: Optional[List[str]] = None,
) -> FetchResult:
    """
    Fetch crystal structures using a RAW OPTIMADE filter (single request across providers).

    What this does
    --------------
    - Sends the *exact* `filter` string to all selected providers in a single aggregated query.
    - Saves up to `n_results` results per provider in the specified `as_format` ("cif" or "json").

    Arguments
    ---------
    filter : str
        An OPTIMADE filter expression. Examples:
          - elements HAS ALL "Al","O","Mg"
          - elements HAS ONLY "Si","O"
          - chemical_formula_reduced="O2Si"
          - chemical_formula_descriptive CONTAINS "H2O"
          - (elements HAS ANY "Si") AND NOT (elements HAS ANY "H")
    as_format : {"cif","json"}
        Output format for saved structures (default "cif").
    n_results : int
        Number of results to save from EACH provider (default 2).
    providers : list[str] | None
        Providers to query. If omitted, uses DEFAULT_PROVIDERS from utils.py.

    Returns
    -------
    FetchResult
        output_dir: Path to the folder with saved results
        cleaned_structures: List[dict]  # list of cleaned structures
        n_found: int  # number of structures found (0 if none)
    """
    filt = (filter or "").strip()
    if not filt:
        logging.error("[raw] empty filter string")
        return {"output_dir": Path(), "cleaned_structures": [], "n_found": 0}
    filt = normalize_cfr_in_filter(filt)

    used_providers = set(providers) if providers else DEFAULT_PROVIDERS
    
    # Get all URLs for the selected providers (flatten the lists)
    used_urls = [url for provider in used_providers 
                 for url in URLS_FROM_PROVIDERS.get(provider, [])]
    
    logging.info(f"[raw] providers={used_providers} urls={len(used_urls)} filter={filt}")

    try:
        client = OptimadeClient(
            base_urls=used_urls,
            # include_providers=used_providers,
            max_results_per_provider=n_results,
            http_timeout=25.0,
        )
        results = await to_thread.run_sync(lambda: client.get(filter=filt))
    except (SystemExit, Exception) as e:  # catch SystemExit too
        logging.error(f"[raw] fetch failed: {e}")
        return {"output_dir": Path(), "cleaned_structures": [], "n_found": 0}

    # Timestamped folder + short hash of filter for traceability
    tag = filter_to_tag(filt)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha1(filt.encode("utf-8")).hexdigest()[:8]
    out_folder = BASE_OUTPUT_DIR / f"{tag}_{ts}_{short}"

    files, warns, providers_seen, cleaned_structures = await to_thread.run_sync(
        save_structures, results, out_folder, n_results, as_format == "cif"
    )

    manifest = {
        "mode": "raw_filter",
        "filter": filt,
        "providers_requested": sorted(list(used_providers)),
        "providers_seen": providers_seen,
        "files": files,
        "warnings": warns,
        "format": as_format,
        "n_results": n_results,
        "n_found": len(cleaned_structures), 
    }
    (out_folder / "summary.json").write_text(json.dumps(manifest, indent=2))

    cleaned_structures = cleaned_structures[:MAX_RETURNED_STRUCTS]

    return {
        "output_dir": out_folder,
        "cleaned_structures": cleaned_structures,
        "n_found": len(cleaned_structures),
    }


# === TOOL 2: SPACE-GROUP AWARE FETCH (provider-specific fields, parallel) ===
@mcp.tool()
async def fetch_structures_with_spg(
    base_filter: Optional[str],
    spg_number: int,
    as_format: Format = "cif",
    n_results: int = 3,
    providers: Optional[List[str]] = None,
) -> FetchResult:
    """
    Fetch structures constrained by space group number across multiple providers in parallel.

    What this does
    --------------
    - Builds a provider-specific space-group clause (e.g., `_tcod_sg="P m -3 m"`, `_oqmd_spacegroup="Pm-3m"`,
      `_alexandria_space_group=221`, etc.) via `get_spg_filter_map`.
    - Combines it with your optional `base_filter` (elements/formula logic).
    - Runs per-provider queries **in parallel**, then saves all results into one folder.

    Arguments
    ---------
    base_filter : str | None
        Common OPTIMADE filter applied to all providers (e.g., "elements HAS ONLY \"Ti\",\"Al\"").
    spg_number : int
        International space-group number (1-230).
    as_format : {"cif","json"}
        Output format for saved structures (default "cif").
    n_results : int
        Number of results to save from EACH provider (default 3).
    providers : list[str] | None
        Providers to query. If omitted, uses DEFAULT_SPG_PROVIDERS from utils.py.

    Returns
    -------
    FetchResult
        output_dir: Path to the folder with saved results
        cleaned_structures: List[dict]  # list of cleaned structures
        n_found: int  # number of structures found (0 if none)
    """
    base = (base_filter or "").strip()
    base = normalize_cfr_in_filter(base)
    used = set(providers) if providers else DEFAULT_SPG_PROVIDERS

    # Build provider-specific SPG clauses and combine with base filter
    spg_map = get_spg_filter_map(spg_number, used)
    filters = build_provider_filters(base, spg_map)
    if not filters:
        logging.warning("[spg] no provider-specific space-group clause available")
        return {"output_dir": Path(), "cleaned_structures": [], "n_found": 0}

    async def _query_one(provider: str, clause: str) -> dict:
        logging.info(f"[spg] {provider}: {clause}")
        try:
            # Get all URLs for this provider (flatten the lists)
            provider_urls = [url for url in URLS_FROM_PROVIDERS.get(provider, [])]
            if not provider_urls:
                logging.warning(f"[spg] No URLs found for provider {provider}")
                return {"structures": {}}
                
            client = OptimadeClient(
                base_urls=provider_urls,
                max_results_per_provider=n_results,
                http_timeout=25.0,
            )
            return await to_thread.run_sync(lambda: client.get(filter=clause))
        except (SystemExit, Exception) as e:  # catch SystemExit too
            logging.error(f"[spg] fetch failed for {provider}: {e}")
            return {"structures": {}}

    # Parallel fan‑out per provider
    results_list = await asyncio.gather(
        *[_query_one(p, clause) for p, clause in filters.items()],
        return_exceptions=True,  # don't cancel all on one failure
    )

    # Turn any exceptions into empty result dicts
    norm_results = []
    for r in results_list:
        if isinstance(r, Exception):
            logging.error(f"[spg] task returned exception: {r}")
            norm_results.append({"structures": {}})
        else:
            norm_results.append(r)

    # Save all results together
    tag = filter_to_tag(f"{base} AND spg={spg_number}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha1(f"{base}|spg={spg_number}".encode("utf-8")).hexdigest()[:8]
    out_folder = BASE_OUTPUT_DIR / f"{tag}_{ts}_{short}"

    all_files: List[str] = []
    all_warnings: List[str] = []
    all_providers: List[str] = []
    all_cleaned: List[dict] = []
    for res in norm_results:
        files, warns, providers_seen, cleaned = await to_thread.run_sync(
            save_structures, res, out_folder, n_results, as_format == "cif"
        )
        all_files.extend(files)
        all_warnings.extend(warns)
        all_providers.extend(providers_seen)
        all_cleaned.extend(cleaned)

    manifest = {
        "mode": "space_group",
        "base_filter": base,
        "spg_number": spg_number,
        "providers_requested": sorted(list(used)),
        "providers_seen": all_providers,
        "files": all_files,
        "warnings": all_warnings,
        "format": as_format,
        "n_results": n_results,
        "per_provider_filters": filters,
        "n_found": len(all_cleaned),
    }
    (out_folder / "summary.json").write_text(json.dumps(manifest, indent=2))

    all_cleaned = all_cleaned[:MAX_RETURNED_STRUCTS]

    return {
        "output_dir": out_folder,
        "cleaned_structures": all_cleaned,
        "n_found": len(all_cleaned),
    }


# === TOOL 3: BAND‑GAP RANGE FETCH (provider-specific fields, parallel) ===
@mcp.tool()
async def fetch_structures_with_bandgap(
    base_filter: Optional[str],
    min_bg: Optional[float] = None,
    max_bg: Optional[float] = None,
    as_format: Format = "cif",
    n_results: int = 2,
    providers: Optional[List[str]] = None,
) -> FetchResult:
    """
    Fetch structures constrained by band-gap range across multiple providers in parallel.

    What this does
    --------------
    - Resolves provider-specific band-gap property names (e.g., `_oqmd_band_gap`, `_gnome_bandgap`,
      `_mcloudarchive_band_gap`, etc.) via `get_bandgap_filter_map`.
    - Builds a per-provider band-gap clause (min/max inclusive), combines with your optional `base_filter`.
    - Runs each provider query **in parallel**, then saves all results into one folder.

    Arguments
    ---------
    base_filter : str | None
        Common OPTIMADE filter applied to all providers (e.g., 'elements HAS ALL "Al"').
    min_bg, max_bg : float | None
        Band-gap range in eV (open-ended allowed, e.g., min only or max only).
    as_format : {"cif","json"}
        Output format for saved structures (default "cif").
    n_results : int
        Number of results to save from EACH provider (default 2).
    providers : list[str] | None
        Providers to query; if None, uses DEFAULT_BG_PROVIDERS from utils.py.

    Returns
    -------
    FetchResult
        output_dir: Path to the folder with saved results
        cleaned_structures: List[dict]  # list of cleaned structures
        n_found: int  # number of structures found (0 if none)
    """
    base = (base_filter or "").strip()
    base = normalize_cfr_in_filter(base)
    used = set(providers) if providers else DEFAULT_BG_PROVIDERS

    # Build per-provider bandgap clause and combine with base
    bg_map = get_bandgap_filter_map(min_bg, max_bg, used)
    filters = build_provider_filters(base, bg_map)

    if not filters:
        logging.warning("[bandgap] no provider-specific band-gap clause available")
        return {"output_dir": Path(), "cleaned_structures": [], "n_found": 0}

    async def _query_one(provider: str, clause: str) -> dict:
        logging.info(f"[bandgap] {provider}: {clause}")
        try:
            # Get all URLs for this provider (flatten the lists)
            provider_urls = [url for url in URLS_FROM_PROVIDERS.get(provider, [])]
            if not provider_urls:
                logging.warning(f"[bandgap] No URLs found for provider {provider}")
                return {"structures": {}}
                
            client = OptimadeClient(
                base_urls=provider_urls,
                max_results_per_provider=n_results,
                http_timeout=25.0,
            )
            return await to_thread.run_sync(lambda: client.get(filter=clause))
        except (SystemExit, Exception) as e:  # catch SystemExit too
            logging.error(f"[bandgap] fetch failed for {provider}: {e}")
            return {"structures": {}}

    # Parallel fan‑out per provider
    results_list = await asyncio.gather(
        *[_query_one(p, clause) for p, clause in filters.items()],
        return_exceptions=True,  # don't cancel all on one failure
    )
    norm_results = []
    for r in results_list:
        if isinstance(r, Exception):
            logging.error(f"[bandgap] task returned exception: {r}")
            norm_results.append({"structures": {}})
        else:
            norm_results.append(r)

    # Save all results together
    tag = filter_to_tag(f"{base} AND bandgap[{min_bg},{max_bg}]")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha1(f"{base}|bg={min_bg}:{max_bg}".encode("utf-8")).hexdigest()[:8]
    out_folder = BASE_OUTPUT_DIR / f"{tag}_{ts}_{short}"

    all_files: List[str] = []
    all_warnings: List[str] = []
    all_providers: List[str] = []
    all_cleaned: List[dict] = []
    for res in norm_results:
        files, warns, providers_seen, cleaned = await to_thread.run_sync(
            save_structures, res, out_folder, n_results, as_format == "cif"
        )
        all_files.extend(files)
        all_warnings.extend(warns)
        all_providers.extend(providers_seen)
        all_cleaned.extend(cleaned)

    manifest = {
        "mode": "band_gap",
        "base_filter": base,
        "band_gap_min": min_bg,
        "band_gap_max": max_bg,
        "providers_requested": sorted(list(used)),
        "providers_seen": all_providers,
        "files": all_files,
        "warnings": all_warnings,
        "format": as_format,
        "n_results": n_results,
        "per_provider_filters": filters,
        "n_found": len(all_cleaned),
    }
    (out_folder / "summary.json").write_text(json.dumps(manifest, indent=2))

    all_cleaned = all_cleaned[:MAX_RETURNED_STRUCTS]

    return {
        "output_dir": out_folder,
        "cleaned_structures": all_cleaned,
        "n_found": len(all_cleaned),
    }


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)