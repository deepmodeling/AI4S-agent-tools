import json
import logging

from mcp import types
from typing import Dict
from Bio.PDB import PDBList, PDBParser
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


PROTEIN_COMMON_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

PROTEIN_COMMON_THREE_TO_ONE = {v: k for k, v in PROTEIN_COMMON_ONE_TO_THREE.items()}


class PDBStore:
    def __init__(self, folder: Path | str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._pdb_list = PDBList()
        self._parser = PDBParser()

    def download_pdb(self, entry_id: str):
        self._pdb_list.retrieve_pdb_file(
            entry_id, pdir=self.folder, overwrite=True, file_format="pdb"
        )

    def get_pdb(self, entry_id: str):
        pdb_path = self.folder / f"pdb{entry_id.lower()}.ent"
        if not pdb_path.exists():
            self.download_pdb(entry_id)
        return pdb_path

    def get_residue_chains(self, entry_id: str) -> Dict[str, str]:
        path = self.get_pdb(entry_id)
        structure = self._parser.get_structure(entry_id, path)

        chains = {}
        for chain in structure.get_chains():
            residues = "".join(
                [
                    PROTEIN_COMMON_THREE_TO_ONE.get(x.resname, "X")
                    for x in chain.get_residues()
                ]
            )
            chains[chain.id] = residues

        return chains


parser = PDBStore("temp")


def get_residue_chains(entry_id: str) -> types.TextContent:
    global parser
    return json.dumps(parser.get_residue_chains(entry_id), indent=2)
