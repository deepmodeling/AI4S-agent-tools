import pytest

from src.protein_data_bank_mcp.pdb_store.storage import PDBStore


@pytest.fixture
def pdb_store(tmp_path):
    return PDBStore(tmp_path)


def test_download_pdb(pdb_store: PDBStore):
    entry_id = "1fat"
    path = pdb_store.get_pdb(entry_id)
    assert path.is_file()


def test_get_residues(pdb_store: PDBStore):
    entry_id = "1fat"
    residues = pdb_store.get_residue_chains(entry_id)
    assert len(residues) == 4
