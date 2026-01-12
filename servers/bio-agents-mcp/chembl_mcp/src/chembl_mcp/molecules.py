import json
from mcp import types
from typing import List, Optional
from chembl_webresource_client.new_client import new_client

CLIENT = new_client.molecule


def get_molecule_pref_name(
    prefix_name: str,
    fields: Optional[List[str]] = None,
) -> types.TextContent:
    global CLIENT
    molecules = CLIENT.filter(pref_name__iexact=prefix_name)
    if fields:
        molecules = molecules.only(fields)

    return json.dumps(molecules, indent=2)


def get_molecule_synonyms(
    synonym: str,
    fields: Optional[List[str]] = None,
) -> types.TextContent:
    global CLIENT
    molecules = CLIENT.filter(
        molecule_synonyms__molecule_synonym_iexact=synonym)
    if fields:
        molecules = molecules.only(fields)

    return json.dumps(molecules, indent=2)


def get_molecule_chembl_id(
    chembl_id: str | List[str],
    fields: Optional[List[str]] = None,
) -> types.TextContent:
    global CLIENT
    if isinstance(chembl_id, str):
        chembl_id = [chembl_id]

    molecule = CLIENT.filter(molecule_chembl_id__in=chembl_id)

    if fields:
        molecule = molecule.only(fields)

    return json.dumps(molecule, indent=2)


def get_molecule_standard_inchi_key(
    standard_inchi_key: str,
    fields: Optional[List[str]] = None,
) -> types.TextContent:
    global CLIENT
    molecule = CLIENT.filter(
        molecule_structures__standard_inchi_key=standard_inchi_key)

    if fields:
        molecule = molecule.only(fields)

    return json.dumps(molecule, indent=2)
