#!/usr/bin/env python3
import re
import os
import socket
import json
import logging
import argparse
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncIterator

from mcp.server.fastmcp import FastMCP, Context

##############################################################################
# PYMOL COMMAND DEFINITIONS AND ERROR PATTERNS (Provided by user)
##############################################################################

PYMOL_COMMANDS = {
    # MOLECULAR VISUALIZATION
    "show": {
        "description": "Shows a representation for the specified selection",
        "pattern": r"^show\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "representation", "required": True, "options": [
                "lines", "sticks", "spheres", "surface", "mesh", "dots", 
                "ribbon", "cartoon", "labels", "nonbonded", "nb_spheres", 
                "ellipsoids", "volume", "slice", "extent", "dots_as_spheres", 
                "cell", "cgo", "everything", "dashes", "angles", "dihedrals", 
                "licorice", "spheres", "putty"
            ]},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "hide": {
        "description": "Hides a representation for the specified selection",
        "pattern": r"^hide\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "representation", "required": True, "options": [
                "lines", "sticks", "spheres", "surface", "mesh", "dots", 
                "ribbon", "cartoon", "labels", "nonbonded", "nb_spheres", 
                "ellipsoids", "volume", "slice", "extent", "dots_as_spheres", 
                "cell", "cgo", "everything", "dashes", "angles", "dihedrals", 
                "licorice", "spheres", "putty"
            ]},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "color": {
        "description": "Sets the color for the specified selection",
        "pattern": r"^color\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "color", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "as": {
        "description": "Shows one representation while hiding all others for the specified selection",
        "pattern": r"^as\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "representation", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True 
    },
    "set": {
        "description": "Sets a PyMOL setting to a specified value",
        "pattern": r"^set\s+([\w.]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "setting", "required": True},
            {"name": "value", "required": True},
            {"name": "selection", "required": False}
        ],
        "check_selection": False
    },
    "cartoon": {
        "description": "Sets the cartoon type for the specified selection",
        "pattern": r"^cartoon\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "type", "required": True, "options": [
                "automatic", "loop", "rectangle", "oval", "tube", "arrow", "dumbbell", "putty"
            ]},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "spectrum": {
        "description": "Colors selection in a spectrum",
        "pattern": r"^spectrum\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "expression", "required": True},
            {"name": "palette", "required": False, "default": "rainbow"},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "label": {
        "description": "Adds labels to atoms in the selection",
        "pattern": r"^label\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "selection", "required": True},
            {"name": "expression", "required": False, "default": "name"}
        ],
        "check_selection": True
    },
    "distance": {
        "description": "Measures the distance between two selections",
        "pattern": r"^distance(?:\s+([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?$",
        "parameters": [
            {"name": "name", "required": False},
            {"name": "selection1", "required": False, "default": "(pk1)"},
            {"name": "selection2", "required": False, "default": "(pk2)"}
        ],
        "check_selection": True
    },
    "angle": {
        "description": "Measures the angle between three selections",
        "pattern": r"^angle(?:\s+([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?$",
        "parameters": [
            {"name": "name", "required": False},
            {"name": "selection1", "required": False, "default": "(pk1)"},
            {"name": "selection2", "required": False, "default": "(pk2)"},
            {"name": "selection3", "required": False, "default": "(pk3)"}
        ],
        "check_selection": True
    },
    "dihedral": {
        "description": "Measures the dihedral angle between four selections",
        "pattern": r"^dihedral(?:\s+([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?$",
        "parameters": [
            {"name": "name", "required": False},
            {"name": "selection1", "required": False, "default": "(pk1)"},
            {"name": "selection2", "required": False, "default": "(pk2)"},
            {"name": "selection3", "required": False, "default": "(pk3)"},
            {"name": "selection4", "required": False, "default": "(pk4)"}
        ],
        "check_selection": True
    },
    
    # VIEWING OPERATIONS
    "center": {
        "description": "Centers the view on a selection",
        "pattern": r"^center(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "orient": {
        "description": "Orients the view to align with principal axes of the selection",
        "pattern": r"^orient(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "zoom": {
        "description": "Zooms the view on a selection",
        "pattern": r"^zoom(?:\s+([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"},
            {"name": "buffer", "required": False, "default": "5"}
        ],
        "check_selection": True
    },
    "reset": {
        "description": "Resets the view, optionally resetting an object's matrix",
        "pattern": r"^reset(?:\s+(.+))?$",
        "parameters": [
            {"name": "object", "required": False}
        ],
        "check_selection": False
    },
    "turn": {
        "description": "Rotates the camera around an axis",
        "pattern": r"^turn\s+([xyz])(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "axis", "required": True, "options": ["x", "y", "z"]},
            {"name": "angle", "required": False, "default": "90"}
        ],
        "check_selection": False
    },
    "move": {
        "description": "Moves the camera along an axis",
        "pattern": r"^move\s+([xyz])(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "axis", "required": True, "options": ["x", "y", "z"]},
            {"name": "distance", "required": False, "default": "1"}
        ],
        "check_selection": False
    },
    "clip": {
        "description": "Adjusts the clipping planes",
        "pattern": r"^clip\s+([\w.]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "mode", "required": True, "options": ["near", "far", "slab", "atoms", "near_slab", "far_slab"]},
            {"name": "distance", "required": False, "default": "1"}
        ],
        "check_selection": False
    },
    
    # FILE OPERATIONS
    "load": {
        "description": "Loads a file into PyMOL",
        "pattern": r"^load\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "filename", "required": True},
            {"name": "object", "required": False},
            {"name": "options", "required": False}
        ],
        "check_selection": False
    },
    "fetch": {
        "description": "Fetches a structure from a database (e.g., PDB)",
        "pattern": r"^fetch\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "code", "required": True},
            {"name": "name", "required": False},
            {"name": "options", "required": False}
        ],
        "check_selection": False
    },
    "save": {
        "description": "Saves data to a file",
        "pattern": r"^save\s+([^,]+)(?:\s*,\s*(.+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "filename", "required": True},
            {"name": "selection", "required": False, "default": "all"},
            {"name": "state", "required": False, "default": "-1"}
        ],
        "check_selection": True
    },
    "png": {
        "description": "Saves a PNG image",
        "pattern": r"^png\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "filename", "required": True},
            {"name": "options", "required": False}
        ],
        "check_selection": False
    },
    
    # SELECTION OPERATIONS
    "select": {
        "description": "Creates a named selection",
        "pattern": r"^select\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "name", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": False
    },
    "deselect": {
        "description": "Clears the current selection",
        "pattern": r"^deselect$",
        "parameters": [],
        "check_selection": False
    },
    
    # OBJECT MANIPULATION
    "create": {
        "description": "Creates a new object from a selection",
        "pattern": r"^create\s+([^,]+)(?:\s*,\s*(.+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "name", "required": True},
            {"name": "selection", "required": False, "default": "all"},
            {"name": "source_state", "required": False, "default": "1"}
        ],
        "check_selection": True
    },
    "extract": {
        "description": "Extracts a selection to a new object",
        "pattern": r"^extract\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "name", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "delete": {
        "description": "Deletes objects or selections",
        "pattern": r"^delete\s+(.+)$",
        "parameters": [
            {"name": "name", "required": True}
        ],
        "check_selection": False
    },
    "remove": {
        "description": "Removes atoms in a selection",
        "pattern": r"^remove\s+(.+)$",
        "parameters": [
            {"name": "selection", "required": True}
        ],
        "check_selection": True
    },
    "align": {
        "description": "Aligns one selection to another",
        "pattern": r"^align\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "mobile", "required": True},
            {"name": "target", "required": False, "default": "all"},
            {"name": "options", "required": False}
        ],
        "check_selection": True
    },
    "super": {
        "description": "Superimposes one selection onto another",
        "pattern": r"^super\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "mobile", "required": True},
            {"name": "target", "required": False, "default": "all"},
            {"name": "options", "required": False}
        ],
        "check_selection": True
    },
    "intra_fit": {
        "description": "Fits all states within an object",
        "pattern": r"^intra_fit\s+(.+)$",
        "parameters": [
            {"name": "selection", "required": True}
        ],
        "check_selection": True
    },
    "intra_rms": {
        "description": "Calculates RMSD between states within an object",
        "pattern": r"^intra_rms\s+(.+)$",
        "parameters": [
            {"name": "selection", "required": True}
        ],
        "check_selection": True
    },
    
    # UTILITY AND MODIFICATION
    "alter": {
        "description": "Alters atomic properties in a selection",
        "pattern": r"^alter\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "selection", "required": True},
            {"name": "expression", "required": True}
        ],
        "check_selection": True
    },
    "alter_state": {
        "description": "Alters atomic coordinates in a state",
        "pattern": r"^alter_state\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "state", "required": True},
            {"name": "selection", "required": True},
            {"name": "expression", "required": True}
        ],
        "check_selection": True
    },
    "h_add": {
        "description": "Adds hydrogens to a selection",
        "pattern": r"^h_add(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "h_fill": {
        "description": "Adds hydrogens and adjusts valences",
        "pattern": r"^h_fill(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "bond": {
        "description": "Creates a bond between two atoms",
        "pattern": r"^bond\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "atom1", "required": True},
            {"name": "atom2", "required": True},
            {"name": "order", "required": False, "default": "1"}
        ],
        "check_selection": True
    },
    "unbond": {
        "description": "Removes a bond between two atoms",
        "pattern": r"^unbond\s+([^,]+)(?:\s*,\s*([^,]+))?$",
        "parameters": [
            {"name": "atom1", "required": True},
            {"name": "atom2", "required": True}
        ],
        "check_selection": True
    },
    "rebuild": {
        "description": "Regenerates all displayed geometry",
        "pattern": r"^rebuild(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": False
    },
    "refresh": {
        "description": "Refreshes the display",
        "pattern": r"^refresh$",
        "parameters": [],
        "check_selection": False
    },
    
    # UTILITY FUNCTIONS
    "util.cbc": {
        "description": "Colors by chain (Color By Chain)",
        "pattern": r"^util\.cbc(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbaw": {
        "description": "Colors by atom, white carbons (Color By Atom, White)",
        "pattern": r"^util\.cbaw(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbag": {
        "description": "Colors by atom, green carbons (Color By Atom, Green)",
        "pattern": r"^util\.cbag(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbac": {
        "description": "Colors by atom, cyan carbons (Color By Atom, Cyan)",
        "pattern": r"^util\.cbac(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbam": {
        "description": "Colors by atom, magenta carbons (Color By Atom, Magenta)",
        "pattern": r"^util\.cbam(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbay": {
        "description": "Colors by atom, yellow carbons (Color By Atom, Yellow)",
        "pattern": r"^util\.cbay(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbas": {
        "description": "Colors by atom, salmon carbons (Color By Atom, Salmon)",
        "pattern": r"^util\.cbas(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbab": {
        "description": "Colors by atom, slate carbons (Color By Atom, slateBLue)",
        "pattern": r"^util\.cbab(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbao": {
        "description": "Colors by atom, orange carbons (Color By Atom, Orange)",
        "pattern": r"^util\.cbao(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbap": {
        "description": "Colors by atom, purple carbons (Color By Atom, Purple)",
        "pattern": r"^util\.cbap(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.cbak": {
        "description": "Colors by atom, pink carbons (Color By Atom, pinK)",
        "pattern": r"^util\.cbak(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.chainbow": {
        "description": "Colors chains in rainbow gradient (CHAINs in rainBOW)",
        "pattern": r"^util\.chainbow(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.rainbow": {
        "description": "Colors residues in rainbow from N to C terminus",
        "pattern": r"^util\.rainbow(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.ss": {
        "description": "Colors by secondary structure",
        "pattern": r"^util\.ss(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.color_by_element": {
        "description": "Colors atoms by their element",
        "pattern": r"^util\.color_by_element(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "util.color_secondary": {
        "description": "Colors secondary structure elements",
        "pattern": r"^util\.color_secondary(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    
    # MOLECULAR DYNAMICS AND ANALYSIS
    "spheroid": {
        "description": "Displays atoms as smooth spheres",
        "pattern": r"^spheroid(?:\s+(.+))?$",
        "parameters": [
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "isomesh": {
        "description": "Creates a mesh isosurface",
        "pattern": r"^isomesh\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "name", "required": True},
            {"name": "map_object", "required": True},
            {"name": "level", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "isosurface": {
        "description": "Creates a solid isosurface",
        "pattern": r"^isosurface\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "name", "required": True},
            {"name": "map_object", "required": True},
            {"name": "level", "required": True},
            {"name": "selection", "required": False, "default": "all"}
        ],
        "check_selection": True
    },
    "sculpt_activate": {
        "description": "Activates sculpting mode for an object",
        "pattern": r"^sculpt_activate\s+(.+)$",
        "parameters": [
            {"name": "object", "required": True}
        ],
        "check_selection": False
    },
    "sculpt_deactivate": {
        "description": "Deactivates sculpting mode for an object",
        "pattern": r"^sculpt_deactivate\s+(.+)$",
        "parameters": [
            {"name": "object", "required": True}
        ],
        "check_selection": False
    },
    "sculpt_iterate": {
        "description": "Performs sculpting iterations",
        "pattern": r"^sculpt_iterate\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "iterations", "required": True},
            {"name": "object", "required": False, "default": "all"}
        ],
        "check_selection": False
    },
    
    # SCENES AND MOVIES
    "scene": {
        "description": "Manages scenes for later recall",
        "pattern": r"^scene\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "key", "required": True},
            {"name": "action", "required": False, "default": "recall"}
        ],
        "check_selection": False
    },
    "scene_order": {
        "description": "Sets the order of scenes",
        "pattern": r"^scene_order\s+(.+)$",
        "parameters": [
            {"name": "scene_list", "required": True}
        ],
        "check_selection": False
    },
    "mset": {
        "description": "Defines a sequence of states for movie playback",
        "pattern": r"^mset\s+(.+)$",
        "parameters": [
            {"name": "specification", "required": True}
        ],
        "check_selection": False
    },
    "mplay": {
        "description": "Starts playing the movie",
        "pattern": r"^mplay$",
        "parameters": [],
        "check_selection": False
    },
    "mstop": {
        "description": "Stops the movie",
        "pattern": r"^mstop$",
        "parameters": [],
        "check_selection": False
    },
    "frame": {
        "description": "Sets or queries the current frame",
        "pattern": r"^frame(?:\s+(.+))?$",
        "parameters": [
            {"name": "frame_number", "required": False}
        ],
        "check_selection": False
    },
    "forward": {
        "description": "Advances one frame",
        "pattern": r"^forward$",
        "parameters": [],
        "check_selection": False
    },
    "backward": {
        "description": "Goes back one frame",
        "pattern": r"^backward$",
        "parameters": [],
        "check_selection": False
    },
    "rock": {
        "description": "Toggles a rocking animation",
        "pattern": r"^rock$",
        "parameters": [],
        "check_selection": False
    },
    
    # RENDERING
    "ray": {
        "description": "Performs ray-tracing",
        "pattern": r"^ray(?:\s+([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "width", "required": False},
            {"name": "height", "required": False}
        ],
        "check_selection": False
    },
    "draw": {
        "description": "Uses OpenGL renderer (faster but lower quality)",
        "pattern": r"^draw(?:\s+([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "width", "required": False},
            {"name": "height", "required": False}
        ],
        "check_selection": False
    },
    "mpng": {
        "description": "Saves a series of PNG images for movie frames",
        "pattern": r"^mpng\s+(.+)$",
        "parameters": [
            {"name": "prefix", "required": True}
        ],
        "check_selection": False
    },
    
    # CRYSTALLOGRAPHY
    "symexp": {
        "description": "Generates symmetry-related copies",
        "pattern": r"^symexp\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "prefix", "required": True},
            {"name": "selection", "required": True},
            {"name": "cutoff", "required": False, "default": "20"},
            {"name": "segi", "required": False}
        ],
        "check_selection": True
    },
    "symexp": {
        "description": "Generates symmetry-related copies",
        "pattern": r"^symexp\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "prefix", "required": True},
            {"name": "selection", "required": True},
            {"name": "cutoff", "required": False, "default": "20"},
            {"name": "segi", "required": False}
        ],
        "check_selection": True
    },
    "set_symmetry": {
        "description": "Sets symmetry parameters for an object",
        "pattern": r"^set_symmetry\s+([^,]+)(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*([^,]+))?$",
        "parameters": [
            {"name": "selection", "required": True},
            {"name": "a", "required": True},
            {"name": "b", "required": True},
            {"name": "c", "required": True},
            {"name": "alpha", "required": True},
            {"name": "beta", "required": True},
            {"name": "gamma", "required": True}
        ],
        "check_selection": True
    },
    
    # OTHER
    "fab": {
        "description": "Creates a peptide chain from a sequence",
        "pattern": r"^fab\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "sequence", "required": True},
            {"name": "options", "required": False}
        ],
        "check_selection": False
    },
    "fragment": {
        "description": "Loads a molecular fragment",
        "pattern": r"^fragment\s+(.+)$",
        "parameters": [
            {"name": "name", "required": True}
        ],
        "check_selection": False
    },
    "full_screen": {
        "description": "Toggles fullscreen mode",
        "pattern": r"^full_screen$",
        "parameters": [],
        "check_selection": False
    },
    "viewport": {
        "description": "Sets the viewport size",
        "pattern": r"^viewport\s+([^,]+)(?:\s*,\s*(.+))?$",
        "parameters": [
            {"name": "width", "required": True},
            {"name": "height", "required": True}
        ],
        "check_selection": False
    },
    "cd": {
        "description": "Changes the current directory",
        "pattern": r"^cd\s+(.+)$",
        "parameters": [
            {"name": "path", "required": True}
        ],
        "check_selection": False
    },
    "pwd": {
        "description": "Prints the current directory",
        "pattern": r"^pwd$",
        "parameters": [],
        "check_selection": False
    },
    "ls": {
        "description": "Lists files in the current directory",
        "pattern": r"^ls(?:\s+(.+))?$",
        "parameters": [
            {"name": "path", "required": False}
        ],
        "check_selection": False
    },
    "system": {
        "description": "Executes a system command",
        "pattern": r"^system\s+(.+)$",
        "parameters": [
            {"name": "command", "required": True}
        ],
        "check_selection": False
    },
    "help": {
        "description": "Shows help for a command",
        "pattern": r"^help(?:\s+(.+))?$",
        "parameters": [
            {"name": "command", "required": False}
        ],
        "check_selection": False
    }
}


ERROR_PATTERNS = {
    "SYNTAX_ERROR": [
        r"Syntax error",
        r"invalid syntax",
        r"Unknown command"
    ],
    "SELECTION_ERROR": [
        r"Invalid selection",
        r"No atoms selected",
        r"Selection not found",
        r"Selection \S+ doesn't exist"
    ],
    "OBJECT_NOT_FOUND": [
        r"object \S+ not found",
        r"Object \S+ does not exist",
        r"Unable to find object named \S+"
    ],
    "ATOM_NOT_FOUND": [
        r"No atoms matched",
        r"No atoms in selection",
        r"Atom not found"
    ],
    "FILE_ERROR": [
        r"Unable to open file",
        r"No such file",
        r"Permission denied",
        r"Error reading file",
        r"Error writing file"
    ],
    "CONNECTION_ERROR": [
        r"Connection refused",
        r"Network error",
        r"Timeout",
        r"Failed to fetch"
    ],
    "PARAMETER_ERROR": [
        r"Incorrect number of parameters",
        r"Invalid parameter",
        r"Parameter out of range"
    ]
}
##############################################################################
# LOGGING
##############################################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyMOLMCPServer")

##############################################################################
# PYMOL SOCKET CONNECTION
##############################################################################

class PyMOLConnection:
    def __init__(self, host: str = 'localhost', port: int = 9876):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to PyMOL at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.sock = None
            return False

    def disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
            finally:
                self.sock = None

    def send_command(self, code: str) -> Dict[str, Any]:
        """
        Sends Python code to PyMOL via the socket plugin and returns a JSON response:
          { "status": "success" or "error",
            "result": {
               "executed": bool,
               "output": str or None,
               "error": str or None
            },
            "message": "error message string if any" }
        """
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to PyMOL")
        data = {"type": "pymol_command", "code": code}
        try:
            self.sock.sendall(json.dumps(data).encode('utf-8'))
            self.sock.settimeout(10.0)
            chunks = []
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                buffer = b''.join(chunks)
                try:
                    response = json.loads(buffer.decode('utf-8'))
                    return response
                except json.JSONDecodeError:
                    continue
            if chunks:
                buffer = b''.join(chunks)
                return json.loads(buffer.decode('utf-8'))
            raise ConnectionError("No response from PyMOL")
        except socket.timeout:
            self.sock = None
            raise TimeoutError("PyMOL response timed out")
        except Exception as e:
            self.sock = None
            raise RuntimeError(f"PyMOL command error: {e}")

_global_connection: Optional['PyMOLConnection'] = None

def get_pymol_connection() -> PyMOLConnection:
    global _global_connection
    if _global_connection is not None:
        try:
            # Test if connection is alive
            _global_connection.send_command("pass")
            return _global_connection
        except:
            try:
                _global_connection.disconnect()
            except:
                pass
            _global_connection = None
    if _global_connection is None:
        conn = PyMOLConnection()
        if not conn.connect():
            raise RuntimeError("Could not connect to PyMOL socket.")
        _global_connection = conn
    return _global_connection

##############################################################################
# PARSING USER INPUT TO PYMOL COMMANDS
##############################################################################

def parse_pymol_input(input_text: str) -> str:
    """
    Attempts to match the user input against known PYMOL_COMMANDS patterns.
    If matched, extracts parameters and builds the final Python code for PyMOL.
    Raises ValueError if no command matches or if there's a parameter error.
    """
    text_stripped = input_text.strip()
    for cmd_name, cmd_info in PYMOL_COMMANDS.items():
        pattern = re.compile(cmd_info["pattern"], re.IGNORECASE)
        match = pattern.match(text_stripped)
        if match:
            groups = match.groups()
            # Extract parameter definitions
            params_def = cmd_info["parameters"]
            param_values = {}
            for i, param_def in enumerate(params_def):
                param_name = param_def["name"]
                required = param_def.get("required", False)
                default_val = param_def.get("default", None)
                options = param_def.get("options", [])
                # Attempt to fetch from match group
                value = None
                if i < len(groups) and groups[i] is not None:
                    value = groups[i].strip()
                elif required and default_val is None:
                    raise ValueError(f"Missing required parameter '{param_name}' for command {cmd_name}")
                elif value is None and default_val is not None:
                    value = default_val
                if options and value and value not in options:
                    raise ValueError(f"Parameter '{param_name}' must be one of {options}")
                param_values[param_name] = value if value is not None else ""
            # (Optional) If check_selection is True, we could do extra checks
            # But for simplicity, just build PyMOL code
            return build_pymol_code(cmd_name, param_values)
    raise ValueError("No recognized PyMOL command pattern matched this input.")

def build_pymol_code(command_name: str, param_values: Dict[str, Any]) -> str:
    """
    Translates a recognized command plus parameters into Python code for PyMOL.
    This is a naive approach that constructs a single cmd.* invocation.
    Modify as needed for more complex logic.
    """
    # Example approach: "cmd.show('sticks', 'sele')"
    # For 'show' -> "cmd.show('sticks', 'all')"
    # This is simplified. Real approach might be more advanced.
    if command_name == "help":
        # We can do a special return for help
        cmd_obj = param_values.get("command") or ""
        if cmd_obj and cmd_obj in PYMOL_COMMANDS:
            return f"print('Help for {cmd_obj}: {PYMOL_COMMANDS[cmd_obj]['description']}')"
        return "print('List of PyMOL commands...')"

    # Generic pattern (the user can adapt this to each command's syntax)
    py_code = []
    py_code.append("from pymol import cmd")
    if command_name in ["util.cbc", "util.cbaw", "util.cbag", "util.cbac", "util.cbam",
                        "util.cbay", "util.cbas", "util.cbab", "util.cbao", "util.cbap",
                        "util.cbak", "util.chainbow", "util.rainbow", "util.ss",
                        "util.color_by_element", "util.color_secondary"]:
        # handle util.* style calls
        # e.g. "import util" doesn't exist in PyMOL by default. It's cmd.util.* typically
        selection = param_values.get("selection","all")
        # e.g. "cmd.util.chainbow('all')"
        # but realistically "util.chainbow(...)" might be "cmd.do('util.chainbow all')"
        call_code = f"cmd.do('{command_name} {selection}')"
        py_code.append(call_code)
        return "; ".join(py_code)

    # For typical direct 'cmd' calls:
    # We'll do a simple switch
    if command_name == "show":
        representation = param_values["representation"]
        selection = param_values["selection"]
        py_code.append(f"cmd.show('{representation}', '{selection}')")
    elif command_name == "hide":
        representation = param_values["representation"]
        selection = param_values["selection"]
        py_code.append(f"cmd.hide('{representation}', '{selection}')")
    elif command_name == "color":
        color_val = param_values["color"]
        selection = param_values["selection"]
        py_code.append(f"cmd.color('{color_val}', '{selection}')")
    else:
        # fallback, naive approach: "cmd.do('original command')"
        # build the original command as a string
        raw_cmd = command_name
        # We skip the prefix if it's something like "util."
        for k,v in param_values.items():
            if v.strip():
                raw_cmd += f" {v}"
        py_code.append(f"cmd.do('{raw_cmd}')")

    return "; ".join(py_code)

def analyze_pymol_output(output_text: str) -> Optional[str]:
    """
    Attempts to map known error patterns in the PyMOL output to a user-friendly error.
    Returns None if no known error patterns are matched.
    """
    lower_out = output_text.lower()
    for error_label, patterns in ERROR_PATTERNS.items():
        for p in patterns:
            if re.search(p.lower(), lower_out):
                return f"{error_label} detected: {p}"
    return None

##############################################################################
# MCP SERVER SETUP
##############################################################################

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    try:
        logger.info("Starting PyMOL MCP server (with command parsing).")
        try:
            get_pymol_connection()
        except Exception as e:
            logger.warning(f"Initial PyMOL connection failure: {e}")
        yield {}
    finally:
        global _global_connection
        if _global_connection:
            _global_connection.disconnect()
            _global_connection = None
        logger.info("PyMOL MCP server shut down.")

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="PyMOL MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = FastMCP("PyMOLMCPServer",
              description="PyMOL integration with advanced command parsing",
              lifespan=server_lifespan)

##############################################################################
# MCP TOOL: parse_and_execute
##############################################################################

@mcp.tool()
def parse_and_execute(ctx: Context, user_input: str) -> str:
    """
    Parses a text command against PYMOL_COMMANDS, builds PyMOL code, 
    executes it, and analyzes any error patterns in the output.
    """
    try:
        code = parse_pymol_input(user_input)
    except ValueError as ve:
        return f"No recognized PyMOL command or parameter issue: {ve}"
    except Exception as e:
        return f"Parsing error: {e}"

    try:
        conn = get_pymol_connection()
        response = conn.send_command(code)
        status = response.get("status", "error")
        if status == "success":
            res = response.get("result", {})
            out = res.get("output","") if isinstance(res, dict) else ""
            # If there's output, check known error patterns
            check_err = analyze_pymol_output(out)
            if check_err:
                return f"PyMOL command completed but possible error:\n{check_err}\nRaw Output:\n{out}"
            return out or "Command executed (no output)."
        else:
            msg = response.get("message","Unknown error")
            check_err = analyze_pymol_output(msg)
            if check_err:
                return f"Command failed: {check_err}"
            return f"Command error: {msg}"
    except Exception as e:
        return f"Execution error: {e}"

##############################################################################
# ENTRY POINT
##############################################################################

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
