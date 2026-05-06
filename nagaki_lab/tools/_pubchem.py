"""Internal PubChem REST client. Used by chemistry.py.

Not exposed as a tool; chemical_info / solution_prep call into here.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .. import config


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for it in node:
            yield from _walk(it)


def _parse_temperature(data: dict, heading: str) -> float | None:
    """Walk a PUG-View JSON; return the first temperature in °C for this heading."""
    pat = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b")
    for section in _walk(data):
        if not isinstance(section, dict):
            continue
        if section.get("TOCHeading") != heading:
            continue
        for info in section.get("Information", []) or []:
            value = info.get("Value", {}) or {}
            for sm in value.get("StringWithMarkup", []) or []:
                s = sm.get("String", "")
                m = pat.search(s)
                if not m:
                    continue
                v = float(m.group(1))
                if m.group(2) == "F":
                    v = (v - 32.0) * 5.0 / 9.0
                return round(v, 2)
    return None


async def fetch_chemical(name: str) -> dict[str, Any]:
    """Look up MW, formula, IUPAC name, melting point, boiling point on
    PubChem. Returns dict; on failure returns {'error': ...}."""
    base = config.PUBCHEM_BASE_URL
    async with httpx.AsyncClient(
        timeout=config.PUBCHEM_HTTP_TIMEOUT_S,
        headers={"User-Agent": "nagaki-lab/0.1"},
    ) as client:
        # 1. name -> CID
        try:
            r = await client.get(f"{base}/pug/compound/name/{name}/cids/JSON")
            if r.status_code == 404:
                return {"error": f"PubChem has no compound named '{name}'"}
            r.raise_for_status()
            cids = r.json()["IdentifierList"]["CID"]
            if not cids:
                return {"error": f"PubChem has no compound named '{name}'"}
            cid = int(cids[0])
        except Exception as e:
            return {"error": f"name resolution failed: {e}"}

        # 2. MW + formula + IUPAC name
        try:
            r = await client.get(
                f"{base}/pug/compound/cid/{cid}/property/"
                "MolecularWeight,MolecularFormula,IUPACName/JSON"
            )
            r.raise_for_status()
            props = r.json()["PropertyTable"]["Properties"][0]
            mw = float(props["MolecularWeight"])
            formula = props.get("MolecularFormula")
            iupac = props.get("IUPACName")
        except Exception as e:
            return {"error": f"property fetch failed for CID {cid}: {e}"}

        # 3. melting / boiling point via PUG-View
        mp_c = bp_c = None
        for heading, slot in (("Melting Point", "mp"), ("Boiling Point", "bp")):
            try:
                r = await client.get(
                    f"{base}/pug_view/data/compound/{cid}/JSON",
                    params={"heading": heading},
                )
                if r.status_code == 200:
                    val = _parse_temperature(r.json(), heading)
                    if slot == "mp":
                        mp_c = val
                    else:
                        bp_c = val
            except Exception:
                pass

        return {
            "name": name,
            "cid": cid,
            "molecular_weight_g_per_mol": mw,
            "molecular_formula": formula,
            "iupac_name": iupac,
            "melting_point_C": mp_c,
            "boiling_point_C": bp_c,
            "source": "PubChem",
        }
