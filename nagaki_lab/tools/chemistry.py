"""Chemistry tools for flow chemistry: tube geometry, residence time,
chemical lookup, solution preparation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from google.genai import types

from . import _pubchem


# ---------- shared cache db path (set by tools.init_caches) ----------
_cache_db: Optional[Path] = None


def init_cache(db_path: Path) -> None:
    global _cache_db
    _cache_db = Path(db_path)


async def _ensure_chem_cache_table() -> None:
    if _cache_db is None:
        return
    async with aiosqlite.connect(_cache_db) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chem_cache (
                name TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


# ---------- tube_volume ----------
async def tube_volume(args: dict) -> dict:
    d_mm = float(args["inner_diameter_mm"])
    l_mm = float(args["length_mm"])
    if d_mm <= 0 or l_mm <= 0:
        return {"error": "inner_diameter_mm and length_mm must be positive"}
    r_mm = d_mm / 2.0
    vol_mm3 = math.pi * r_mm * r_mm * l_mm   # 1 mm^3 == 1 uL
    return {
        "inner_diameter_mm": d_mm,
        "length_mm": l_mm,
        "volume_uL": round(vol_mm3, 4),
        "volume_mL": round(vol_mm3 / 1_000.0, 6),
        "volume_L": round(vol_mm3 / 1_000_000.0, 9),
        "formula": "V = pi * (d/2)^2 * L",
    }


# ---------- residence_time ----------
async def residence_time(args: dict) -> dict:
    d_mm = float(args.get("inner_diameter_mm", 1.0))
    l_cm = float(args["length_cm"])
    flow_mL_min = float(args["flow_rate_mL_per_min"])
    if d_mm <= 0 or l_cm <= 0 or flow_mL_min <= 0:
        return {"error": "all inputs must be positive"}
    l_mm = l_cm * 10.0
    r_mm = d_mm / 2.0
    vol_mm3 = math.pi * r_mm * r_mm * l_mm
    vol_mL = vol_mm3 / 1_000.0
    flow_mL_s = flow_mL_min / 60.0
    tau_s = vol_mL / flow_mL_s
    return {
        "inner_diameter_mm": d_mm,
        "length_cm": l_cm,
        "flow_rate_mL_per_min": flow_mL_min,
        "tube_volume_uL": round(vol_mm3, 3),
        "tube_volume_mL": round(vol_mL, 6),
        "residence_time_s": round(tau_s, 3),
        "residence_time_min": round(tau_s / 60.0, 4),
        "calculation": (
            f"V = pi*(d/2)^2*L = pi*({d_mm}/2)^2 * {l_mm} mm = {vol_mm3:.3f} uL "
            f"= {vol_mL:.4f} mL; tau = V/Q = {vol_mL:.4f}/( {flow_mL_min}/60 ) = {tau_s:.3f} s"
        ),
    }


# ---------- chemical_info (PubChem-backed, cached) ----------
async def chemical_info(args: dict) -> dict:
    name = str(args["name"]).strip()
    if not name:
        return {"error": "empty name"}

    if _cache_db is None:
        return await _pubchem.fetch_chemical(name)

    await _ensure_chem_cache_table()
    key = name.lower()
    async with aiosqlite.connect(_cache_db) as db:
        async with db.execute("SELECT data FROM chem_cache WHERE name=?", (key,)) as cur:
            row = await cur.fetchone()
        if row:
            data = json.loads(row[0])
            data["_cache"] = "hit"
            return data
        result = await _pubchem.fetch_chemical(name)
        if "error" not in result:
            await db.execute(
                "INSERT OR REPLACE INTO chem_cache (name, data) VALUES (?, ?)",
                (key, json.dumps(result)),
            )
            await db.commit()
        return result


# ---------- solution_prep ----------
async def solution_prep(args: dict) -> dict:
    chem = str(args["chemical_name"]).strip()
    conc = float(args["concentration"])
    unit = str(args["concentration_unit"]).strip()
    vol_mL = float(args["final_volume_mL"])

    if conc <= 0 or vol_mL <= 0:
        return {"error": "concentration and final_volume_mL must be positive"}

    info = await chemical_info({"name": chem})
    if "error" in info:
        return {"error": f"could not look up {chem}: {info['error']}"}

    mw = float(info["molecular_weight_g_per_mol"])
    vol_L = vol_mL / 1000.0

    if unit == "M":
        moles = conc * vol_L
        mass_g = mw * moles
        calc = f"mass = MW × C × V = {mw:.3f} × {conc} × {vol_L} L"
    elif unit == "mM":
        moles = conc * 1e-3 * vol_L
        mass_g = mw * moles
        calc = f"mass = MW × C × V = {mw:.3f} × {conc/1000:g} × {vol_L} L"
    elif unit == "uM":
        moles = conc * 1e-6 * vol_L
        mass_g = mw * moles
        calc = f"mass = MW × C × V = {mw:.3f} × {conc*1e-6:g} × {vol_L} L"
    elif unit in ("%w/v", "%(w/v)", "%"):
        mass_g = (conc / 100.0) * vol_mL
        moles = mass_g / mw
        calc = f"mass = (C/100) × V_mL = ({conc}/100) × {vol_mL} mL"
    elif unit == "g/L":
        mass_g = conc * vol_L
        moles = mass_g / mw
        calc = f"mass = C × V = {conc} g/L × {vol_L} L"
    elif unit in ("mg/mL", "mg/ml"):
        mass_g = (conc / 1000.0) * vol_mL
        moles = mass_g / mw
        calc = f"mass = (C × V_mL) / 1000 = ({conc} × {vol_mL}) / 1000"
    else:
        return {"error": f"unsupported unit '{unit}'. Use M / mM / uM / %w/v / g/L / mg/mL"}

    return {
        "chemical": chem,
        "molecular_weight_g_per_mol": mw,
        "molecular_formula": info.get("molecular_formula"),
        "target_concentration": f"{conc} {unit}",
        "final_volume_mL": vol_mL,
        "mass_to_weigh_g": round(mass_g, 4),
        "mass_to_weigh_mg": round(mass_g * 1000.0, 2),
        "moles": round(moles, 6),
        "calculation": calc,
        "procedure": (
            f"Weigh {round(mass_g, 4)} g of {chem} (MW = {mw:.2f}). "
            f"Dissolve in less than {vol_mL} mL of solvent, then bring the total "
            f"volume to {vol_mL} mL with solvent."
        ),
    }


# ---------- declarations + dispatch ----------
S = types.Schema
T = types.Type

DECLARATIONS = [
    types.FunctionDeclaration(
        name="tube_volume",
        description=("Compute the internal volume of a cylindrical tube, column, "
                     "or capillary, given inner diameter and length, both in mm. "
                     "Returns volume in microliters, milliliters, and liters."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "inner_diameter_mm": S(type=T.NUMBER,
                                                description="Inner diameter in mm. Convert from cm/inch first."),
                         "length_mm": S(type=T.NUMBER, description="Length in mm."),
                     },
                     required=["inner_diameter_mm", "length_mm"]),
    ),
    types.FunctionDeclaration(
        name="residence_time",
        description=("Mean residence time tau = V/Q in a flow reactor or HPLC line. "
                     "Inputs: inner_diameter_mm (default 1.0 if user does not say), "
                     "length_cm, flow_rate_mL_per_min. Convert other units first."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "inner_diameter_mm": S(type=T.NUMBER,
                                                description="Inner diameter in mm. Default 1.0 if unspecified."),
                         "length_cm": S(type=T.NUMBER,
                                        description="Tube length in centimeters."),
                         "flow_rate_mL_per_min": S(type=T.NUMBER,
                                                   description="Flow rate in mL/min."),
                     },
                     required=["length_cm", "flow_rate_mL_per_min"]),
    ),
    types.FunctionDeclaration(
        name="chemical_info",
        description=("Look up molecular weight, formula, melting point, and "
                     "boiling point on PubChem. Use English / IUPAC name."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "name": S(type=T.STRING,
                                   description="Chemical name, formula, or synonym in English."),
                     },
                     required=["name"]),
    ),
    types.FunctionDeclaration(
        name="solution_prep",
        description=("Calculate mass of solute to weigh out for a target "
                     "concentration and final volume. Looks up MW automatically."),
        parameters=S(type=T.OBJECT,
                     properties={
                         "chemical_name": S(type=T.STRING),
                         "concentration": S(type=T.NUMBER),
                         "concentration_unit": S(type=T.STRING,
                                                 description="One of: M, mM, uM, %w/v, g/L, mg/mL"),
                         "final_volume_mL": S(type=T.NUMBER),
                     },
                     required=["chemical_name", "concentration",
                               "concentration_unit", "final_volume_mL"]),
    ),
]

DISPATCH = {
    "tube_volume": tube_volume,
    "residence_time": residence_time,
    "chemical_info": chemical_info,
    "solution_prep": solution_prep,
}
