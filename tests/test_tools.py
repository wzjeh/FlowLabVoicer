"""Smoke-test every tool in nagaki_lab.tools standalone (no Live API)."""
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nagaki_lab.config import MEMORY_DB_PATH
from nagaki_lab.tools import DISPATCH, init_caches


async def main():
    init_caches(MEMORY_DB_PATH)
    cases = [
        ("current_time", {}),
        ("tube_volume", {"inner_diameter_mm": 4.6, "length_mm": 250}),
        ("residence_time", {"length_cm": 50, "flow_rate_mL_per_min": 1.0}),
        ("residence_time", {"inner_diameter_mm": 0.5, "length_cm": 50,
                            "flow_rate_mL_per_min": 0.5}),
        ("chemical_info", {"name": "NaCl"}),
        ("chemical_info", {"name": "ethanol"}),
        ("solution_prep", {"chemical_name": "NaCl", "concentration": 0.5,
                           "concentration_unit": "M", "final_volume_mL": 250}),
        ("solution_prep", {"chemical_name": "glucose", "concentration": 5,
                           "concentration_unit": "%w/v", "final_volume_mL": 100}),
        ("set_timer", {"seconds": 2, "label": "test"}),
        ("list_active_timers", {}),
    ]
    for name, args in cases:
        print(f"\n--- {name}({json.dumps(args, ensure_ascii=False)}) ---")
        r = await DISPATCH[name](args)
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str)[:400])

    print("\n[waiting 3s for timer to fire]")
    await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
