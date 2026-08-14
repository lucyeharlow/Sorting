"""
ERA5 Pressure-Level Cloud Variable Download — Southern Africa
=============================================================
Downloads monthly mean cloud fraction, cloud ice and cloud liquid water
on pressure levels from CDS for comparison with SCM cloud profiles.

Variables:
  - fraction_of_cloud_cover          (0-1)
  - specific_cloud_ice_water_content (kg/kg)
  - specific_cloud_liquid_water_content (kg/kg)

Pressure levels: standard set covering 1000–1 hPa (37 levels)
Domain: full southern Africa  [N=-10, W=10, S=-35, E=40]
"""

import cdsapi
import os
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


OUT_DIR = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/era5_cloud_profiles_world"
os.makedirs(OUT_DIR, exist_ok=True)

# Full world domain 
AREA = [90, -180, -90, 180]

YEARS  = [str(y) for y in range(1979, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Standard ERA5 pressure levels (hPa) — full set for vertical profiles
PRESSURE_LEVELS = [
    '1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
    '100', '125', '150', '175', '200', '225', '250', '300',
    '350', '400', '450', '500', '550', '600', '650', '700',
    '750', '775', '800', '825', '850', '875', '900', '925',
    '950', '975', '1000'
]

CLOUD_VARIABLES = {
    "fraction_of_cloud_cover":             "cloud_fraction",
    "specific_cloud_ice_water_content":    "cloud_ice",
    "specific_cloud_liquid_water_content": "cloud_liquid",
}

# Download
log("=== ERA5 Pressure-Level Cloud Download — Full Southern Africa ===")
log(f"Output: {OUT_DIR}")
log(f"Area: {AREA}  (N={AREA[0]}, W={AREA[1]}, S={AREA[2]}, E={AREA[3]})")
log(f"Years: {YEARS[0]} – {YEARS[-1]}  |  All 12 months")
log(f"Pressure levels: {len(PRESSURE_LEVELS)} levels "
    f"from {PRESSURE_LEVELS[-1]} to {PRESSURE_LEVELS[0]} hPa")

c = cdsapi.Client()

for cds_name, short_name in CLOUD_VARIABLES.items():
    out_file = os.path.join(OUT_DIR,
                            f"era5_monthly_{short_name}_pressure_levels_global.nc")
    log(f"\n--- {cds_name} ---")
    if os.path.exists(out_file):
        size_mb = os.path.getsize(out_file) / 1e6
        log(f"  SKIP — already exists ({size_mb:.1f} MB): {out_file}")
        continue

    log(f"  Downloading to: {out_file}")
    t0 = datetime.now()
    try:
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable":     cds_name,
                "pressure_level": PRESSURE_LEVELS,
                "year":         YEARS,
                "month":        MONTHS,
                "time":         "00:00",
                "area":         AREA,
                "format":       "netcdf",
            },
            out_file,
        )
        elapsed = (datetime.now() - t0).seconds / 60
        size_mb  = os.path.getsize(out_file) / 1e6
        log(f"  DONE — {size_mb:.1f} MB in {elapsed:.1f} min")
    except Exception as e:
        log(f"  FAILED: {e}")
        continue

log("\n=== Download Complete ===")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".nc"):
        size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
        log(f"  {f}: {size_mb:.1f} MB")
