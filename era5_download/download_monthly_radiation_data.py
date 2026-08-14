import cdsapi
import os
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


SCRATCH = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025"
os.makedirs(SCRATCH, exist_ok=True)

VARIABLES = [
    'surface_net_thermal_radiation',
    'surface_net_solar_radiation',
    'surface_thermal_radiation_downwards',
    'surface_solar_radiation_downwards',
    'surface_sensible_heat_flux',
    'surface_latent_heat_flux',
]

DOMAINS = {
    "global" : {
        "area": [90, -180, -90, 180], # full globe 
    }
}

# Adjust to your desired analysis period
YEARS = [str(y) for y in range(1979, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


log("=== ERA5 Flux Download Job Started ===")
log(f"Output directory: {SCRATCH}")
log(f"Variables: {VARIABLES}")
log(f"Years: {YEARS[0]} - {YEARS[-1]}")
log(f"Domains: {list(DOMAINS.keys())}")

c = cdsapi.Client()

total = len(DOMAINS) * len(VARIABLES)
count = 0

for domain_name, domain in DOMAINS.items():
    for var in VARIABLES:
        count += 1
        out_file = os.path.join(SCRATCH, f"{domain_name}_{var}.nc")

        log(f"[{count}/{total}] Checking: {domain_name} / {var}")

        if os.path.exists(out_file):
            size_mb = os.path.getsize(out_file) / 1e6
            log(f" SKIP — already exists ({size_mb:.1f} MB): {out_file}")
            continue

        log(f" START downloading {var} for {domain_name}...")
        t0 = datetime.now()

        try:
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': var,
                    'year': YEARS,
                    'month': MONTHS,
                    'time': '00:00',
                    'area': domain['area'],
                    'format': 'netcdf',
                },
                out_file
            )
            elapsed = (datetime.now() - t0).seconds / 60
            size_mb = os.path.getsize(out_file) / 1e6
            log(f"  **DONE** — {size_mb:.1f} MB in {elapsed:.1f} min: {out_file}")

        except Exception as e:
            log(f"  FAILED — {var} / {domain_name}: {e}")
            # Continue to next variable rather than crashing the whole job
            continue

log("=== ERA5 Flux Download Job Finished ===")
log(f"Completed {count} variable/domain combinations")

# Print summary of what's in the output directory
log("=== Output Summary ===")
for f in sorted(os.listdir(SCRATCH)):
    if f.endswith('.nc'):
        full_path = os.path.join(SCRATCH, f)
        size_mb = os.path.getsize(full_path) / 1e6
        log(f"  {f}: {size_mb:.1f} MB")
