"""
combined_cloud_fraction_trends.py

Combined 3-row x 4-column cloud fraction OLS trend figure.

Rows    : ERA5 (top) | MERRA-2 (middle) | CLARA-A3 (bottom)
Columns : Total | High | Middle | Low

All datasets are on the same MERRA-2 grid (produced by
preprocess_cloud_fraction.py), so a single shared colourbar
can be used across all 12 panels.

Shared colour scale : symmetric, based on the 99th percentile of
                      |trend| across ALL datasets and variables.

Projection          : PlateCarree with lon/lat gridlines and labels.
Significance        : FDR-corrected (Benjamini-Hochberg, alpha=0.10).
                      Grey = fails FDR; coloured = significant.
OLS                 : lag-1 autocorrelation effective-N correction.
Units               : % decade⁻¹
"""

# Packages 
from __future__ import annotations
import os
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# Paths
data_dir = Path(
    "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/combined_cloud_fraction"
)
out_dir= Path(
    "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis"
    "/1979_2025_plots/combined_cloud_fraction"
)
out_dir.mkdir(parents=True, exist_ok=True)

era5   = data_dir / "era5_cf_october_merra2grid.nc"
merra2 = data_dir / "merra2_cf_october.nc"
clara  = data_dir / "clara_cf_october_merra2grid.nc"

# Constants for trend computation and plotting
fdr_alpha = 0.10
decade    = 10

# Layout for 3x4 figure
"""
CLARA_A3 and MERRA-2 divide their cloud fraction datasets into high, middle and low bins (slightly different definitions, out by 50hPa)
ERA5 does not, so it has been subdivided during the preprocessing stage. 
"""
columns = [
    ("cf_total", "Total cloud fraction"),
    ("cf_high",  "High cloud fraction\n(<400 hPa)"),
    ("cf_mid",   "Middle cloud fraction\n(400–700 hPa)"),
    ("cf_low",   "Low cloud fraction\n(>700 hPa)"),
]

"""
CLARA-A3 years to be used: 1982-2020 (for initial analysis, but can be extended to 1982-2025 if needed) 
Years should be consistent across datasets for comparison
"""
rows = [
    ("era5",   "1979–2025"),
    ("merra2", "1980–2025"),
    ("clara",  "1979–2020"),
]

# Font settings and figure DPI for crisp formatting 
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "figure.dpi":      150,
})


# Sbatch logging
def log(msg: str) -> None:
    print(msg, flush=True)


# Statistics: OLS with lag-1 autocorrelation effective-N correction
"""
Lag -1 autocorrelation function:
* Checks autocorrelation for a 1D time series using shifted arrays- checks whether a value at time t is correlated with the value at time t-1
* Checks the number of valid points are sufficient for further analysis (at least 3 valid points)
* Results in a correlation coefficient between -1 and 1

"""
def lag1_autocorr(ts: np.ndarray) -> float: 
    ts = np.asarray(ts, dtype=float) #Converts numpy array to float
    ok = np.isfinite(ts) #Keeps finite values - this is checking for NaN values in the time series
    if ok.sum() < 3:
        return np.nan # If there are fewer than 3 valid point, then NaN is returned and no further calculations are done 
    t0, t1 = ts[:-1], ts[1:] # Begins autocorr -> shifts timestep by one time step to create two arrays, t0 and t1
    m = np.isfinite(t0) & np.isfinite(t1) #Keeps only the valid points in both arrays, t0 and t1 and masks NaNs
    return float(np.corrcoef(t1[m], t0[m])[0, 1]) if m.sum() >= 3 else np.nan #Does the correlation calculation coor (x_t-1, x_t)


def ols_pixel(ts: np.ndarray, years: np.ndarray) -> tuple[float, float]:
    """
    Ordinary least squares (OLS) linear regression with lag-1 autocorrelation and effective-N correction for a single pixel time series.
    Returns the slope and p-value for further trend analysis. Additional statistics are returned for debugging or analysis purposes.
    """
    y  = np.asarray(ts, dtype=float) # Converts timeseries (ts) to float
    x  = np.asarray(years, dtype=float) # Converts years ot floats
    ok = np.isfinite(y) & np.isfinite(x) # Keeps only the valid points in both arrays, y and x and masks NaNs
    n  = ok.sum()
    if n < 8: # But if there are less than 8 valid points, then the function returns NaN for both slope and p-value
        return np.nan, np.nan
    y, x = y[ok], x[ok] # Keeps only the valid points in both arrays, y and x and masks NaNs
    x = x - x.mean() # Centers the x values around zero by subtracting the mean of x from each value in x. This helps to reduce numerical errors in the regression calculation
    slope, intercept = np.polyfit(x, y, 1) # Fits a linear regression model to the data using numpy's polyfit function. 
    resid  = y - (slope * x + intercept) # Calculates the residuals (the difference between the observed y values and the predicted y values based on the fitted line)
    sigma2 = np.sum(resid ** 2) / max(n - 2, 1) # Calculates the variance of the residuals (sigma^2) by summing the squared residuals and dividing by the degrees of freedom (n - 2). The max function ensures that the denominator is at least 1 to avoid division by zero.
    sxx    = np.sum(x ** 2) # Calculates the sum of squares of the centered x values (sxx) by summing the squared centered x values. This is used in the calculation of the standard error of the slope.
    if sxx == 0: # Checks that if sum of squares of the centered x values is zero, which would indicate that all x values are the same (no variation in x), then the function returns NaN for both slope and p-value since a trend cannot be computed.
        return np.nan, np.nan
    se   = np.sqrt(sigma2 / sxx) # Computes standard error of the slope (se) by taking the square root of the variance of the residuals divided by the sum of squares of the centered x values. This gives an estimate of the uncertainity in the slope estimate. 
    rho1 = lag1_autocorr(resid) # Computes the lag -1 autocorrelation of the residuals using above function 
    if np.isnan(rho1): # Checks that if the lag-1 autocorrelation is NaN (not a number), which would indicate that there are not enough valid points to compute the autocorrelation, then the effective sample size (neff) is set to the number of valid points (n).
        neff = float(n) 
    else:
        denom = 1.0 + rho1 # Computes the denominator for the effective sample size calculation by adding 1 to the lag-1 autocorrelation. This accounts for the fact that the residuals are not independent due to autocorrelation.
        neff  = float(n) * (1.0 - rho1) / denom if denom > 1e-6 else float(n) # Computes the effective sample size (neff) by adjusting the number of valid points (n) based on the lag-1 autocorrelation. If the denominator is very small (less than 1e-6), then neff is set to n to avoid division by zero or numerical instability.
        neff  = max(neff, 3.0) # Ensures that the effective sample size is at least 3 to avoid issues with statistical tests.
    se_adj = se * np.sqrt(float(n) / neff) # Adjusts standard error by multiplying it by the square root of the ratio of the number of valid points (n) to the effective sample size (neff). This accounts for the reduced degrees of freedom due to autocorrelation.
    df     = max(int(np.floor(neff - 2)), 1) # Computes the degree of freedom for the t test, by subtracting 2 from the effective sample size. The max function ensurse the df are at least 1
    tstat  = slope / se_adj if se_adj > 0 else np.nan # Computes the t statistic for the slope by dividing the slope by the adjusted standard error
    pval   = 2.0 * tdist.sf(abs(tstat), df=df) # Computes the two-sided p-value for the t statistic using the survival function (sf) of the t distribution 
    return float(slope), float(pval), float(df), float(se_adj), float(neff), float(rho1)


# Vectorised ols computation using above function 
"""
Above is the vectorised implementation of the OLS function, which applies the _ols function to each pixel 
in the input data array (da) using xarray's apply_ufunc method. The input_core_dims argument specifies that the "year" dimension 
is the core dimension for the input data array, and the output_core_dims argument specifies that there are no core dimensions for the output arrays
(slope and pval). The vectorize argument is set to True to enable vectorization, and the output_dtypes argument 
specifies that the output arrays should be of type float. Finally, the slope values are multiplied by decade to convert them to % decade⁻¹ units.
"""
def compute_ols_field(
    da: xr.DataArray, years: np.ndarray # Outlines the input data array and year numpy array for OLS calc
) -> tuple[np.ndarray, np.ndarray]: # and converts the output to a tuple of numpy arrays for slope and p-value
    years_f = years.astype(float) 
    def _ols(ts: np.ndarray): # Defines a nested function _ols that takes a 1D numpy array ts (time series) as input and returns the slope and p-value for that time series using the ols_pixel function defined above. The years_f array is passed to ols_pixel to provide the corresponding years for the time series.
        return ols_pixel(ts, years_f)

    slope, pval = xr.apply_ufunc(
        _ols, da,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float],
    )
    return (slope.values * decade), pval.values



# FDR correction (significance mask))
"""
Uses the Benjamini-Hochberg procedure to control the false discovery rate (FDR) for multiple hypothesis testing.
The function takes a 2D array of p-values (pval_2d) and an alpha level (default 0.10) as input, and returns a boolean mask indicating 
which p-values are significant after FDR correction. The p-values are flattened into a 1D array, and the multipletests function from 
statsmodels is used to perform the FDR correction.
"""
def apply_fdr(pval_2d: np.ndarray, alpha: float = fdr_alpha) -> np.ndarray: # Ensure pval numpy array and float alpha level are converted to a numpy array
    p      = pval_2d.flatten() # Flattens 2D array of p-values into a 1D array for processing
    finite = np.isfinite(p) # Creates a boolean mask for checking whether data is missing (NaN values etc.)
    reject = np.zeros(p.shape, dtype=bool)
    if finite.sum() > 0: # Actual fdr correction is only applied if there are finite p-values to process
        rej, *_ = multipletests(p[finite], alpha=alpha, method="fdr_bh") # Uses multiple tests to apply FDR correction and work out which ones are significant
        reject[finite] = rej
    return reject.reshape(pval_2d.shape) # Resize the boolean mask to original pval shape


# Load datasets and run trend computation
def load_and_compute(
    filepath: Path,
    dataset_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    log(f"  Loading {dataset_name} from {filepath.name} ...")
    ds   = xr.open_dataset(filepath)
    yrs  = ds["year"].values.astype(int)
    results: dict[str, dict[str, np.ndarray]] = {}

    for var, _ in columns:
        if var not in ds:
            log(f"{var} not found in {dataset_name}, skipping")
            continue
        log(f"{dataset_name}: computing ols for {var} ...")
        da    = ds[var]
        slope, pval = compute_ols_field(da, yrs)
        sig   = apply_fdr(pval)
        results[var] = {
            "slope": slope,
            "pval":  pval,
            "sig":   sig,
        }
        log(f"Range: {np.nanmin(slope):.3f} – {np.nanmax(slope):.3f} %/dec")

    ds.close()
    return results


# Plotting features 
def _add_features(ax: plt.Axes) -> None:
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), zorder=1, facecolor="white")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), zorder=1, facecolor="white")
    ax.coastlines(resolution="110m", linewidth=0.45, zorder=5)
    ax.set_global()


def _add_gridlines(
    ax: plt.Axes,
    left_labels: bool = False,
    bottom_labels: bool = False,
) -> None:
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color="grey",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlocator      = mplticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator      = mplticker.FixedLocator([-60, -30, 0, 30, 60])
    gl.xformatter    = LongitudeFormatter()
    gl.yformatter    = LatitudeFormatter()
    gl.xlabel_style  = {"size": 6, "color": "black", "rotation": 40}
    gl.ylabel_style  = {"size": 6, "color": "black"}


def _plot_panel(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    slope: np.ndarray,
    sig: np.ndarray,
    norm: TwoSlopeNorm,
    left_labels: bool,
    bottom_labels: bool,
) -> None:
    ax.contourf(
        lons, lats, slope,
        levels=np.linspace(norm.vmin, norm.vmax, 21),
        colors="#CCCCCC",
        antialiased=False,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    slope_sig = np.where(sig, slope, np.nan)
    levels    = np.linspace(norm.vmin, norm.vmax, 21)
    ax.contourf(
        lons, lats, slope_sig,
        levels=levels,
        cmap="RdBu_r",
        norm=norm,
        extend="both",
        antialiased=False,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    _add_features(ax)
    _add_gridlines(ax, left_labels=left_labels, bottom_labels=bottom_labels)


# Main function
def main() -> None:
    log("Combined Cloud Fraction Trend Figure")
    for f in (era5, merra2, clara):
        if not f.exists():
            raise FileNotFoundError(
                f"Input file not found: {f}\n"
                "Run preprocess_cloud_fraction.py first."
            )

    with xr.open_dataset(merra2) as _ds:
        lats = _ds["lat"].values
        lons = _ds["lon"].values

    log("\nComputing OLS trends ...")
    all_results = {
        "era5":   load_and_compute(era5,   "ERA5"),
        "merra2": load_and_compute(merra2, "MERRA-2"),
        "clara":  load_and_compute(clara,  "CLARA-A3"),
    }

    log("\nComputing shared colour scale ...")
    all_sig_slopes: list[np.ndarray] = []
    for ds_name, res in all_results.items():
        for var, _ in columns:
            if var not in res:
                continue
            arr = res[var]["slope"]
            sig = res[var]["sig"]
            sig_vals = np.abs(arr[sig & np.isfinite(arr)])
            if len(sig_vals):
                all_sig_slopes.append(sig_vals)

    if all_sig_slopes:
        combined = np.concatenate(all_sig_slopes)
        clim = float(np.nanpercentile(combined, 99))
        clim = max(clim, 0.5)
        clim = np.ceil(clim * 10) / 10
    else:
        clim = 5.0

    log(f"  Shared colour scale: ±{clim:.2f} % decade⁻¹")
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    log("\nBuilding figure ...")

    nrows, ncols = 3, 4
    PROJ = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(22, 12),
        subplot_kw={"projection": PROJ},
    )
    fig.subplots_adjust(
        left=0.04, right=0.96,
        top=0.90,  bottom=0.10,
        wspace=0.06, hspace=0.28,
    )

    for ri, (row_label, ds_key, period) in enumerate(rows):
        res = all_results[ds_key]
        for ci, (var, col_title) in enumerate(columns):
            ax = axes[ri, ci]

            left_labels   = (ci == 0)
            bottom_labels = (ri == nrows - 1)

            if var not in res:
                ax.set_visible(False)
                continue

            slope = res[var]["slope"]
            sig   = res[var]["sig"]

            _plot_panel(
                ax, lons, lats, slope, sig, norm,
                left_labels=left_labels,
                bottom_labels=bottom_labels,
            )

            if ri == 0:
                ax.set_title(col_title, fontsize=9, fontweight="bold", pad=6)

            if ci == 0:
                ax.text(
                    -0.14, 0.5,
                    f"{row_label}\n{period}",
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    va="center", ha="center",
                    rotation=90,
                )

            n_sig   = sig.sum()
            n_total = np.isfinite(slope).sum()
            frac    = n_sig / n_total * 100 if n_total > 0 else 0
            ax.text(
                0.98, 0.02,
                f"{frac:.0f}% sig",
                transform=ax.transAxes,
                fontsize=6, ha="right", va="bottom",
                color="black",
                bbox=dict(facecolor="white", alpha=0.6,
                          edgecolor="none", pad=1.5),
                zorder=10,
            )

    # Shared colourbar 
    cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.020])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(
        "Cloud fraction trend (% decade⁻¹)",
        fontsize=11,
    )
    cbar.ax.tick_params(labelsize=10)

    # Title
    fig.suptitle(
        "October cloud fraction OLS trends  |  Grey = fails FDR correction",
        fontsize=16, fontweight="bold", y=0.975,
    )

    outfile = out_dir / "combined_cloud_fraction_trends_3x4.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"\nSaved → {outfile}")
    log("Done.")


if __name__ == "__main__":
    main()
