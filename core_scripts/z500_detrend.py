import argparse
import xarray as xr
import numpy as np
import os
import subprocess


g = 9.80665

def parse_args():
    parser = argparse.ArgumentParser(description = "detrending of z500, removal of the stationary seasonal cycle based on ERA5")
    
    parser.add_argument("--model", required = True, 
                        help = 'name of model simulation to load as input, e.g. CNRM-CM6-1')
    parser.add_argument("--experiment", required=True, 
                        help = 'name of simulation experiment, e.g. historical, ssp370')
    parser.add_argument("--member", required=True, 
                        help = 'Which ensemble member, if any')

# Optional arguments
    parser.add_argument("--basedir", default="/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/raw", 
                        help = 'root directory')
    parser.add_argument("--varname", default="zg", 
                        help = 'name of the variable to detrend')
    parser.add_argument("--era5_cycle", default="/Data/skd/projects/global/cmip6_precursors/aux/cycles/ERA5.nc", 
                        help = 'file with the ERA5 seasonal cycle')
    parser.add_argument('--latmin', type=float, default=30, 
                        help='min lat')
    parser.add_argument('--latmax', type=float, default=85, 
                        help='max lat')

    return parser.parse_args()


def main(args):
    
    input = f"{args.basedir}/{args.model}/z500/{args.experiment}/"
    file=[files for files in os.listdir(input) if files.endswith('.nc')]
    if len(file) == 1:
        file = file[0]
    
    input_pattern = f'{input}/{file}'
    date_range = file.split('_')[6]
    date_start, date_end = date_range.split('-')
    date_end = date_end.split('.')[0]
    date_start = f"{date_start[:4]}-{date_start[4:6]}-{date_start[6:]}"
    date_end = f"{date_end[:4]}-{date_end[4:6]}-{date_end[6:]}"
    if not input_pattern :
        raise FileNotFoundError(f"No files found in : {input/input_pattern}")

    ds = xr.open_dataset(input_pattern)
    x = ds[args.varname]
    x.load()
    x = x.sel(time = slice(date_start, date_end))

    x_detrended = detrend_seasonal_cycle(x, args.era5_cycle, args.latmin, args.latmax)
    if args.varname.split :
        x_detrended = x_detrended.rename(f'{args.varname}_detrend')
    else :
        x_detrended = x_detrended.rename(f'{args.varname}')

    output_pattern = f"{args.basedir}/{args.model}/z500_detrend/{args.experiment}/z500_detrend_day_{args.model}_{args.experiment}_{args.member}_gn_{date_start}-{date_end}.nc"
    os.makedirs(os.path.dirname(output_pattern), exist_ok=True)
    subprocess.run(['chmod','-R','g+wrx',os.path.dirname(output_pattern)])
    x_detrended.to_netcdf(output_pattern)




def detrend_seasonal_cycle(x, era5_cycle, latmin, latmax):
    # Subset latitudes
    x = x.sel(lat = slice(latmin, latmax))

    # Weighted mean lat/lon
    weights = np.cos(np.deg2rad(x.lat))
    xmean = x.weighted(weights).mean(('lat', 'lon'))

    # Group by (years, months)
    xmean = xmean.assign_coords(year = xmean['time.year'], month = xmean['time.month']).set_index(time = ('year', 'month'))
    monthly_x = xmean.groupby('time').mean().unstack('time')

    if 'plev' in monthly_x.coords:
        monthly_x = monthly_x.drop_vars('plev')

    # Gaussian smoothing
    y = monthly_x.to_dataset('month').to_dataframe()
    smooth_y = y.rolling(window = 31, min_periods = 1, win_type = 'gaussian', center = True).mean(std = 17)
    smooth_y = smooth_y.to_xarray().to_array('month')

    # Removing the smoothed trend
    stacked_x = x.assign_coords(year = x['time.year'], month = x['time.month']).set_index(time = ('year', 'month'))
    z = smooth_y.stack(time = ('year', 'month'))
    x_detrended = stacked_x - z.sel(time = stacked_x.time)
    

    x_detrended = x_detrended.reset_index('time')  # supprimer le MultiIndex
    x_detrended = x_detrended.assign_coords(time = x.time)

    # Removing the stationary seasonal cycle and mean state (ERA5)
    era_cycle = xr.open_dataset(era5_cycle).z500
    era_cycle = era_cycle.sel(lat = slice(latmin, latmax + 1))
    NH_mean_cycle = era_cycle.groupby('time.month').mean().weighted(np.cos(np.deg2rad(era_cycle.lat))).mean(('lat', 'lon'))

    # Conversion of the unit (m²/s² → m)
    NH_mean_cycle = NH_mean_cycle/g

    # Substract the seasonal cycle
    x_detrended = x_detrended.groupby('time.month') + NH_mean_cycle
    x_detrended = x_detrended.assign_attrs(x.attrs)    

    return x_detrended




if __name__ == "__main__":
    
    args = parse_args()
    main(args)