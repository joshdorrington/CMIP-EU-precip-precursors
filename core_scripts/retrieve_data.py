

import os
import argparse
import xarray as xr 
import cmipaccess as cmip 

from dask.distributed import Client, LocalCluster
from xarray.coding.times import CFDatetimeCoder
from tqdm.notebook import tqdm
import subprocess

import logging
logging.getLogger("distributed").setLevel(logging.WARNING)


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Aggregate variable over ERA5 region masks.")

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='name of model simulation to retrieve.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='name of simulation experiment, e.g. historical, ssp370.')
    parser.add_argument('--member', type=str, required=True,
                        help='which ensemble member to retrieve.')
    parser.add_argument('--server', type=str, default=None,
                        help="Which esgf server to use. Change this to one of the available if the default doesn't work.")
    return parser.parse_args(arg_list)


def main(model,experiment,member, server=None):
    for variable in ['ua','va']:
        retrieve_data_single_variable(model=model, 
                    experiment=experiment,
                    member_id=member,
                    variable=variable,
                    select_plev=True,
                    plev=85000, server=server)
    retrieve_data_single_variable(model=model, 
                experiment=experiment,
                member_id=member,
                variable='zg',
                select_plev=True,
                plev=50000, server=server)
    retrieve_data_single_variable(model=model, 
            experiment=experiment,
            member_id=member,
            variable='pr',
            select_plev=False,server=server)
            

def retrieve_data_single_variable(model, experiment, member_id, variable, select_plev, plev=85000,**path_kwargs):
    path = cmip.esgf.get_path_CMIP6_data(model, experiment, member_id, variable, freq='day',table='day',**path_kwargs)
    if select_plev:
        chunks = dict(plev=1, lon=50, lat=50, time=31*6)
    else:
        chunks = dict(lon=50, lat=50, time=365*5)
    ds = xr.open_mfdataset(path, chunks = chunks, decode_times=CFDatetimeCoder(use_cftime=True))
    if select_plev:
        ds = ds.sel(plev=plev)
    ds=ds.load()
    if select_plev:
        variable_name=f"{variable[:1]}{plev//100:.0f}"
    else:
        variable_name=variable
    dates = '-'.join(ds.time.dt.strftime("%Y%m%d").isel(time=[0,-1]).values)
    file_name = f"{variable_name}_day_{model}_{experiment}_{member_id}_gn_{dates}.nc"
    path = f'/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/raw/{model}/{variable_name}/{experiment}/'
    if not os.path.exists(path):
        os.makedirs(path)
        subprocess.run(["chmod", "-R", "g+rwx", path], check=True)
    ds.to_netcdf(f'{path}/{file_name}')
    print('Downloading', file_name, 'successful!')
    print(f'Saved to {path}')


if __name__=='__main__':
    args = parse_args()
    cluster = LocalCluster(n_workers=10, memory_limit='8GiB')
    client = Client(cluster)
    print('Access dask dashboard: ', client.dashboard_link)

    
    main(model =args.model, experiment = args.experiment,member=args.member, server=args.server)
