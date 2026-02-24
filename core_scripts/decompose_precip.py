from genericpath import exists
import numpy as np
import sys
import os
import argparse
import xarray as xr

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Use precursor indices and precipitation indices to decompose model bias and trends.")

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='name of model to use.')

    parser.add_argument('--future_experiment', type=str, required=True,
                        help='Future simulation to compute changes for, e.g. ssp370.')
    
    
    # Optional arguments

    parser.add_argument('--eventthreshold',type=float,default=0.95,
                        help='extreme event percentile in referencemodel')
    
    parser.add_argument('--nprecursorbins',type=int,default=10,
                        help='Number of precursor bins to use.')
    
    parser.add_argument('--seasons', nargs='+', type=str, default=['DJF', 'MAM','JJA','SON'],
                    help='Which seasons to compute metrics for.')
    
    parser.add_argument('--regions', nargs='+', type=str, default=None,
                help='Which regions to compute metrics for. Default is to use all regions.')

    parser.add_argument('--members', type=str, default='',
                        help='Which ensemble member or members to use. Defaults to assuming not an ensemble.')
    
    parser.add_argument('--referencemodel', type=str, default='ERA5',
                    help='name of reference simulation.')

    parser.add_argument('--historical_experiment', type=str, default='historical',
                    help='name of experiment with historical overlap')

    parser.add_argument('--hist_period',type=int,nargs=2,default=[1979,2014],
                        help='The historical period to use.')

    parser.add_argument('--future_period',type=int,nargs=2,default=[2060,2100],
                        help='The future period to use.')
    
    parser.add_argument('--variables', nargs='+', type=str, default=['z500','u850','v850'],
                    help='Which precursors to use. If more than one provided, use PCA and take first PC.')

    parser.add_argument('--hazardvariable',type=str, default='pr',
                        help='Which variable defines the hazard index.')

    parser.add_argument('--overwrite',action='store_true',
                        help='Overwrite metrics if they already exist.')
    
    parser.add_argument('--inputdir',type=str,default='/Data/skd/projects/global/cmip6_precursors/outputs/indices/',
                        help='Directory in which to look for indices.')
    
    parser.add_argument('--auxdir',type=str,default='/Data/skd/projects/global/cmip6_precursors/aux/',
                    help='Directory in which to look for decomposition.py')
    
    parser.add_argument('--savedir',type=str,default='/Data/skd/projects/global/cmip6_precursors/outputs/decompositions/',
                    help='Directory in which to save output.')

    return parser.parse_args(arg_list)

def get_ref_data(args):
    bp=f'{args.inputdir}{args.referencemodel}/'
    # vars=np.atleast_1d(args.variables).append(args.hazardvariable)
    vars=list(np.atleast_1d(args.variables)) + [args.hazardvariable]
    ds=[]
    for v in vars:
        dir=f'{bp}{v}/{args.historical_experiment}/'
        a1=[]
        for r in args.regions:
            a2=[]
            for s in args.seasons:
                # da=xr.open_dataarray(f'{dir}+{args.season}_region{args.region_id}.nc')
                da=xr.open_dataarray(f'{dir}/{s}_region{r}.nc')
                a2.append(da.assign_coords(season=s))
            a1.append(xr.concat(a2,'season').assign_coords(region_id=r))
        da=xr.concat(a1,'region_id')
        ds.append(da.rename(v))
    ds=xr.merge(ds)

    y0,y1=args.hist_period
    # hist_period=np.arange(y0,y1+1)
    # ds=ds.isel(np.isin(ds['time.year'],hist_period))
    ds = ds.sel(time=slice(str(y0), str(y1)))
    
    return ds

def get_hist_data(args):
    bp=f'{args.inputdir}{args.model}/'

    # vars=np.atleast_1d(args.variables).append(args.hazardvariable)
    vars=list(np.atleast_1d(args.variables)) + [args.hazardvariable]
    ds=[]
    mems=np.atleast_1d(args.members)
    not_ens= (len(mems)==1) and (mems[0]=='')
    
    for v in vars:
        vdir=f'{bp}{v}/{args.historical_experiment}/'
        os.makedirs(vdir,exist_ok=True)
        a0=[]
        for mem in mems:
            if not_ens:
                dir=vdir
            else:
                dir=vdir+f'member_{mem}_'

            a1=[]
            for r in args.regions:
                a2=[]
                for s in args.seasons:
                    # da=xr.open_dataarray(f'{dir}+{args.season}_region{args.region_id}.nc')
                    # print(f'{dir}{s}_region{r}.nc')
                    da=xr.open_dataarray(f'{dir}{s}_region{r}.nc')
                    a2.append(da.assign_coords(season=s))
                a1.append(xr.concat(a2,'season').assign_coords(region_id=r))

                a1=xr.concat(a1,'region_id')
                if not not_ens: a1=a1.assign_coords(member=mem)
            a0.append(a1)

        a0=xr.concat(a0,'member').rename(v)
        ds.append(a0)

    ds=xr.merge(ds)

    y0,y1=args.hist_period
    # hist_period=np.arange(y0,y1+1)
    # ds=ds.isel(np.isin(ds['time.year'],hist_period))
    ds = ds.sel(time=slice(str(y0), str(y1)))
    return ds

def get_future_data(args):
    bp=f'{args.inputdir}{args.model}/'

    # vars=np.atleast_1d(args.variables).append(args.hazardvariable)
    vars=list(np.atleast_1d(args.variables)) + [args.hazardvariable]

    ds=[]
    mems=np.atleast_1d(args.members)
    not_ens= (len(mems)==1) and (mems[0]=='')
    
    for v in vars:
        vdir=f'{bp}{v}/{args.future_experiment}/'
        os.makedirs(vdir,exist_ok=True)
        a0=[]
        for mem in mems:
            if not_ens:
                dir=vdir
            else:
                dir=vdir+f'member_{mem}_'

            a1=[]
            for r in args.regions:
                a2=[]
                for s in args.seasons:
                    # da=xr.open_dataarray(f'{dir}_{args.season}_region{args.region_id}.nc')
                    da=xr.open_dataarray(f'{dir}{s}_region{r}.nc')
                    a2.append(da.assign_coords(season=s))
                a1.append(xr.concat(a2,'season').assign_coords(region_id=r))

                a1=xr.concat(a1,'region_id')
                if not not_ens: a1=a1.assign_coords(member=mem)
            a0.append(a1)

        a0=xr.concat(a0,'member').rename(v)
        ds.append(a0)

    ds=xr.merge(ds)

    y0,y1=args.future_period
    # future_period=np.arange(y0,y1+1)
    # ds=ds.isel(np.isin(ds['time.year'],future_period))
    ds = ds.sel(time=slice(str(y0), str(y1)))
    return ds

def get_savepaths(args,s,r):
    s1=f'{args.savedir}/{args.model}/'
    
    s2=f'{s}_region{r}.nc'
    return s1+'decomp_'+s2, s1+'terms_'+s2

if __name__=='__main__':

    args = parse_args()

    sys.path.append(args.auxdir)
    print(args.auxdir)
    from decomposition import decompose_hazard_odds_ratio, compute_terms_from_decomposition_with_alpha_blending

    condition_var=args.variables
    if len(condition_var)==1:
        condition_var=condition_var[0]
    else:
        #This principal component is fitted in the reference dataset and projected
        #on the models.
        condition_var='PC1'
    
    p_dvs=[v+'_lag0_index_val1' for v in args.variables]


    #use a pre-established set of regions if none specified
    if args.regions is None:
        # region 2 was an uninhabited island so we dropped it.
        args.regions=[1,*np.arange(3,40)] 
    else:
        pass

    ref_data=get_ref_data(args)
    hist_data=get_hist_data(args)
    future_data=get_future_data(args)
    hazard_var=args.hazardvariable
    make_h_var_cat=True
    p=args.eventthreshold
    bin_num=args.nprecursorbins
    for s in args.seasons:
        for r in args.regions:
            decomp_path,term_path=get_savepaths(args,s,r)

            # decomposed_hazard=decompose_hazard_odds_ratio(ref_data.sel(season=s,region=r),
            #                                               hist_data.sel(season=s,region=r),
            #                                               future_data.sel(season=s,region=r),
            #                                             hazard_var,condition_var,
            #                                             make_h_var_cat=make_h_var_cat,
            #                                             p_dvs=p_dvs,
            #                                             quantile=p,bin_num=bin_num)
            decomposed_hazard=decompose_hazard_odds_ratio(ref_data.sel(season=s,region_id=r),
                                                          hist_data.sel(season=s,region_id=r),
                                                          future_data.sel(season=s,region_id=r),
                                                        hazard_var,condition_var,
                                                        make_h_var_cat=make_h_var_cat,
                                                        p_dvs=p_dvs,
                                                        quantile=p,bin_num=bin_num)

            bias_and_trend_terms=compute_terms_from_decomposition_with_alpha_blending(*[decomposed_hazard[i] for i in range(6)])

            os.makedirs('/'.join(decomp_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(term_path.split('/')[:-1]), exist_ok=True)
            os.chmod('/'.join(decomp_path.split('/')[:-1]), 0o777)
            os.chmod('/'.join(term_path.split('/')[:-1]), 0o777)

            decomposed_hazard.rename('decomposition').to_netcdf(decomp_path)
            bias_and_trend_terms.to_netcdf(term_path)

