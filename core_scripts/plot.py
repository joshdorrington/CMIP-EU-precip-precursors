import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

import matplotlib.colors as mcolors

def lighten_color(color, amount=0.2):
    """Lighten color by blending with white."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - amount * (1 - c))


def add_categories(a,boundary_ratio,minimal_thresh):
    clist=['xkcd:powder blue','xkcd:sand','#c5b5d4','#de7f8b','xkcd:seafoam green']
    edge_c='k'
    v=100
    rr=minimal_thresh*2**-0.5
    theta=np.rad2deg(np.tan(boundary_ratio))


    h1=a.add_patch(Wedge((0,0),v*2**0.5,-theta-90,theta-90,facecolor=clist[0],edgecolor=edge_c,zorder=1,linewidth=2,label='Dynamical Bias'))
    h2=a.add_patch(Wedge((0,0),v*2**0.5,-theta-270,theta-270,facecolor=clist[0],edgecolor=edge_c,zorder=1,linewidth=2,label='_'))
    h3=a.add_patch(Wedge((0,0),v*2**0.5,-theta,theta,facecolor=clist[1],edgecolor=edge_c,zorder=1,linewidth=2,label='Conversion Bias'))
    h4=a.add_patch(Wedge((0,0),v*2**0.5,-theta-180,theta-180,facecolor=clist[1],edgecolor=edge_c,zorder=1,linewidth=2,label='_'))
    h5=a.add_patch(plt.Rectangle((0,0), v,v, facecolor=clist[2],edgecolor=edge_c,zorder=0,linewidth=2,label='Compounding Bias'))
    h6=a.add_patch(plt.Rectangle((0,0), -v,-v, facecolor=clist[2],edgecolor=edge_c,zorder=0,linewidth=2,label='_'))
    h7=a.add_patch(plt.Rectangle((0,0), v,-v, facecolor=clist[3],edgecolor=edge_c,zorder=0,linewidth=2,label='Compensating Bias'))
    h8=a.add_patch(plt.Rectangle((0,0), -v,v, facecolor=clist[3],edgecolor=edge_c,zorder=0,linewidth=2,label='_'))
    rr=20
    rr=rr*2**-0.5
    minimal_patch=a.add_patch(plt.Rectangle((-rr,-rr), 2*rr, 2*rr,angle=45,rotation_point=(0,0),
                            facecolor=clist[4],edgecolor=edge_c,zorder=1,linewidth=2,label='Minimal Bias'))
    
    dyn_patches=[h1,h2]
    cnv_patches=[h3,h4]
    cmp_patches=[h5,h6]
    can_patches=[h7,h8]
    
    #add grey dashed lines on negative diagonals
    for dv in [-60,0,60]:
        a.plot([-v,v],[v+dv,-v+dv],c='gray',zorder=1,linestyle=':')

    #add grey diamonds
    for rr in 100*np.array(np.arange(0.2,2.41,0.2)):
            rr=rr*2**-0.5
            a.add_patch(plt.Rectangle((-rr,-rr), 2*rr, 2*rr,angle=45,rotation_point=(0,0),edgecolor='gray',label='_',facecolor='none'))

    return dyn_patches,cnv_patches,cmp_patches,can_patches,minimal_patch

def prepare_mean_dataframe(
    df,
    x_vars_to_sum,
    y_vars_to_sum,
    models=None,
    seasons=None,
    regions=None,
    term="bias",
    ens=False
):
    d = df.copy()

    # ---- Filtering ----
    if models is not None:
        d = d[d["model"].isin(models)]
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    if regions is not None:
        d = d[d["region_id"].isin(regions)]

    d = d[d["term"] == term]

    if ens:
        indexes=["model", "season", "region_id","member"]
    else:
        indexes=["model", "season", "region_id"]
    # ---- Pivot so sources become columns ----
    wide = d.pivot_table(
        index=indexes,
        columns="source",
        values="value",
        aggfunc="sum"
    ).reset_index()

    # ---- Compute x and y ----
    wide["xval"] = wide[x_vars_to_sum].sum(axis=1)
    wide["yval"] = wide[y_vars_to_sum].sum(axis=1)
    return wide

def get_model_markers(models):
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    return {m: markers[i % len(markers)] for i, m in enumerate(models)}

def get_season_colors(seasons):
    #cmap = plt.get_cmap("tab10")

    colors=dict(DJF='tab:blue',
    MAM='tab:green',
    JJA='tab:red',
    SON='tab:orange'
    )

    return {s: colors[s] for s in seasons}#{s: cmap(i) for i, s in enumerate(seasons)}

def get_model_colors(models):
    model_colors={
       'CESM2':'tab:blue',
        'MPI-GE':'tab:red'
    }
    return {m: model_colors[m] for m in models}

def get_region_colors(regions, base_color=None):
    cmap = plt.get_cmap("tab10")
    if base_color is None:
        return {r: cmap(i) for i, r in enumerate(regions)}
    else:
        # vary lightness of base color
        return {r: lighten_color(base_color, 1 - 0.8*i/len(regions))
                for i, r in enumerate(regions)}

def _add_scatter(ax, wide, style,base_color='tab:blue'):
    model_markers = get_model_markers(wide["model"].unique())
    season_colors = get_season_colors(wide["season"].unique())
    model_colors = get_model_colors(wide["model"].unique())

    regions = wide["region_id"].unique()

    multi_season = len(season_colors) > 1
    multi_region = len(regions) > 1
    multi_model = len(model_markers) > 1

    #set colors for data points:

    # by season
    if multi_season:
        #or region, hue set by season
        if multi_region:
            colors = [get_region_colors(regions, base_color=season_colors[s])[r] for s,r in zip(wide['season'],wide['region_id'])]
        else:
            colors = [season_colors[x] for x in wide["season"]]

    #or by region only
    elif multi_region:
            colors = [get_region_colors(regions)[r] for r in wide['region_id']]
    #or else by model
    elif multi_model:
        colors = [model_colors[x] for x in wide['model']]
    #or finally, fixed
    else:
        colors=[base_color for _ in wide['xval']]


    wide['_plot_color']=colors
    for i, r in wide.iterrows():

        ax.scatter(
            r["xval"], r["yval"],
            color=r["_plot_color"],
            marker=model_markers[r["model"]],
            **style
        )

    return model_markers, season_colors, regions, wide
    
def add_mean_scatter(ax, wide):
    style = dict(edgecolors="w", s=100, alpha=1, zorder=50)
    return _add_scatter(ax, wide, style)

def add_ens_scatter(ax, wide):
    style = dict(s=20, edgecolors="k", linewidths=0.3, alpha=0.95)
    return _add_scatter(ax, wide, style)

def add_uncertainty_ellipses(ax, wide, mean_colors, n_std=3):

    for _, r in wide.iterrows():
        # correlation ellipse base radii
        ell_radius_x = np.sqrt(1 + r["pcorr"])
        ell_radius_y = np.sqrt(1 - r["pcorr"])

        ellipse = mpatches.Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor="none",
            lw=2,
            edgecolor=r["_plot_color"],
            zorder=20,
        )

        scale_x = r["xstd"] * n_std
        scale_y = r["ystd"] * n_std

        transf = (
            mtransforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(r["xval"], r["yval"])
        )

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

def attach_uncertainty(wide, std_df, x_vars_to_sum, y_vars_to_sum,
                       models=None, seasons=None, regions=None, term="bias"):

    # ---- Build variable names ----
    xstd = "std_" + "+".join(x_vars_to_sum)
    ystd = "std_" + "+".join(y_vars_to_sum)
    corr = "corr_" + "_".join(["+".join(y_vars_to_sum), "+".join(x_vars_to_sum)])

    # ---- Filter std_df exactly like mean df ----
    std_d = std_df.copy()
    if models is not None:
        std_d = std_d[std_d["model"].isin(models)]
    if seasons is not None:
        std_d = std_d[std_d["season"].isin(seasons)]
    if regions is not None:
        std_d = std_d[std_d["region_id"].isin(regions)]

    std_d = std_d[std_d["term"] == term]

    # ---- Keep only needed columns ----
    std_d = std_d[["model", "season", "region_id", xstd, ystd, corr]]

    # ---- Merge (safe!) ----
    wide = wide.merge(std_d, on=["model", "season", "region_id"], how="left")

    # ---- Rename for plotting convenience ----
    wide = wide.rename(columns={xstd: "xstd", ystd: "ystd", corr: "pcorr"})

    return wide

from matplotlib.lines import Line2D

def attach_labels(ax, x_vars_to_sum, y_vars_to_sum, term,
                  wide, model_markers, mean_colors, regions, has_ens=False):

    short_vars = dict(dynamical='dyn', conversion='conv', nonlinear='NL')

    # ---- Axis labels ----
    xlonlab = "+".join(x_vars_to_sum).capitalize()
    xshortlab = "+".join([short_vars[xv.lower()] for xv in x_vars_to_sum])
    xlabel = fr'$b_\text{{{xshortlab}}}$, {xlonlab} {term} (relative %)'
    ax.set_xlabel(xlabel)

    ylonlab = "+".join(y_vars_to_sum).capitalize()
    yshortlab = "+".join([short_vars[yv.lower()] for yv in y_vars_to_sum])
    ylabel = fr'$b_\text{{{yshortlab}}}$, {ylonlab} {term} (relative %)'
    ax.set_ylabel(ylabel)

    season_handles = []
    model_handles = []
    region_handles = []
    size_handles = []

    # Determine dimensionality
    multi_season = len(mean_colors) > 1
    multi_region = len(regions) > 1
    multi_model = len(model_markers) > 1
    # ---- Seasons and/or Regions ----
    if multi_season and multi_region:
        # Both season and region vary: create handles using actual colors from wide
        # Get unique season-region pairs that appear in the data
        unique_pairs = wide[['season', 'region_id', '_plot_color']].drop_duplicates().sort_values(['season', 'region_id'])
        for _, row in unique_pairs.iterrows():
            season_handles.append(
                Line2D([0],[0], marker='o', color='w',
                       markerfacecolor=row['_plot_color'], 
                       label=f"{row['season']} Region {row['region_id']}", markersize=8)
            )
    elif multi_season:
        # Only season varies
        season_handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=mean_colors[s], label=s, markersize=8)
            for s in mean_colors
        ]
    elif multi_region:
        # Only region varies
        reg_colors = get_region_colors(regions)
        region_handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=reg_colors[r], label=f"Region {r}", markersize=8)
            for r in regions
        ]

    # ---- Models ----
    if multi_model:
        if not multi_season and not multi_region:
            model_colors = get_model_colors(model_markers)
        else: 
            model_colors= {m:'k' for m in model_markers}
        model_handles = [
            Line2D([0],[0], marker=model_markers[m], color=model_colors[m],
                   linestyle='None', label=m, markersize=8)
            for m in model_markers
        ]

    # ---- Mean vs ensemble ----
    if has_ens:
        size_handles = [
            Line2D([0],[0], marker='o', color='k', linestyle='None',
                   markersize=10, label="Mean"),
            Line2D([0],[0], marker='o', color='k', linestyle='None',
                   markersize=4, label="Ensemble member"),
        ]

    handles = season_handles + region_handles + model_handles + size_handles

    if handles:
        ax.legend(handles=handles, frameon=True)

def add_rosette(ax,xvars,yvars,mean_df,ens_df=None,boot_df=None,models=None,seasons=None,regions=None,term='bias',br=0.2,minthresh=0.2,n_std=2):

    wide = prepare_mean_dataframe(mean_df, xvars, yvars,
                                models, seasons, regions, term)

    # colors for ellipses
    season_colors = get_season_colors(wide["season"].unique())
    wide["color"] = wide["season"].map(season_colors)

    #background
    handles=add_categories(ax,br,minthresh)
    # scatter
    mean_markers, mean_colors, regions, wide = add_mean_scatter(ax, wide)
    ens_markers = ens_colors = None
    has_ens = False
    if ens_df is not None:
        ens_wide = prepare_mean_dataframe(ens_df, xvars, yvars,
                                        models, seasons, regions, term,
                                        ens=True)
        ens_markers, ens_colors,regions,ens_wide = add_ens_scatter(ax, ens_wide)
        has_ens = True

    # Mean uncertainty
    if boot_df is not None:
        wide = attach_uncertainty(wide, boot_df,
                                xvars, yvars,
                                models, seasons, regions, term)
        add_uncertainty_ellipses(ax, wide, mean_colors,n_std=n_std)

    #labels
    attach_labels(ax, xvars, yvars, term,
                wide, mean_markers, mean_colors, regions, has_ens=has_ens)    
    return


if __name__ == 'main':

    #this is the ensemble mean estimate of the different terms
    mean_df=pd.read_csv('/Data/gfi/users/jodor4442/LENS_paper_terms_df.csv',index_col=0)
    #These are the estimates of terms from each ensemble member totally independently
    #INTERPRETATION: if we only had 1 realisation of the model, this would be our uncertainty window
    ens_df=pd.read_csv('/Data/gfi/users/jodor4442/LENS_paper_terms_ens_df.csv',index_col=0)
    #These are estimates of future terms for each ensemble member, but with reference to the ensemble mean biases
    #INTERPRETATION: Given we have a large ensemble, how much internal variability is there in possible future values?
    fixed_hist_df=pd.read_csv('/Data/gfi/users/jodor4442/LENS_paper_mean_hist_ens_future_df.csv',index_col=0)

    #These are DIFFERENTLY FORMATTED metrics of sampling uncertainty in the ensemble mean estimates.
    mean_boot_df=pd.read_csv('/Data/gfi/users/jodor4442/LENS_paper_terms_mean_sampling_df.csv',index_col=0)


    #nans got zero_filled: send them back
    ens_df.loc[abs(ens_df.value) < 1e-12,'value']=np.nan
    fixed_hist_df.loc[abs(fixed_hist_df.value) < 1e-12,'value']=np.nan
    ens_df=ens_df.dropna()
    fixed_hist_df=fixed_hist_df.dropna()

    #more interpretable scaling
    scale_coeff=100 /0.05 # as percentage of reference event probability
    mean_df['value']*=scale_coeff
    ens_df['value']*=scale_coeff
    fixed_hist_df['value']*=scale_coeff
    for col in mean_boot_df.columns:
        if col.startswith('std_'):
            mean_boot_df[col]*=scale_coeff

    fig, ax = plt.subplots()
    fig.set_figwidth(7)
    fig.set_figheight(7)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect(1)

    xvars=['conversion']
    yvars=['dynamical','nonlinear']
    add_rosette(
        ax,xvars,yvars,
        mean_df,ens_df,mean_boot_df,
        models=['CESM2'],
        seasons=['DJF','MAM'],
        regions=[1,3,11,14],
        term='bias')
    fig.savefig('test_rosette.png')
