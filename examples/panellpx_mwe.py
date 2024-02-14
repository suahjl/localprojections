# %%
import pandas as pd
import localprojections as lp
from datetime import date, timedelta

# %%
# Prelims + Read data
path_data = "./data/"
path_output = "./output/"
df = pd.read_parquet(path_data + "data.parquet")

# %%
# Wrangle data
# Set numeric time index (the underlying panelols api requires numeric dates)
df["time"] = df.groupby("country").cumcount()
del df["quarter"]
# Set multiindex (entity, then time)
df = df.set_index(["country", "time"])

# %%
# Setup
endog_base = ["y1", "y2", "y3"]  # list of column labels for endogenous variables 
exog_base = ["x1", "x2"]  # list of column labels for exogenous variables
# Estimate
irf = lp.PanelLPX(
    data=df,  # dataframe with both endog and exog variables
    Y=endog_base,  # list of endog variables
    X=exog_base,  # list of exog variables
    response=endog_base, 
    horizon=8, 
    lags=1,
    varcov="kernel",
    ci_width=0.95,
)
# Plot
fig_irf = lp.IRFPlot(
    irf=irf,
    response=endog_base,
    shock=endog_base,
    n_columns=len(endog_base),
    n_rows=len(endog_base),
    maintitle="Local Projections Model: Impulse Response Functions",
    show_fig=False,
    save_pic=False,
    out_path="",
    out_name="",
    annot_size=14,
    font_size=16,
)
fig_irf.write_image(path_output + "irf_plot" + ".png", height=1080, width=1920)
# %%
