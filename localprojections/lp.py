# To implement the local projections methodology described in Jorda (2005), including the panel data version
# Written by Jing Lian Suah, Senior Economist at the Data, Analytics, and Research Unit, Central Bank of Malaysia

import pandas as pd
import numpy as np
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statistics import NormalDist
from plotly.subplots import make_subplots
import plotly.graph_objects as go

### PanelLP
## Input attributes
# data = pandas entity-time(int64 or datetime) multi-indexed dataframe
# Y = list of variables, cholesky ordered (last = contemporaneous shock from all previous variables)
# response = list of variables contained in Y to be shocked
# horizon = integer indicating estimation horizon for the IRFs (e.g., input 8 for 8 quarters ahead)
# lags = integer indicating number of lags to be used in the estimation models (e.g., 4 for 4 lags)
# varcov = string indicating type of varcov, see PanelOLS documentation
# ci_width = float within (0, 1) indicating the width of the confidence band (e.g., 0.95 for 95% CI)
def PanelLP(data, Y, response, horizon, lags, varcov='kernel', ci_width=0.95):
    ## Illegal inputs
    if (ci_width >= 1) | (ci_width <= 0): raise NotImplementedError('CI Width must be within (0, 1), non-inclusive!')
    if horizon < 1: raise NotImplementedError('Estimation horizon for IRFs must be at least 1')
    if lags < 1: raise NotImplementedError('Number of lags in the model must be at least 1')
    ## Preliminaries
    col_entity = data.index.names[0] # Retrieves name of entity index (assumed that 0 = entity, 1 = time as per PanelOLS)
    col_output = ['Shock', 'Response', 'Horizon', 'Mean', 'LB', 'UB'] # Column names of the output dataframe
    irf_full = pd.DataFrame(columns=col_output) # Empty output dataframe to be filled over every iteration
    z_val = NormalDist().inv_cdf((1 + ci_width) / 2) # Determines what multiplier to use when calculating UB & LB from SE
    for r in response:
        ## Check ordering of response variable in the full list of Y
        r_loc = Y.index(r)
        ## Generate copy of data for horizon h + first difference RHS variables + transform response variable to desired horizon
        for h in range(horizon+1):
            d = data.copy()
            d[r+'forward'] = d.groupby(col_entity)[r].shift(-h) # forward; equivalent to F`h'. in Stata
            ## Generate lags of RHS variables (only the first, either l0 or l1 will be used in the IRFs)
            list_RHS_forReg = []
            list_RHS_forIRF = []
            for y in Y:
                d[y] = d[y] - d.groupby(col_entity)[y].shift(1) # first difference
                if Y.index(y) == r_loc: # include lagged response variables on the RHS
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l) # for lagged dependent variable, we will use _l1 to generate the IRF
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    list_RHS_forIRF = list_RHS_forIRF + [y] # to figure out if the own-shock model should be used
                if Y.index(y) < r_loc: # y affects r contemporaneously
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l) # keep original name for convenience (d[y] = _l0 will be used for IRF)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                elif Y.index(y) > r_loc: # y affects r with a lag
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(2, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    d[y] = d[y].shift(1) # replace original with first lag (d[y] = _l1 will be used for IRF)
            d = d.dropna(axis=0) # clear all rows with NAs from the lag / forward transformations
            eqn = r+'forward' + '~' + '+'.join(list_RHS_forReg) + '+EntityEffects'
            eqn_ownshock = r+'forward' + '~' + '+'.join([r]+list_RHS_forReg) + '+EntityEffects' # own-shock model includes contemp first diff dependent
            mod = PanelOLS.from_formula(eqn, data=d)
            mod_ownshock = PanelOLS.from_formula(eqn_ownshock, data=d) # own-shock model
            est = mod.fit(cov_type=varcov)
            est_ownshock = mod_ownshock.fit(cov_type=varcov) # own-shock model
            beta = est.params
            beta_ownshock = est_ownshock.params # own-shock model
            se = est.std_errors
            se_ownshock = est_ownshock.std_errors # own-shock model
            for z in list_RHS_forIRF:
                irf = pd.DataFrame([[1] * len(col_output)], columns=col_output) # double list = single row
                irf['Response'] = r
                irf['Horizon'] = h
                if z == r: # shock = response
                    irf['Shock'] = r
                    irf['Mean'] = beta_ownshock[z]
                    irf['LB'] = beta_ownshock[z] - z_val * se_ownshock[z]
                    irf['UB'] = beta_ownshock[z] + z_val * se_ownshock[z]
                else: # shock =/= response
                    irf['Shock'] = z
                    irf['Mean'] = beta[z]
                    irf['LB'] = beta[z] - z_val * se[z]
                    irf['UB'] = beta[z] + z_val * se[z]
                irf_full = pd.concat([irf_full, irf], axis=0) # top to bottom concat
    ## Sort by response, shock, horizon
    irf_full = irf_full.sort_values(by=['Response', 'Shock', 'Horizon'], axis=0, ascending=[True, True, True])
    return irf_full

### PanelLPX
## Input attributes
# This is almost identical procedurally with the Panel version
# data = pandas dataframe (each row = 1 period)
# Y = list of variables, cholesky ordered (last = contemporaneous shock from all previous variables)
# response = list of variables contained in Y to be shocked
# horizon = integer indicating estimation horizon for the IRFs (e.g., input 8 for 8 quarters ahead)
# lags = integer indicating number of lags to be used in the estimation models (e.g., 4 for 4 lags)
# newey_lags = max lags for Newey-West HAC SEs
# ci_width = float within (0, 1) indicating the width of the confidence band (e.g., 0.95 for 95% CI)
def PanelLPX(data, Y, X, response, horizon, lags, varcov='kernel', ci_width=0.95):
    ## Illegal inputs
    if (ci_width >= 1) | (ci_width <= 0): raise NotImplementedError('CI Width must be within (0, 1), non-inclusive!')
    if horizon < 1: raise NotImplementedError('Estimation horizon for IRFs must be at least 1')
    if lags < 1: raise NotImplementedError('Number of lags in the model must be at least 1')
    if any(col in X for col in Y): raise NotImplementedError('Exogenous and endogenous blocks overlap! Please ensure they are mutually exclusive')
    ## Preliminaries
    col_entity = data.index.names[0] # Retrieves name of entity index (assumed that 0 = entity, 1 = time as per PanelOLS)
    col_output = ['Shock', 'Response', 'Horizon', 'Mean', 'LB', 'UB'] # Column names of the output dataframe
    irf_full = pd.DataFrame(columns=col_output) # Empty output dataframe to be filled over every iteration
    z_val = NormalDist().inv_cdf((1 + ci_width) / 2) # Determines what multiplier to use when calculating UB & LB from SE
    for r in response:
        ## Check ordering of response variable in the full list of Y
        r_loc = Y.index(r)
        ## Generate copy of data for horizon h + first difference RHS variables + transform response variable to desired horizon
        for h in range(horizon+1):
            d = data.copy()
            d[r+'forward'] = d.groupby(col_entity)[r].shift(-h) # forward; equivalent to F`h'. in Stata
            ## Generate lags of RHS variables (only the first, either l0 or l1 will be used in the IRFs)
            list_RHS_forReg = []
            list_RHS_forIRF = []
            for y in Y:
                d[y] = d[y] - d.groupby(col_entity)[y].shift(1) # first difference
                if Y.index(y) == r_loc: # include lagged response variables on the RHS
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l) # for lagged dependent variable, we will use _l1 to generate the IRF
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    list_RHS_forIRF = list_RHS_forIRF + [y] # to figure out if the own-shock model should be used
                if Y.index(y) < r_loc: # y affects r contemporaneously
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l) # keep original name for convenience (d[y] = _l0 will be used for IRF)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                elif Y.index(y) > r_loc: # y affects r with a lag
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(2, lags+1):
                        d[y + str(l) + 'lag'] = d.groupby(col_entity)[y].shift(l)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    d[y] = d[y].shift(1) # replace original with first lag (d[y] = _l1 will be used for IRF)
            d = d.dropna(axis=0) # clear all rows with NAs from the lag / forward transformations
            eqn = r+'forward' + '~' + '+'.join(list_RHS_forReg) + '+' + '+'.join(X) + '+EntityEffects'
            eqn_ownshock = r+'forward' + '~' + '+'.join([r]+list_RHS_forReg) + '+' + '+'.join(X) + '+EntityEffects' # own-shock model includes contemp first diff dependent
            mod = PanelOLS.from_formula(eqn, data=d)
            mod_ownshock = PanelOLS.from_formula(eqn_ownshock, data=d) # own-shock model
            est = mod.fit(cov_type=varcov)
            est_ownshock = mod_ownshock.fit(cov_type=varcov) # own-shock model
            beta = est.params
            beta_ownshock = est_ownshock.params # own-shock model
            se = est.std_errors
            se_ownshock = est_ownshock.std_errors # own-shock model
            for z in list_RHS_forIRF:
                irf = pd.DataFrame([[1] * len(col_output)], columns=col_output) # double list = single row
                irf['Response'] = r
                irf['Horizon'] = h
                if z == r: # shock = response
                    irf['Shock'] = r
                    irf['Mean'] = beta_ownshock[z]
                    irf['LB'] = beta_ownshock[z] - z_val * se_ownshock[z]
                    irf['UB'] = beta_ownshock[z] + z_val * se_ownshock[z]
                else: # shock =/= response
                    irf['Shock'] = z
                    irf['Mean'] = beta[z]
                    irf['LB'] = beta[z] - z_val * se[z]
                    irf['UB'] = beta[z] + z_val * se[z]
                irf_full = pd.concat([irf_full, irf], axis=0) # top to bottom concat
    ## Sort by response, shock, horizon
    irf_full = irf_full.sort_values(by=['Response', 'Shock', 'Horizon'], axis=0, ascending=[True, True, True])
    return irf_full

### TimeSeriesLP
## Input attributes
# This is almost identical procedurally with the Panel version
# data = pandas dataframe (each row = 1 period)
# Y = list of variables, cholesky ordered (last = contemporaneous shock from all previous variables)
# response = list of variables contained in Y to be shocked
# horizon = integer indicating estimation horizon for the IRFs (e.g., input 8 for 8 quarters ahead)
# lags = integer indicating number of lags to be used in the estimation models (e.g., 4 for 4 lags)
# newey_lags = max lags for Newey-West HAC SEs
# ci_width = float within (0, 1) indicating the width of the confidence band (e.g., 0.95 for 95% CI)
def TimeSeriesLP(data, Y, response, horizon, lags, newey_lags=4, ci_width=0.95):
    ## Illegal inputs
    if (ci_width >= 1) | (ci_width <= 0): raise NotImplementedError('CI Width must be within (0, 1), non-inclusive!')
    if horizon < 1: raise NotImplementedError('Estimation horizon for IRFs must be at least 1')
    if lags < 1: raise NotImplementedError('Number of lags in the model must be at least 1')
    ## Preliminaries
    col_output = ['Shock', 'Response', 'Horizon', 'Mean', 'LB', 'UB'] # Column names of the output dataframe
    irf_full = pd.DataFrame(columns=col_output) # Empty output dataframe to be filled over every iteration
    z_val = NormalDist().inv_cdf((1 + ci_width) / 2) # Determines what multiplier to use when calculating UB & LB from SE
    for r in response:
        ## Check ordering of response variable in the full list of Y
        r_loc = Y.index(r)
        ## Generate copy of data for horizon h + first difference RHS variables + transform response variable to desired horizon
        for h in range(horizon+1):
            d = data.copy()
            d[r+'forward'] = d[r].shift(-h) # forward; equivalent to F`h'. in Stata
            ## Generate lags of RHS variables (only the first, either l0 or l1 will be used in the IRFs)
            list_RHS_forReg = []
            list_RHS_forIRF = []
            for y in Y:
                d[y] = d[y] - d[y].shift(1) # first difference
                if Y.index(y) == r_loc: # include lagged response variables on the RHS
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d[y].shift(l) # for lagged dependent variable, we will use _l1 to generate the IRF
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    list_RHS_forIRF = list_RHS_forIRF + [y] # to figure out if the own-shock model should be used
                if Y.index(y) < r_loc: # y affects r contemporaneously
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(1, lags+1):
                        d[y + str(l) + 'lag'] = d[y].shift(l) # keep original name for convenience (d[y] = _l0 will be used for IRF)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                elif Y.index(y) > r_loc: # y affects r with a lag
                    list_RHS_forReg = list_RHS_forReg + [y]
                    list_RHS_forIRF = list_RHS_forIRF + [y]
                    for l in range(2, lags+1):
                        d[y + str(l) + 'lag'] = d[y].shift(l)
                        list_RHS_forReg = list_RHS_forReg + [y + str(l) + 'lag']
                    d[y] = d[y].shift(1) # replace original with first lag (d[y] = _l1 will be used for IRF)
            d = d.dropna(axis=0) # clear all rows with NAs from the lag / forward transformations
            eqn = r+'forward' + '~' + '+'.join(list_RHS_forReg)
            eqn_ownshock = r+'forward' + '~' + '+'.join([r]+list_RHS_forReg) # own-shock model includes contemp first diff dependent
            mod = smf.ols(eqn, data=d)
            mod_ownshock = smf.ols(eqn_ownshock, data=d) # own-shock model
            est = mod.fit(cov_type='HAC',cov_kwds={'maxlags':newey_lags})
            est_ownshock = mod_ownshock.fit(cov_type='HAC',cov_kwds={'maxlags':newey_lags}) # own-shock model
            beta = est.params
            beta_ownshock = est_ownshock.params # own-shock model
            se = est.bse
            se_ownshock = est_ownshock.bse # own-shock model
            for z in list_RHS_forIRF:
                irf = pd.DataFrame([[1] * len(col_output)], columns=col_output) # double list = single row
                irf['Response'] = r
                irf['Horizon'] = h
                if z == r: # shock = response
                    irf['Shock'] = r
                    irf['Mean'] = beta_ownshock[z]
                    irf['LB'] = beta_ownshock[z] - z_val * se_ownshock[z]
                    irf['UB'] = beta_ownshock[z] + z_val * se_ownshock[z]
                else: # shock =/= response
                    irf['Shock'] = z
                    irf['Mean'] = beta[z]
                    irf['LB'] = beta[z] - z_val * se[z]
                    irf['UB'] = beta[z] + z_val * se[z]
                irf_full = pd.concat([irf_full, irf], axis=0) # top to bottom concat
    ## Sort by response, shock, horizon
    irf_full = irf_full.sort_values(by=['Response', 'Shock', 'Horizon'], axis=0, ascending=[True, True, True])
    return irf_full


### IRFPlot
## Input attributes
# irf = output from PanelLP
# response = list of response variables to be plotted
# shock = list of shock variables to be plotted
# n_columns = number of columns in the consolidated IRF chart
# n_rows = number of rows in the consolidated IRF chart (e.g., a 3 rows by 2 columns figure of 5-6 IRF plots)
# maintitle = Title of the chart, default is 'Panel Local Projections Model: Impulse Response Functions'
# show_fig = invokes plotly's Figure.show() function, default is True
# save_pic = saves html and png versions of the chart, default is True
# out_path = directory to save output, default is the existing working directory
# out_name = name of output, default is 'IRFPlot'
def IRFPlot(irf,
            response,
            shock,
            n_columns,
            n_rows,
            maintitle='Local Projections Model: Impulse Response Functions',
            show_fig=True,
            save_pic=True,
            out_path='',
            out_name='IRFPlot', 
            annot_size=6,
            font_size=9):
    if (len(response) * len(shock)) > (n_columns * n_rows):
        raise NotImplementedError('Number of subplots (n_columns * n_rows) is smaller than number of IRFs to be plotted (n)')
    if list(irf.columns) == ['Shock', 'Response', 'Horizon', 'Mean', 'LB', 'UB']:
        n_col = n_columns
        n_row = n_rows
        ## Generate titles first
        list_titles=[]
        for r in response:
            for s in shock:
                subtitle = [s + ' -> ' + r]
                list_titles = list_titles + subtitle
        ## Main plot settings
        fig=make_subplots(rows=n_row, cols=n_col, subplot_titles=list_titles)
        ## Subplot loops
        count_col = 1
        count_row = 1
        for r in response:
            for s in shock:
                d = irf.loc[(irf['Response'] == r) & (irf['Shock'] == s)]
                d['Zero'] = 0 # horizontal line
                ## Zero
                fig.add_trace(go.Scatter(x=d['Horizon'],
                                         y=d['Zero'],
                                         mode='lines',
                                         line=dict(color='grey', width=1, dash='solid')),
                              row=count_row, col=count_col
                              )
                ## Mean
                fig.add_trace(go.Scatter(x=d['Horizon'],
                                         y=d['Mean'],
                                         mode='lines',
                                         line=dict(color='black', width=3, dash='solid')),
                              row=count_row, col=count_col
                              )
                ## LB
                fig.add_trace(go.Scatter(x=d['Horizon'],
                                         y=d['LB'],
                                         mode='lines',
                                         line=dict(color='black', width=1, dash='dash')),
                              row=count_row, col=count_col
                              )
                ## UBs
                fig.add_trace(go.Scatter(x=d['Horizon'],
                                         y=d['UB'],
                                         mode='lines',
                                         line=dict(color='black', width=1, dash='dash')),
                              row=count_row, col=count_col
                              )
                count_col += 1 # move to next
                if count_col <= n_col: pass
                elif count_col > n_col:
                    count_col = 1
                    count_row += 1
        fig.update_annotations(font_size=annot_size)
        fig.update_layout(title=maintitle,
                          plot_bgcolor='white',
                          hovermode='x unified',
                          showlegend=False,
                          font=dict(color='black', size=font_size))
        if show_fig == True:
            fig.show()
        if save_pic == True:
            fig.write_image(out_path + out_name + '.png', height=768, width=1366)
            fig.write_html(out_path + out_name + '.html')
    else:
        raise AttributeError("Input needs to be from PanelLP() or TimeSeriesLP(), which has these as columns: ['Shock', 'Response', 'Horizon', 'Mean', 'LB', 'UB']")
    return fig
