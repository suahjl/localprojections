# localprojections
This module implements the local projections models for single entity time series, and panel / longitudinal data settings, due to Jorda (2005), and based on codes available [here](https://sites.google.com/site/oscarjorda/home/local-projections).

# Installation
1. ```pip install localprojections```

# Implementation
## Panel Local Projections Model
### Documentation
```python
localprojections.PanelLP(data, Y, response, horizon, lags, varcov, ci_width)
```
#### Parameters
data :  
	Pandas MultiIndex dataframe with entity as the outer index, and time as the inner index.

Y :  
	List of column names in ```data``` to be used in the model estimation

response :  
	List of column names in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

horizon :  
	Integer indicating the estimation horizon of the IRFs

lags :  
	Integer indicating the number of lags to be included in the model estimation

varcov :  
	Variance-covariance estimator to be used in estimating standard errors; refer to the [linearmodels package](https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.fit.html#linearmodels.panel.model.PanelOLS.fit).

ci_width :  
	Float higher than 0 and less than 1, i.e., (0, 1), indicating the width of the confidence intervals of the IRFs; ```ci_width=0.95``` indicates a 95% confidence interval

#### Output
This function returns a pandas dataframe of 6 columns: 
1. ```Shock``` indicates the shock variable
2. ```Response``` indicates the response variable
3. ```Horizon``` indicates the response horizon of the IRF
4. ```Mean``` indicates the point estimate of the IRF
5. ```LB``` indicates the lower bound of the confidence interval of the IRF
6. ```LB``` indicates the upper bound of the confidence interval of the IRF

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```

### Example

```python
from statsmodels.datasets import grunfeld
import localprojections as lp

df = grunfeld.load_pandas().data # import the Grunfeld investment data set
df = df.set_index(['firm', 'year']) # set entity-year indices (as per requirements in bashtage's linearmodels)

endog = ['invest', 'value', 'capital'] # cholesky ordering: invest --> value --> capital
response = endog.copy() # estimate the responses of all variables to shocks from all variables
irf_horizon = 8 # estimate IRFs up to 8 periods ahead
opt_lags = 2 # include 2 lags in the local projections model
opt_cov = 'robust' # HAC standard errors
opt_ci = 0.95 # 95% confidence intervals

irf = lp.PanelLP(data=df, # input dataframe
                 Y=endog, # variables in the model
                 response=response, # variables whose IRFs should be estimated
                 horizon=irf_horizon, # estimation horizon of IRFs
                 lags=opt_lags, # lags in the model
                 varcov=opt_cov, # type of standard errors
                 ci_width=opt_ci # width of confidence band
                 )
irfplot = lp.IRFPlot(irf=irf, # take output from the estimated model
                     response=['invest'], # plot only response of invest ...
                     shock=endog, # ... to shocks from all variables
                     n_columns=2, # max 2 columns in the figure
                     n_rows=2, # max 2 rows in the figure
                     maintitle='Panel LP: IRFs of Investment', # self-defined title of the IRF plot
                     show_fig=True, # display figure (from plotly)
                     save_pic=False # don't save any figures on local drive
                     )
```

## Single Entity Time Series Local Projections Model
### Documentation
```python
localprojections.TimeSeriesLP(data, Y, response, horizon, lags, newey_lags, ci_width)
```
#### Parameters
data :  
	Pandas dataframe

Y :  
	List of column names in ```data``` to be used in the model estimation

response :  
	List of column names in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

horizon :  
	Integer indicating the estimation horizon of the IRFs

lags :  
	Integer indicating the number of lags to be included in the model estimation

newey_lags :  
	Maximum number of lags to be used when estimating the Newey-West standard errors

ci_width :  
	Float higher than 0 and less than 1, i.e., (0, 1), indicating the width of the confidence intervals of the IRFs; ```ci_width=0.95``` indicates a 95% confidence interval

#### Output
This function also returns a pandas dataframe of 6 columns: 
1. ```Shock``` indicates the shock variable
2. ```Response``` indicates the response variable
3. ```Horizon``` indicates the response horizon of the IRF
4. ```Mean``` indicates the point estimate of the IRF
5. ```LB``` indicates the lower bound of the confidence interval of the IRF
6. ```LB``` indicates the upper bound of the confidence interval of the IRF

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```

### Example
```python
from statsmodels.datasets import grunfeld
import localprojections as lp

df = grunfeld.load_pandas().data # import the Grunfeld investment data set
df = df[df['firm'] == 'General Motors'] # keep only one entity (as an example of a single entity time series setting)
df = df.set_index(['year']) # set time variable as index

endog = ['invest', 'value', 'capital'] # cholesky ordering: invest --> value --> capital
response = endog.copy() # estimate the responses of all variables to shocks from all variables
irf_horizon = 8 # estimate IRFs up to 8 periods ahead
opt_lags = 2 # include 2 lags in the local projections model
opt_cov = 'robust' # HAC standard errors
opt_ci = 0.95 # 95% confidence intervals

# Use TimeSeriesLP for the single entity case
irf = lp.TimeSeriesLP(data=df, # input dataframe
                      Y=endog, # variables in the model
                      response=response, # variables whose IRFs should be estimated
                      horizon=irf_horizon, # estimation horizon of IRFs
                      lags=opt_lags, # lags in the model
                      newey_lags=2, # maximum lags when estimating Newey-West standard errors
                      ci_width=opt_ci # width of confidence band
                      )
irfplot = lp.IRFPlot(irf=irf, # take output from the estimated model
                     response=['invest'], # plot only response of invest ...
                     shock=endog, # ... to shocks from all variables
                     n_columns=2, # max 2 columns in the figure
                     n_rows=2, # max 2 rows in the figure
                     maintitle='Single Entity Time Series LP: IRFs of Investment', # self-defined title of the IRF plot
                     show_fig=True, # display figure (from plotly)
                     save_pic=False # don't save any figures on local drive
                     )

```

## Plotting Impulse Response Functions
### Documentation
```python
localprojections.IRFPlot(irf, response, shock, n_columns, n_rows, maintitle, show_fig, save_pic, out_path, out_name)
```
#### Parameters
irf :  
	Output from ```PanelLP()```, or ```TimeSeriesLP()```

response :  
	List of variables contained in ```irf```'s ```Response``` column whose IRFs is to be plotted 

shock :  
	List of variables contained in ```irf```'s ```Shock``` column whose IRFs is to be plotted 

n_columns :  
	Integer indicating the number of IRF figures per row in the overall figure

n_rows :  
	Integer indicating the number of IRF figures per column in the overall figure

maintitle :  
	Strings to be used as the title of the overall figure; default is ```''Local Projections Model: Impulse Response Functions'```

show_fig :  
	Boolean indicating whether to render the overall figure

save_pic :  
	Boolean indicating whether to save the overall figure in the local directory; if ```True```, a ```html``` file and a ```png``` file will be saved

out_path :  
	Strings indicating the directory at which the overall figure should be saved in; only used if ```save_pic``` is ```True```

out_name :  
	Strings indicating the name of the file in which the overall figure should be saved as; only used if ```save_pic``` is ```True```, and default is ```IRFPlot```

#### Output
This function returns a [plotly graph objects figure](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html) with ```n_columns``` (columns) x ```n_rows``` (rows) subplots. Depending on arguments passed, the figure may be rendered during implementation and / or saved in the local directory.

### Examples
See above.

# Requirements
## Python Packages
- pandas>=1.4.3
- numpy>=1.23.0
- linearmodels>=4.27
- plotly>=5.9.0
- statsmodels>=0.13.2

