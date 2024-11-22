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
	List of column labels in ```data``` to be used in the model estimation

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

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

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

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

## Panel Local Projections Model with Exogenous Variables (Panel LPX)
### Documentation
```python
localprojections.PanelLPX(data, Y, X, response, horizon, lags, varcov, ci_width)
```
#### Parameters
data :  
	Pandas MultiIndex dataframe with entity as the outer index, and time as the inner index.

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

X :  
	List of column labels in ```data``` to be used in the model estimation as exogenous variables

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

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

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

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

## Threshold Panel Local Projections Model with Exogenous Variables (Threshold Panel LPX)
### Documentation
```python
localprojections.ThresholdPanelLPX(data, Y, X, threshold_var, response, horizon, lags, varcov, ci_width)
```
#### Parameters
data :  
	Pandas MultiIndex dataframe with entity as the outer index, and time as the inner index.

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

X :  
	List of column labels in ```data``` to be used in the model estimation as exogenous variables

threshold_var :  
	String indicating column in ```data``` to be used as the threshold variable; must take values 0 or 1 for technically correct implementation

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

horizon :  
	Integer indicating the estimation horizon of the IRFs

lags :  
	Integer indicating the number of lags to be included in the model estimation

varcov :  
	Variance-covariance estimator to be used in estimating standard errors; refer to the [linearmodels package](https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.fit.html#linearmodels.panel.model.PanelOLS.fit).

ci_width :  
	Float higher than 0 and less than 1, i.e., (0, 1), indicating the width of the confidence intervals of the IRFs; ```ci_width=0.95``` indicates a 95% confidence interval

#### Output
This function returns *two* pandas dataframes of 6 columns each, with the first output corresponding to when ```threshold_var``` takes value 1, and the second when ```threshold_var`` takes value 0: 
1. ```Shock``` indicates the shock variable
2. ```Response``` indicates the response variable
3. ```Horizon``` indicates the response horizon of the IRF
4. ```Mean``` indicates the point estimate of the IRF
5. ```LB``` indicates the lower bound of the confidence interval of the IRF
6. ```LB``` indicates the upper bound of the confidence interval of the IRF

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

### Example

```python
from statsmodels.datasets import grunfeld
import localprojections as lp

df = grunfeld.load_pandas().data  # import the Grunfeld investment data set
df = df.set_index(['firm', 'year'])  # set entity-year indices (as per requirements in bashtage's linearmodels)
df["state"] = np.random.randint(0, 1, size=len(df))  # creates the state dummy variable (random numbers for illustration)
df["exog"] = np.random.normal(loc=5,scale=1,size=n)  # new column of floats as exogenous variable (random numbers for illustration)

endog = ['invest', 'value', 'capital']  # cholesky ordering: invest --> value --> capital
exog = ["exog"]
threshold = ["state"]
response = endog.copy()  # estimate the responses of all variables to shocks from all variables
irf_horizon = 8  # estimate IRFs up to 8 periods ahead
opt_lags = 2  # include 2 lags in the local projections model
opt_cov = 'kernel'  # HAC standard errors
opt_ci = 0.95  # 95% confidence intervals

irf_on, irf_off = lp.ThresholdPanelLPX(
    data=df,  # input dataframe
    Y=endog,  # endogenous variables in the model
    X=exog,  # exogenous variables in the model
    threshold_var=threshold,  # the threshold dummy variable
    response=response,  # variables whose IRFs should be estimated
    horizon=irf_horizon,  # estimation horizon of IRFs
    lags=opt_lags,  # lags in the model
    varcov=opt_cov,  # type of standard errors
     ci_width=opt_ci  # width of confidence band
     )
irfplot = lp.ThresholdIRFPlot(
    irf_threshold_on=irf_on,  # IRF for when the threshold variable takes value 1
    irf_threshold_off=irf_off,  # IRF for when the threshold variable takes value 0
    response=['invest'],  # plot only response of invest ...
    shock=endog,  # ... to shocks from all variables
    n_columns=2,  # max 2 columns in the figure
    n_rows=2,  # max 2 rows in the figure
    maintitle='Panel LP: IRFs of Investment',  # self-defined title of the IRF plot
    show_fig=True,  # display figure (from plotly)
    save_pic=False  # don't save any figures on local drive
    )
```


## Threshold Single Entity Time Series Local Projectiosn Model with Exogenous Variables (Threshold LPX)
### Documentation
```python
ThresholdTimeSeriesLPX(data, Y, X, threshold_var, response, horizon, lags, newey_lags=4, ci_width=0.95)
``` 
#### Parameters 
data :  
	Pandas dataframe

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

X :  
	List of column labels in ```data``` to be used in the model estimation as exogenous variables

threshold_var :  
	String indicating column in ```data``` to be used as the threshold variable; must take values 0 or 1 for technically correct implementation

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

horizon :  
	Integer indicating the estimation horizon of the IRFs

lags :  
	Integer indicating the number of lags to be included in the model estimation

newey_lags :  
	Maximum number of lags to be used when estimating the Newey-West standard errors

ci_width :  
	Float higher than 0 and less than 1, i.e., (0, 1), indicating the width of the confidence intervals of the IRFs; ```ci_width=0.95``` indicates a 95% confidence interval
#### Output
This function returns *two* pandas dataframes of 6 columns each, with the first output corresponding to when ```threshold_var``` takes value 1, and the second when ```threshold_var`` takes value 0: 
1. ```Shock``` indicates the shock variable
2. ```Response``` indicates the response variable
3. ```Horizon``` indicates the response horizon of the IRF
4. ```Mean``` indicates the point estimate of the IRF
5. ```LB``` indicates the lower bound of the confidence interval of the IRF
6. ```LB``` indicates the upper bound of the confidence interval of the IRF

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

## Single Entity Time Series Local Projections Model (LP)
### Documentation
```python
localprojections.TimeSeriesLP(data, Y, response, horizon, lags, newey_lags, ci_width)
```
#### Parameters
data :  
	Pandas dataframe

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

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

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

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

## Single Entity Time Series Local Projections Model with Exogenous Variables (LPX)
### Documentation
```python
localprojections.TimeSeriesLPX(data, Y, X, response, horizon, lags, newey_lags=4, ci_width=0.95)
```
#### Parameters
data :  
	Pandas dataframe

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

X :  
	List of column labels in ```data``` to be used in the model estimation as exogenous variables
response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

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

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

## Panel Quantile Local Projections Model with Exogenous Variables (Panel Quantile LPX)
### Documentation
Note: This function implements the panel quantile LPX using ```statsmodel```'s panel quantile regression and entity dummies, rather than "de-meaned" fixed effects as would ```PanelOLS```.
```
PanelQuantileLPX(data, Y, X, Entity, response, horizon, lags, varcov="robust", kernel="epa", bandwidth="hsheather", ci_width=0.95, quantile=0.5)
```
#### Parameters
data :  
	Pandas dataframe

Y :  
	List of column labels in ```data``` to be used in the model estimation as endogenous variables

X :  
	List of column labels in ```data``` to be used in the model estimation as exogenous variables

Entity :  
	Column label corresponding to the entity identifiers, which will be used to construct dummy fixed effects. 

response :  
	List of column labels in ```Y``` to be used as response variables when estimating the impulse response functions (IRFs)

horizon :  
	Integer indicating the estimation horizon of the IRFs

lags :  
	Integer indicating the number of lags to be included in the model estimation

varcov :  
	Variance-covariance estimator to be used in estimating standard errors; refer to the [statsmodels package](https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html).

kernel :  
	Asymptotic kernel matrix; refer to the [statsmodels package](https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html).

bandwidth :  
	Bandwidth selection method for asymptotic covariance estimate; refer to the [statsmodels package](https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html).

ci_width :  
	Float higher than 0 and less than 1, i.e., (0, 1), indicating the width of the confidence intervals of the IRFs; ```ci_width=0.95``` indicates a 95% confidence interval

quantile :  
	Float between 0 and 1 indicating the quantile of interest. E.g., 0.05 corresponds to the 5th percentile and 0.95 corresponds to the 95th percentile.

#### Output
This function also returns a pandas dataframe of 6 columns: 
1. ```Shock``` indicates the shock variable
2. ```Response``` indicates the response variable
3. ```Horizon``` indicates the response horizon of the IRF
4. ```Mean``` indicates the point estimate of the IRF
5. ```LB``` indicates the lower bound of the confidence interval of the IRF
6. ```LB``` indicates the upper bound of the confidence interval of the IRF

For instance, the estimates of the 6-period ahead IRF of y from a shock in x, can be found in the row with ```Shock=x```, ```Response=y```, and ```Horizon=6```.

## Plotting Impulse Response Functions
### Documentation
```python
localprojections.IRFPlot(irf, response, shock, n_columns, n_rows, maintitle, show_fig, save_pic, out_path, out_name, annot_size, font_size)
```
#### Parameters
irf :  
	pd.Dataframe containing 6 columns, labelled as ```Shock```, ```Response```, ```Horizon```, ```Mean```, ```LB```, ```UB```

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

annot_size :  
    Integer indicating the font size of titles of each subplot in the figure; defaults to 6

font_size :  
    Integer indicating the font size of the title, and axes labels; defaults to 9


#### Output
This function returns a [plotly graph objects figure](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html) with ```n_columns``` (columns) x ```n_rows``` (rows) subplots. Depending on arguments passed, the figure may be rendered during implementation and / or saved in the local directory.

### Example
See above.

# Requirements
## Python Packages
- pandas>=1.4.3
- numpy>=1.23.0
- linearmodels>=4.27
- plotly>=5.9.0
- statsmodels>=0.13.2


## Plotting Impulse Response Functions of a Threshold Local Projections Model 
### Documentation
This function plots IRFs estimated from ```ThresholdPanelLPX``` and ```ThresholdTimeSeriesLPX```.

```python
localprojections.ThresholdIRFPlot(irf_threshold_on, irf_threshold_off, response, shock, n_columns, n_rows, maintitle, show_fig, save_pic, out_path, out_name, annot_size, font_size)
```

#### Parameters
irf_threshold_on :  
	pd.Dataframe containing 6 columns, labelled as ```Shock```, ```Response```, ```Horizon```, ```Mean```, ```LB```, ```UB```, correspoinding to when the threshold variable is switched on; the first output from ```ThresholdPanelLPX``` and ```ThresholdTimeSeriesLPX```

irf_threshold_off :  
	pd.Dataframe containing 6 columns, labelled as ```Shock```, ```Response```, ```Horizon```, ```Mean```, ```LB```, ```UB```, correspoinding to when the threshold variable is switched on; the second output from ```ThresholdPanelLPX``` and ```ThresholdTimeSeriesLPX```

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

annot_size :  
    Integer indicating the font size of titles of each subplot in the figure; defaults to 6

font_size :  
    Integer indicating the font size of the title, and axes labels; defaults to 9

#### Output
This function returns a [plotly graph objects figure](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html) with ```n_columns``` (columns) x ```n_rows``` (rows) subplots. Depending on arguments passed, the figure may be rendered during implementation and / or saved in the local directory.

### Example
See above.