# __init__.py

from importlib import resources
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Version of the localprojections package
__version__ = "0.0.2"
 
# Load scripts / classes / functions so that they can be called directly
from .lp import (
    PanelLP,
    PanelLPX,
    ThresholdPanelLPX,
    TimeSeriesLP,
    IRFPlot,
    ThresholdIRFPlot
)