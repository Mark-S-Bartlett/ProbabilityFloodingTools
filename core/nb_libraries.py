import os
import re
import sys
import csv
import copy
import json
import math
import subprocess
import numpy as np
import pandas as pd
import pathlib as pl
from glob import glob
import scrapbook as sb
import papermill as pm
from scipy import stats
from nptyping import Array
from scipy import interpolate
from scipy.optimize import minimize
import plotly.offline as po
from plotly.offline import download_plotlyjs, init_notebook_mode, plot