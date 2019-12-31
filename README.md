# probmod-tools
Probabilistic Modeling Tools

---

## Contents
 
#### Hydrology

 1. Pluvial tools moved to [pluvial-hydromet](https://github.com/Dewberry/pfra-hydromet).

 2. Fluvial tools include __Mean Curve__ and __Normalized Hydrograph__ development notebooks.
 
#### Hydraulics

 1. Breakline development routines. 

#### Scaling

 1. Notebooks for scaling tests & creating production runs on S3.

#### Risk

 1. Notebooks for calcualting risk at points and heatmap creation. 


#### QAQC

 1. Notebooks for querying results from RAS runs. 

---

#### core

Core code called by notebooks in this repo.
 
#### bin

Miscelaneous binaries called by notebooks.
 
#### docs
Repo Documentation

---


## Background & Description

Over the past few years, the National Academy of Sciences, the Technical Mapping Advisory Council, and flood insurance legislation (BW-12 and HFIAA) have all recommended or mandated that FEMA modernize the process by which flood risk is calculated and communicated, and that the data produced must be able to support evolving and expanding program needs.  Additionally, it has been advised that more modern risk-based analyses be performed to evaluate the risk and insurance rates for properties behind levees, and to more accurately calculate the flooding probabilities in those areas.

Although most Flood Risk Projects have typically used a “deterministic” analysis of multiple events (such as the 10%, 4%, 2%, 1%, and 0.2% annual chance floods), they have rarely considered the uncertainty that inherently exists in much of the hydrology and hydraulics of the analysis, and have historically only analyzed up to the 0.2% annual chance flood.  As more and more flood events continue to occur that affect those living outside the mapped special flood hazard areas (SFHAs), and as the Risk MAP program continues advancing towards structure-level risk assessment, mitigation and insurance ratings, it is critical that these uncertainties are taken into account in considering the range of likely flood scenarios that should be analyzed within the program.  One potential engineering solution is to perform and utilize probabilistic modeling outputs to help accomplish these goals.

The primary difference between a deterministic and a probabilistic flood mapping approach in calculating the 1% annual chance exceedance limits, for example, is that the deterministic approach typically involves modeling one event or scenario (e.g. the 1% annual chance discharge from a gage or regression equation) and mapping its extents as the 1% annual chance floodplain, whereas the probabilistic approach models and maps multiple (hundreds or thousands) of different scenarios, taking into account uncertainies in the data inputs, from which the extents where 1% of all the different scenarios is exceeded can be delineated.

Visit the [wiki](https://github.com/Dewberry/probmod-tools/wiki) page for descriptions and instructions of tools, overview of processes and details on technical background.

Visit the [issues](https://github.com/Dewberry/probmod-tools/issues) page for trouble shooting or bug reporting.

---

<img src="docs/images/starrII_logos.PNG" alt="drawing" width="700px" align='center'/>

---