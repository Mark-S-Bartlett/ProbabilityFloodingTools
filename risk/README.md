# Average Annual Loss (AAL) Calculations

### Note: 
- This will only work if all data has been uploaded to s3, including the weights. Pluvial weights should be json files under `s3://{PROJECT}/Data/Forcing/Pluvial/outputs`. Fluvial weights should be csv files under `s3://pfra/RiskAssessment/{PROJECT}/BreachAnalysis/weighted_adjusted_results_{PROJECT}_{MODEL}.csv`. See the file structure of DC for an example.

## Procedures

1. **RUN: `PM-All_AAL.ipynb`**
    - This will run the `Prep_AAL_Inputs.ipynb`, `AAL-Calculator.ipynb`, `CreateAALResults.ipynb`, `AAL-ResultSummary.ipynb`, and `AAL-AnomalyReport.ipynb` notebooks for a given project and list of models.
    - The user needs to specify the *project*, *models*, *books*, *TRI*, and the *output directory*.
    - To sepcify TRI, comment out one of the two tri variables. `tri = ''` means that this is _**NOT**_ a Global Scaling Test. `tri = '_TRI'` means that this _**IS**_ a Global Scaling Test.
    - See below for an example of the user inputs cell:
>    ```python
>        project = 'DC'
>        models = ['P01', 'P03', 'P04', 'P06']
>        books = ['MB', 'Uncorrelated', 'Uniform']
>        tri = ''  # if IT IS NOT a TRI run
>        # tri = '_TRI'  # if IT IS a TRI run
>        output_dir_name = 'outputs'
>    ```



___
### [Deprecated procedures](https://github.com/Dewberry/probmod-tools/blob/9dcb79b76024f5f33c90a40c5cd6e33ca32cb0db/risk/README.md)
___
