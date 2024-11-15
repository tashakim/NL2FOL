# Running this file compares FOL output from LINC and compares them with FOL output from NL2FOL.

import pandas as pd
import json


# Load NL2FOL results
nl2fol_results = pd.read_csv("nl2fol_results.csv")
# Load LINC results
with open("output/linc_results.json", "r") as f:
    linc_results = json.load(f)

# Display results for comparison
print("NL2FOL results:", nl2fol_results)
print("LINC results:", linc_results)
