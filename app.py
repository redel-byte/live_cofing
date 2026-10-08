import numpy as np
import pandas as pd
from src.livecoding.extract import extract
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
from fg-data_profiling import ProfileReport

plt.style.use("ggplot")


df = iris.frame.copy()

print(df.columns)
# df = df.rename(columns={"target": "species"})

# df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

# X = df[iris.feature_names]
# y = df["species"]
