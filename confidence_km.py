from nelson_aalen import NelsonAalen_raw, _build
import matplotlib.pyplot as plt 
import pandas as pd
from typing import Tuple, List
import math

data = _build()
na1, event_times1, ri1, mi1 = NelsonAalen_raw(df = data, group = "g1")
na2, event_times2, ri2, mi2 = NelsonAalen_raw(df = data, group = "g2")

print(f"\n-----------------------------------") 
print(f">>>> The size of the CHF for group 1: {len(na1)}\
      \n>>>> The size of the CHF for group 2: {len(na2)}\
      \n>>>> The size of risks set_1: {len(ri1)}\
      \n>>>> The size of failure set_1: {len(mi1)}\
      \n>>>> The size of risks set_2: {len(ri2)}\
      \n>>>> The size of failure set_2: {len(mi2)}")

print(f"\n-----------------------------------")
df1 = pd.DataFrame(data = event_times1, columns = ["t"])

df1["ri"] = ri1
df1["mi"] = mi1
df1["s1"] = [math.exp(-k) for k in na1]

df2 = pd.DataFrame(data = event_times2, columns = ["t"])
df2["ri"] = ri2
df2["mi"] = mi2
df2["s2"] = [math.exp(-k) for k in na2]

def KM_ConfidenceInterval(
        df: Tuple[pd.DataFrame, pd.DataFrame] = (df1, df2), 
        alpha: float = 0.05, group: str = "g1"):
    
    '''
    @Martin
    --------
   This method computes the 95% confidence interval for the Kaplan-Meier estimator. 
   [Here we use the $S_t$ derived from the naive estimator.] 
   We compute the 95% CI, assuming normality (large sample):

  \[ \hat{S_t} \pm Z_{\alpha} = 1.96 \times \sqrt{\text{Var}(\hat{S_t})} \]
    where the variance is given by Greenwood's estimator as:
    \[ \text{Var}(\hat{S_{KM}})(t) = \tau \rightarrow \frac{m_i}{r_i \times (r_i - m_i)} \]
   
   where $m_i$ is the number of failures and $r_i$ is the number at risk."

    ## NOTE: This is a naive implementation. 
    Results might have slight variations compared to other software.


    Arguments:
    -----------
    df: Tuple of pd.DataFrames, built from the outputs of the NA-estimator for both groups
    alpha: Level of significance, Default we compute 95% CI ==> 5% error is tolerated
    group: str, Default group ==> g1

    Returns:
    --------

    '''
    df1, df2 = df # unpacking the dataframes
    data = df1 if group == "g1" else df2
    #print(data)

    # compute the variance component to be accumulated at each time step
    data["tau"] = data['mi'] / (data['ri'] * (data['ri'] - data['mi']))
    data["tau"][0] = 0
    q = 0 # Start at 0 to accumulate the varince [using Greenwood's method]
    v_component = [] 
    
    z_alpha = 1.96 if alpha == 0.05 else input("enter the value of z")

    for t in range(len(data)): 
        q+=data.iloc[t,:]["tau"] # accumulating the sum for the "tau" column
        v_component.append(q)

    # Adding the accumulated sum as a separate column to a dataframe
    data["sum"] = v_component
    
    # Compute the Greenwood estimator for the variance [two groups case]
    if group == "g1":
        data["var_g"] = data["sum"] * data["s1"]**2
        data["sd"] = data["var_g"]**(0.5)
    
    else:

        data['var_g'] = data["sum"] * data["s2"]**2
        data["sd"] = data["var_g"]**(0.5)

    # compute the 95 % CI for \hat{S}_{KM} (Here we park lower and upper)
    # intervals separately
    if group == "g1":
        data["upper_ci"] = data["s1"] + z_alpha * data['var_g']**0.5#(data["s1"]**2 * (data["sum"]))**2 
        data["lower_ci"] = data["s1"] - z_alpha * data["var_g"]**0.5#(data["s1"]**2 * (data["sum"]))**2
    else:
        
        data["upper_ci"] = data["s2"] + z_alpha * data["sd"]
        data["lower_ci"] = data["s2"] - z_alpha * data["sd"] 
    return data

def plotCIs(groups: List = ["g1", "g2"]) -> None:

    '''
    @Martin
    --------
    This method plot
    '''
    for g in groups:
        if g == "g1":

            data = KM_ConfidenceInterval()
            plt.figure(figsize = (10, 12))
            plt.plot(data["t"], data["s1"], color = "fuchsia",
                linewidth = 3.8, label = "group1 survival")
            plt.fill_between(data["t"], data["lower_ci"], 
                        data["upper_ci"], color = "gray", 
                        alpha = 0.8)
            plt.legend(loc = "best")
            plt.xlabel("survival time")
            plt.ylabel("KM estimate(From NA = exp(-CHF))")
            plt.title("95% Confidence intervals for the KM estimate (KM = exp(-CHF))")
            plt.show()
        else:
            data = KM_ConfidenceInterval(group = g)
            plt.figure(figsize = (12, 12))
            # **** We exclude the last timestamp, 'which is infinity'
            plt.plot(data.iloc[:-1]["t"], data.iloc[:-1]["s2"], color = "bisque",
                linewidth = 3.8, label = "group2 survival")
            plt.fill_between(data.iloc[:-1]["t"], data.iloc[:-1]["lower_ci"], 
                        data.iloc[:-1]["upper_ci"], color = "salmon", 
                        alpha = 0.8)
            plt.legend(loc = "best")
            plt.xlabel("survival time")
            plt.ylabel("KM estimate(From NA = exp(-CHF))")
            plt.title("95% Confidence intervals for the KM estimate (KM = exp(-CHF))")
            plt.show()

data = KM_ConfidenceInterval(group = "g1")
print(data)
plotCIs()



