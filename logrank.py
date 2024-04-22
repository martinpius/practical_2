from lifelines import KaplanMeierFitter, NelsonAalenFitter
from nelson_aalen import _build
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

data = _build()
df1, df2 = data.loc[:, ['surv1',"delta1"]], data.loc[:, ['surv2',"delta2"]]

km1 = KaplanMeierFitter()
km1.fit(data.loc[:, "surv1"], event_observed = data.loc[:, "delta1"], label = "group1")

km2 = KaplanMeierFitter()
km2.fit(data.loc[:, "surv2"], event_observed = data.loc[:, "delta2"], label = "group2")

ax = km1.plot(ci_show=False)
km2.plot(ax = ax, ci_show=False)

plt.xlabel('time')
plt.ylabel('Survival probability estimate')
plt.savefig('two_km_curves', dpi=300)
#plt.show()
print(f"\n-----------------------------------")
print(km1.confidence_interval_)


def logrank_p_value(df1: pd.DataFrame = df1, df2: pd.DataFrame = df2) -> Tuple:
    result = logrank_test(df1.surv1, df2.surv2,
                          df1.delta1,df2.delta2)
    return result.p_value

p = logrank_p_value()
print(f"\n-----------------------------------")
print(f"\n >>>> The level of significance used: alpha = 0.05\
      \n >>>> p-value obtained from the log-rank test: p = {p:4f}")
print(f"\n-----------------------------------")
