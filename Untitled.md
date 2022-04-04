```python
fname = "10k_files/sec-edgar-filings/XEL/10-K/0000072903-20-000011/filing-details.html"
with open(fname, encoding="utf-8") as report_file:
    html = report_file.read()
words = ['(material|materials|raw materials|raw material)','(commodity|products|goods|merchandise|risk|risks|supply|shortage)']     # search for a greeting 
rgx = NEAR_regex(words,max_words_between=4)
print(len(re.findall(rgx,texts)))            # both are caught
[m.group(0) for m in re.finditer(rgx,texts)]
```


```python
fname = "10k_files/sec-edgar-filings/WMB/10-K/0000107263-20-000005/filing-details.html"
with open(fname, encoding="utf-8") as report_file:
    html = report_file.read()
words = ['(energy|oil|fuel|gas|coal|renewable｜electricity )','(commodity|products|goods|merchandise|risk|risks)']     # search for a greeting 
rgx = NEAR_regex(words,max_words_between=5)
print(len(re.findall(rgx,texts)))            # both are caught
[m.group(0) for m in re.finditer(rgx,texts)]
```


```python
#For each firm, load the corresponding 10-K and create (at least) 5 different risk measures, 
#and save those new measurements to each of 5 new variables in that row.
for index, row in tqdm(sp500[:5].iterrows()):
    
    # get the filename for this secutity
    # sometimes, there is no 10-K
    
    possible_fnames = glob.glob('10k_files/sec-edgar-filings/'+row['Symbol']+'/10-K/*/*.html')
    if len(possible_fnames) > 0:
        fname = possible_fnames[0]
        text = extracto(fname)
    print(row['Security'])
    words = ['(energy|oil|fuel|gas|coal|renewable｜electricity )','(commodity|products|goods|merchandise)']
    rgx = NEAR_regex(words,max_words_between=5)    
    sp500.at[index, 'commodity_risk_energy']=len(re.findall(rgx,text)) 
    new_risk_var = sp500.at[index, 'commodity_risk_energy']=len(re.findall(rgx,text))
    print(new_risk_var)
    sp500.at[index, 'commodity_risk_energy']=new_risk_var # add new_risk_var to your toy database
    [m.group(0) for m in re.finditer(rgx,texts)]
    print('------')
```


```python
import glob
import os
from tqdm import tqdm
import pandas as pd
import re
from bs4 import BeautifulSoup
from near_regex import NEAR_regex 
```


```python
# Creates an output/ folder
os.makedirs('output',exist_ok=True)
```


```python
#Loads the initial dataset of sample firms saved inside of input/.
sp500 = pd.read_csv('input/sp500_firms.csv')
```


```python
#Loads the initial dataset of sample firms saved inside of input/.
sp500 = pd.read_csv('input/sp500_firms.csv')
```


```python
  for index, row in tqdm(sp500.iterrows()):
    
    # get the filename for this secutity
    # sometimes, there is no 10-K
   possible_fnames = glob.glob('10k_files/sec-edgar-filings/'+row['Symbol']+'/10-K/*/*.html')
   print(possible_fnames)
```
