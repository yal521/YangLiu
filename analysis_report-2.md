## Risk Factors Analysis


```python
import glob
import os
from tqdm import tqdm
import pandas as pd
import re
from bs4 import BeautifulSoup
from near_regex import NEAR_regex 
import numpy as np
import seaborn as sns
import pandas_datareader as pdr 
from datetime import datetime
import matplotlib.pyplot as plt
import urllib
import statsmodels.api as sm
from statsmodels.formula.api import ols as sm_ols
from sklearn.linear_model import LinearRegression 
```


```python
sp500_accting_plus_textrisks =pd.read_csv('output/sp500_accting_plus_textrisks.csv')
```

### 1.Risk Measurements


####  
- After using BeautifulSoup to remove the html tags and transferring it to a pure text, I plan to use regex to search for certain words and topics related to financial risks before covid-19. I did some preparation work before doing the measurements. Besides the hint risks in the list of textbook, I picked up one big company in each industry and opened their 10-k file to do some manual search. I search risk to find the relative words and paragraphs. I collected the topics and key words that appear frequently and overlap among different industries. Then I group these key words and professional terms for test in the regex. For the first risk that I need to find three factors to measure it, I should find a general topic that related to the risks. From the big topic financial risks to its subtopic, I tried them and finally found that  the commodity risk and its measurable factors are suitable for analysis. They according to brainstorming, experience and the keywords list, I also found the credit risk and currency risk that may be covered in different industries. In this process, I have tried simple nouns and words group to see its influence on the companies. Most of the time, the counts is zero. For example, supply chain is an hot risk for enterprises. However, it is not easy to combine it with the risk simply. I should know more terms related to supply issue and use specific terms to measure it influence. After I split it up to subtopics, I found it is easier to build other risk topics to test and then I adjusted the words and topics whenever I found it is not appropriate to analyze.


####
- Risk Measurement Details
  - risk 1
    words =['(energy|oil|fuel|gas|coal|renewable｜electricity)','(commodity|products|goods|merchandise|risk|risks)']
    
    rgx = NEAR_regex(words,max_words_between=5)    
  - risk 2
    words =['(legal|regulation|approval|approve|regulatory|lawsuit|government|governmental)','(commodity|products|goods|merchandise|risk|risks)']
  
    rgx = NEAR_regex(words,max_words_between=6)    
  - risk 3
    words =['(material|materials|raw|materials|raw|material)','(commodity|products|goods|merchandise|risk|risks|supply|shortage)']
  
    rgx = NEAR_regex(words,max_words_between=4)    
  - risk 4
    words =['(credit|payment|collateral)','(risk|risks|risky|adversely|adverse|problem|problems|challenges|challenge)']
    
    rgx = NEAR_regex(words,max_words_between=2)    
  - risk 5
    words =['(international currency|local currency|currency|foreign currency|currency|exchange|currency|translation)','(risk|risks|risky|losses)']
  
    rgx = NEAR_regex(words,max_words_between=3)    

####   
- Economic Reasoning
  - Commodity Risk
Products and services are the main source of the revenues. They exist either individually or simultaneously. Normally, most of the companies has their own merchandise or their service that has some relations with commodities. Commodity risks also implies the possible fluctuations of future market values, which will directly influence future revenues and profits. I try to search for the relations between commodity and risks, it shows non zero number in most of the companies. This gives me confidence that companies have issues with commodity problems. Then I start to research what is the internal factors inside this problems. I have read some articles that explains how to measure the risks about commodity. Generally, cost, political factors, energy problems and so on make up the commodity risks. Both the material supply and energy affect the cost of the goods because material can be used to manufacture products or as inventories for selling or production while the energy is used to multiple aspects of the production and transportation. These factors are the dominant to the future market value of the products. Governmental regulations and approval is also important for the transportation and circulation of the merchandise. For example, medicines and medical equipment should obtain regulatory approval before entering the markets.
  - Credit Risk
Most companies need to lend money to achieve further development or solve the crises.The loan can help them deal with the recent cash difficulties. However, the results of investment are still uncertain. There is still potential risk that the company cannot fulfill the obligations. The same situation will happen when the companies become the creditors. Generally, the amount of the loan is very large and once the risks will happen, the companies may lose credit and reputation for refinancing as borrowers and capital losses as creditors.
  - Currency Risk
Many big companies are international enterprises or have business with foreign countries. Their daily transactions are affected by the fluctuating exchange rate. This change closely relates to the revenues and value of the assets.For example, the currency usually have relations with imports and exports that affect the supply chain. Also, inflation, options,futures,interest rate and so on are the essential consideration or solutions to deal with currency risks. For example, currency risk can offer companies unpredictable profits and losses immediately. Sometimes, in order to reduce the currency risks, US prefer to invest in some currency that its country has increasing interest rate and currency value. The transactions are always in the currency circulation so every company should focus on this risk no matter it has international business or not. Currency is also influenced by the domestic economical environment and policies.

#### 
- Statistical Properties
  - These risk factors that I choose have values for most firms.For commodity risk, it correlates with accounting factors such as inventories,accounts receivables,accounts payable, revenues, costs of goods, expenses and so on. For credit risks, it correlates with cash flow, revenues and liabilities. For currency risks, it also influence the revenues and loans.



### 2.Validation checks and discussion of the risk measurements 

- Most of these factors are valid because they help me find the relations between them and firm risks. Also, they tell some details about this type risks when the search area is larger.
- I choose two companies in two risk factors to measure their validation. For example, when I use keys words such as energy and oil to match products with the condition of max_words_between=5 to measure energy factor of commedity risks, I find the following match examples, such as "commodity risks associated with energy","energy and energy-related products",and "gas commodity". These examples shows the relations among risks,commedity,and energy. When I pick up several related words to measure materials factor of commedity risks, I can find examples, such as  "risks that management believes are material" and "risk that a material". They are closely related. However, I also find some examples that share few correlation with risks.For instance, it may show "supply of enriched nuclear material".

<img width="440" alt="image" src="https://user-images.githubusercontent.com/98285249/160140703-f1fcd52e-7c9a-4f85-a3a7-a06e6bb28ef3.png">
    


<img width="361" alt="image" src="https://user-images.githubusercontent.com/98285249/159194995-648669f3-577b-4a20-a9ba-aea9461ad291.png">

### 3.Description of the final sample for tests
In this report,there are 492 companies 10-k files for the analysis. In the overall dataset, there are 505 rows and 53 columns.Most of the columns have missing values because most of the columns have values that are less than 505.The range of mean of the different types of risks counts is from 5 to 10. The credit risk counts has an outlier that is much larger than the other data in the credit risk group.Because this company named JPM is an financial investment company so that it makes sense that it has high possibility to have credit risk. For most companies, higher risks counts are not ordinary. Generally, most companies risk counts are less than 10 and only a few companies have high risk counts.


```python
sp500_accting_plus_textrisks.shape
```




    (505, 53)




```python
sp500_accting_plus_textrisks.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 505 entries, 0 to 504
    Data columns (total 53 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   gvkey                      355 non-null    float64
     1   lpermno                    355 non-null    float64
     2   datadate                   355 non-null    object 
     3   fyear                      355 non-null    float64
     4   tic                        355 non-null    object 
     5   sic                        355 non-null    float64
     6   sic3                       355 non-null    float64
     7   td                         355 non-null    float64
     8   long_debt_dum              355 non-null    float64
     9   me                         355 non-null    float64
     10  l_a                        355 non-null    float64
     11  l_sale                     355 non-null    float64
     12  div_d                      355 non-null    float64
     13  age                        355 non-null    float64
     14  atr                        355 non-null    float64
     15  smalltaxlosscarry          276 non-null    float64
     16  largetaxlosscarry          276 non-null    float64
     17  l_emp                      355 non-null    float64
     18  l_ppent                    355 non-null    float64
     19  l_laborratio               355 non-null    float64
     20  Inv                        355 non-null    float64
     21  Ch_Cash                    355 non-null    float64
     22  Div                        355 non-null    float64
     23  Ch_Debt                    355 non-null    float64
     24  Ch_Eqty                    355 non-null    float64
     25  Ch_WC                      355 non-null    float64
     26  CF                         355 non-null    float64
     27  td_a                       355 non-null    float64
     28  td_mv                      355 non-null    float64
     29  mb                         355 non-null    float64
     30  prof_a                     355 non-null    float64
     31  ppe_a                      355 non-null    float64
     32  cash_a                     355 non-null    float64
     33  xrd_a                      355 non-null    float64
     34  dltt_a                     355 non-null    float64
     35  invopps_FG09               334 non-null    float64
     36  sales_g                    0 non-null      float64
     37  dv_a                       355 non-null    float64
     38  short_debt                 349 non-null    float64
     39  Symbol                     505 non-null    object 
     40  Security                   505 non-null    object 
     41  SEC filings                505 non-null    object 
     42  GICS Sector                505 non-null    object 
     43  GICS Sub-Industry          505 non-null    object 
     44  Headquarters Location      505 non-null    object 
     45  Date first added           460 non-null    object 
     46  CIK                        505 non-null    int64  
     47  Founded                    505 non-null    object 
     48  commodity_risk_energy      492 non-null    float64
     49  commodity_risk_government  492 non-null    float64
     50  commodity_risk_material    492 non-null    float64
     51  credit_risk                492 non-null    float64
     52  currency_risk              492 non-null    float64
    dtypes: float64(42), int64(1), object(10)
    memory usage: 209.2+ KB



```python
a=sp500_accting_plus_textrisks.isnull().sum() / sp500_accting_plus_textrisks.shape[0] * 100
d=a.rename_axis('column names').reset_index(name='percentage of missing value')
d
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column names</th>
      <th>percentage of missing value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gvkey</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lpermno</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>datadate</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fyear</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tic</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sic</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sic3</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>7</th>
      <td>td</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>8</th>
      <td>long_debt_dum</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>9</th>
      <td>me</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>10</th>
      <td>l_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>11</th>
      <td>l_sale</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>12</th>
      <td>div_d</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>13</th>
      <td>age</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>14</th>
      <td>atr</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>15</th>
      <td>smalltaxlosscarry</td>
      <td>45.346535</td>
    </tr>
    <tr>
      <th>16</th>
      <td>largetaxlosscarry</td>
      <td>45.346535</td>
    </tr>
    <tr>
      <th>17</th>
      <td>l_emp</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>18</th>
      <td>l_ppent</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>19</th>
      <td>l_laborratio</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Inv</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ch_Cash</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Div</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ch_Debt</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ch_Eqty</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Ch_WC</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CF</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>27</th>
      <td>td_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>28</th>
      <td>td_mv</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mb</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>30</th>
      <td>prof_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>31</th>
      <td>ppe_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>32</th>
      <td>cash_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>33</th>
      <td>xrd_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>34</th>
      <td>dltt_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>35</th>
      <td>invopps_FG09</td>
      <td>33.861386</td>
    </tr>
    <tr>
      <th>36</th>
      <td>sales_g</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>dv_a</td>
      <td>29.702970</td>
    </tr>
    <tr>
      <th>38</th>
      <td>short_debt</td>
      <td>30.891089</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Symbol</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Security</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>SEC filings</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>GICS Sector</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>GICS Sub-Industry</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Headquarters Location</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Date first added</td>
      <td>8.910891</td>
    </tr>
    <tr>
      <th>46</th>
      <td>CIK</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Founded</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>commodity_risk_energy</td>
      <td>2.574257</td>
    </tr>
    <tr>
      <th>49</th>
      <td>commodity_risk_government</td>
      <td>2.574257</td>
    </tr>
    <tr>
      <th>50</th>
      <td>commodity_risk_material</td>
      <td>2.574257</td>
    </tr>
    <tr>
      <th>51</th>
      <td>credit_risk</td>
      <td>2.574257</td>
    </tr>
    <tr>
      <th>52</th>
      <td>currency_risk</td>
      <td>2.574257</td>
    </tr>
  </tbody>
</table>
</div>




```python
sp500_accting_plus_textrisks.describe().T
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gvkey</th>
      <td>355.0</td>
      <td>45305.952113</td>
      <td>61170.060945</td>
      <td>1045.000000</td>
      <td>6286.000000</td>
      <td>13700.000000</td>
      <td>6.158250e+04</td>
      <td>3.160560e+05</td>
    </tr>
    <tr>
      <th>lpermno</th>
      <td>355.0</td>
      <td>53570.729577</td>
      <td>30143.136238</td>
      <td>10104.000000</td>
      <td>19531.500000</td>
      <td>58683.000000</td>
      <td>8.262000e+04</td>
      <td>9.343600e+04</td>
    </tr>
    <tr>
      <th>fyear</th>
      <td>355.0</td>
      <td>2018.884507</td>
      <td>0.320067</td>
      <td>2018.000000</td>
      <td>2019.000000</td>
      <td>2019.000000</td>
      <td>2.019000e+03</td>
      <td>2.019000e+03</td>
    </tr>
    <tr>
      <th>sic</th>
      <td>355.0</td>
      <td>4320.836620</td>
      <td>1946.653427</td>
      <td>100.000000</td>
      <td>2844.000000</td>
      <td>3760.000000</td>
      <td>5.455500e+03</td>
      <td>8.742000e+03</td>
    </tr>
    <tr>
      <th>sic3</th>
      <td>355.0</td>
      <td>431.864789</td>
      <td>194.696486</td>
      <td>10.000000</td>
      <td>284.000000</td>
      <td>376.000000</td>
      <td>5.455000e+02</td>
      <td>8.740000e+02</td>
    </tr>
    <tr>
      <th>td</th>
      <td>355.0</td>
      <td>12163.408327</td>
      <td>21665.685376</td>
      <td>0.000000</td>
      <td>1853.650000</td>
      <td>5135.385000</td>
      <td>1.250950e+04</td>
      <td>1.884020e+05</td>
    </tr>
    <tr>
      <th>long_debt_dum</th>
      <td>355.0</td>
      <td>0.983099</td>
      <td>0.129084</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>me</th>
      <td>355.0</td>
      <td>57088.310991</td>
      <td>116259.792773</td>
      <td>2963.886500</td>
      <td>13156.590000</td>
      <td>22421.930000</td>
      <td>5.146518e+04</td>
      <td>1.023856e+06</td>
    </tr>
    <tr>
      <th>l_a</th>
      <td>355.0</td>
      <td>9.710928</td>
      <td>1.228732</td>
      <td>6.569794</td>
      <td>8.799639</td>
      <td>9.693013</td>
      <td>1.056390e+01</td>
      <td>1.322070e+01</td>
    </tr>
    <tr>
      <th>l_sale</th>
      <td>355.0</td>
      <td>9.316292</td>
      <td>1.253828</td>
      <td>4.097822</td>
      <td>8.470949</td>
      <td>9.232229</td>
      <td>1.001762e+01</td>
      <td>1.314555e+01</td>
    </tr>
    <tr>
      <th>div_d</th>
      <td>355.0</td>
      <td>0.743662</td>
      <td>0.437227</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>age</th>
      <td>355.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>atr</th>
      <td>355.0</td>
      <td>0.238233</td>
      <td>0.241296</td>
      <td>0.000000</td>
      <td>0.126235</td>
      <td>0.200283</td>
      <td>2.418163e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>smalltaxlosscarry</th>
      <td>276.0</td>
      <td>0.717391</td>
      <td>0.451086</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>largetaxlosscarry</th>
      <td>276.0</td>
      <td>0.202899</td>
      <td>0.402888</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>l_emp</th>
      <td>355.0</td>
      <td>3.324613</td>
      <td>1.159023</td>
      <td>0.455524</td>
      <td>2.437980</td>
      <td>3.269569</td>
      <td>4.189655e+00</td>
      <td>6.025866e+00</td>
    </tr>
    <tr>
      <th>l_ppent</th>
      <td>355.0</td>
      <td>7.907718</td>
      <td>1.546264</td>
      <td>3.690204</td>
      <td>6.794235</td>
      <td>7.822405</td>
      <td>9.037771e+00</td>
      <td>1.111335e+01</td>
    </tr>
    <tr>
      <th>l_laborratio</th>
      <td>355.0</td>
      <td>4.656031</td>
      <td>1.316389</td>
      <td>0.511044</td>
      <td>3.837853</td>
      <td>4.387984</td>
      <td>5.336413e+00</td>
      <td>9.931146e+00</td>
    </tr>
    <tr>
      <th>Inv</th>
      <td>355.0</td>
      <td>0.054196</td>
      <td>0.084643</td>
      <td>-0.329408</td>
      <td>0.020729</td>
      <td>0.047795</td>
      <td>8.872825e-02</td>
      <td>4.238831e-01</td>
    </tr>
    <tr>
      <th>Ch_Cash</th>
      <td>355.0</td>
      <td>0.008871</td>
      <td>0.064776</td>
      <td>-0.315808</td>
      <td>-0.007922</td>
      <td>0.003967</td>
      <td>2.391021e-02</td>
      <td>3.837106e-01</td>
    </tr>
    <tr>
      <th>Div</th>
      <td>355.0</td>
      <td>0.025464</td>
      <td>0.026991</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020454</td>
      <td>3.761685e-02</td>
      <td>1.385936e-01</td>
    </tr>
    <tr>
      <th>Ch_Debt</th>
      <td>355.0</td>
      <td>0.013737</td>
      <td>0.072132</td>
      <td>-0.265326</td>
      <td>-0.019937</td>
      <td>-0.000001</td>
      <td>3.047276e-02</td>
      <td>4.217628e-01</td>
    </tr>
    <tr>
      <th>Ch_Eqty</th>
      <td>355.0</td>
      <td>-0.042515</td>
      <td>0.058366</td>
      <td>-0.282758</td>
      <td>-0.062046</td>
      <td>-0.023208</td>
      <td>-2.361254e-03</td>
      <td>1.741915e-01</td>
    </tr>
    <tr>
      <th>Ch_WC</th>
      <td>355.0</td>
      <td>0.011466</td>
      <td>0.044408</td>
      <td>-0.252402</td>
      <td>-0.005141</td>
      <td>0.006642</td>
      <td>2.438149e-02</td>
      <td>3.726431e-01</td>
    </tr>
    <tr>
      <th>CF</th>
      <td>355.0</td>
      <td>0.123176</td>
      <td>0.077063</td>
      <td>-0.288764</td>
      <td>0.074449</td>
      <td>0.113701</td>
      <td>1.612885e-01</td>
      <td>3.332969e-01</td>
    </tr>
    <tr>
      <th>td_a</th>
      <td>355.0</td>
      <td>0.329747</td>
      <td>0.192310</td>
      <td>0.000000</td>
      <td>0.207953</td>
      <td>0.321905</td>
      <td>4.333579e-01</td>
      <td>1.245754e+00</td>
    </tr>
    <tr>
      <th>td_mv</th>
      <td>355.0</td>
      <td>0.187401</td>
      <td>0.144410</td>
      <td>0.000000</td>
      <td>0.092015</td>
      <td>0.160159</td>
      <td>2.637551e-01</td>
      <td>8.095309e-01</td>
    </tr>
    <tr>
      <th>mb</th>
      <td>355.0</td>
      <td>3.022942</td>
      <td>2.092596</td>
      <td>0.877849</td>
      <td>1.567714</td>
      <td>2.408047</td>
      <td>3.661417e+00</td>
      <td>1.308288e+01</td>
    </tr>
    <tr>
      <th>prof_a</th>
      <td>355.0</td>
      <td>0.151314</td>
      <td>0.074428</td>
      <td>-0.323828</td>
      <td>0.102413</td>
      <td>0.138699</td>
      <td>1.868827e-01</td>
      <td>3.903839e-01</td>
    </tr>
    <tr>
      <th>ppe_a</th>
      <td>355.0</td>
      <td>0.247454</td>
      <td>0.218987</td>
      <td>0.009521</td>
      <td>0.091581</td>
      <td>0.162561</td>
      <td>3.367286e-01</td>
      <td>9.285623e-01</td>
    </tr>
    <tr>
      <th>cash_a</th>
      <td>355.0</td>
      <td>0.126002</td>
      <td>0.138469</td>
      <td>0.002073</td>
      <td>0.031900</td>
      <td>0.072171</td>
      <td>1.666899e-01</td>
      <td>6.946123e-01</td>
    </tr>
    <tr>
      <th>xrd_a</th>
      <td>355.0</td>
      <td>0.031169</td>
      <td>0.050173</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009526</td>
      <td>4.293628e-02</td>
      <td>3.367946e-01</td>
    </tr>
    <tr>
      <th>dltt_a</th>
      <td>355.0</td>
      <td>0.296568</td>
      <td>0.181230</td>
      <td>0.000000</td>
      <td>0.177941</td>
      <td>0.285137</td>
      <td>3.906725e-01</td>
      <td>1.071959e+00</td>
    </tr>
    <tr>
      <th>invopps_FG09</th>
      <td>334.0</td>
      <td>2.698513</td>
      <td>2.107435</td>
      <td>0.405435</td>
      <td>1.234730</td>
      <td>2.155533</td>
      <td>3.301717e+00</td>
      <td>1.216423e+01</td>
    </tr>
    <tr>
      <th>sales_g</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>dv_a</th>
      <td>355.0</td>
      <td>0.025464</td>
      <td>0.026991</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020454</td>
      <td>3.761685e-02</td>
      <td>1.385936e-01</td>
    </tr>
    <tr>
      <th>short_debt</th>
      <td>349.0</td>
      <td>0.112481</td>
      <td>0.111168</td>
      <td>0.000000</td>
      <td>0.028043</td>
      <td>0.084992</td>
      <td>1.512310e-01</td>
      <td>7.610294e-01</td>
    </tr>
    <tr>
      <th>CIK</th>
      <td>505.0</td>
      <td>788730.051485</td>
      <td>550104.995760</td>
      <td>1800.000000</td>
      <td>97476.000000</td>
      <td>882095.000000</td>
      <td>1.137789e+06</td>
      <td>1.868275e+06</td>
    </tr>
    <tr>
      <th>commodity_risk_energy</th>
      <td>492.0</td>
      <td>4.002033</td>
      <td>9.950898</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000e+00</td>
      <td>7.200000e+01</td>
    </tr>
    <tr>
      <th>commodity_risk_government</th>
      <td>492.0</td>
      <td>7.186992</td>
      <td>9.085787</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>8.000000e+00</td>
      <td>7.200000e+01</td>
    </tr>
    <tr>
      <th>commodity_risk_material</th>
      <td>492.0</td>
      <td>5.973577</td>
      <td>4.175541</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>8.000000e+00</td>
      <td>2.600000e+01</td>
    </tr>
    <tr>
      <th>credit_risk</th>
      <td>492.0</td>
      <td>9.621951</td>
      <td>15.885004</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>1.000000e+01</td>
      <td>1.760000e+02</td>
    </tr>
    <tr>
      <th>currency_risk</th>
      <td>492.0</td>
      <td>5.477642</td>
      <td>6.168832</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>8.000000e+00</td>
      <td>5.600000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
sp500_accting_plus_textrisks['currency_risk'].nunique()
```




    31



risk counts and value_counts of risk counts analysis 

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160135274-6e60d4d4-57ed-49a3-a12a-4c3efe534407.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160135462-a9a336d9-d567-4803-93f7-27bdbaa350e1.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160135675-0d7dec93-d128-4ee9-9437-690e18413b38.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160135822-48cda6f7-e91d-47da-b9d1-0bf1f783bbb7.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160136381-c2031abf-24a4-42bf-b73a-253f606fbc3a.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160136628-cd620d6b-86b9-4884-8023-0947d0adc100.png">


### 4.Explore the correlation between risk values and stock returns around key dates for the onset of covid.


```python
final_01 = pd.read_csv('output/final_01.csv')
```

#### coorelation heatmap

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160133845-c905ff2a-9035-42b6-9281-2042036ed1d2.png">


#### three types of return-commodity_risk_energy

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160136871-b8877dbc-7a63-47f8-8228-2eddd5c7057a.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160137034-d386c518-ed34-47ea-8656-c6415b922ce8.png">



<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160137196-434a46af-aaed-403a-9f15-f0dec4682c2c.png">


#### three types of return-commodity_risk_government

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160137348-dab600cc-c9fa-4ed5-91ef-91bc829ca21c.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160137485-f3c616da-c788-4238-a8fd-921bedb513a0.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160137789-806a43c3-e00c-4685-bfcd-81a800c6cdb6.png">


#### three types of return-commodity_risk_material

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160138010-47148a76-6aba-4745-8daa-dec83843c0c6.png">



<img width="553" alt="image" src="https://user-images.githubusercontent.com/98285249/160118766-689b6069-198b-458a-a6fe-600329140f40.png">



<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160138442-87911a06-9b61-41f9-97ab-9af266a403db.png">

#### three types of return-credit_risk

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160138698-e62c9087-7eb0-4347-8361-48bd786531be.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160138843-75100a1e-2e9c-4582-875e-502bd1ef07d9.png">

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160139029-83cda3e2-5a96-42bf-b6f4-746ce5f81898.png">


#### three types of return-currency_risk

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160139238-a746b6c1-3a67-4f5d-928a-39a147d499c5.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160139573-5be94552-d6a3-4d11-8941-629b6f9427a2.png">


<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160139848-e9f590f1-6921-4697-a27f-1efd8a023197.png">

#### comparison among three types of returns

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160140074-18ee5485-acb2-4147-8d02-c279f7e9a5ea.png">



#### an example of accounting variable with risk factors accounts

<img width="558" alt="image" src="https://user-images.githubusercontent.com/98285249/160140207-0f27bb65-7430-41fb-9644-66c99d255602.png">

####
- Summary
  - According to the linear regression visualization plot,during the period that is from March 9 2020 to March 13 2020,some type of risk factor counts have relations with the returns while the other factors do not.Energy factor of commodity has the maximum degree influence on the firms return among other factors.Energy factor negatively relates to the returns because of the position of the slop in the plot. The more energy risk factors are, the less of the firm returns. The energy risk should be concentrated by the management team to reduce the cost of operation. The credit risk shows the same trend but its effects is less then energy risk factors. However, the governmental factor of commodity shows positive relations with the firms returns. Firms with more governmental factor counts have higher returns. Though government regulations is strict to most of the companies, the companies may earn more profits and reputation once they obtain the regulatory approval. Surprisingly, both the material factor of commodity counts and currency risk factor counts have nearly no relations with the firms returns. For the monthly return from February 23 to March 23, the general trend is the same as the week from March 9 to March 13.However, after the stimulus was announced, the trend is opposite to the two previous time period.When previous period is negatively related, they are related positively after the March 23. By the way, accounting variables such as profitability can indicate that a firm should be more resilient to the crisis.


```python
sm_ols('W03090313wret ~ commodity_risk_energy + commodity_risk_government+commodity_risk_material+credit_risk+currency_risk',
       data=final_01).fit().summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>W03090313wret</td>  <th>  R-squared:         </th> <td>   0.099</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.086</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   7.582</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 05 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>9.00e-07</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:23:09</td>     <th>  Log-Likelihood:    </th> <td>  345.25</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   350</td>      <th>  AIC:               </th> <td>  -678.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   344</td>      <th>  BIC:               </th> <td>  -655.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>   -0.1198</td> <td>    0.011</td> <td>  -11.119</td> <td> 0.000</td> <td>   -0.141</td> <td>   -0.099</td>
</tr>
<tr>
  <th>commodity_risk_energy</th>     <td>   -0.0036</td> <td>    0.001</td> <td>   -5.280</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.002</td>
</tr>
<tr>
  <th>commodity_risk_government</th> <td>    0.0008</td> <td>    0.001</td> <td>    1.345</td> <td> 0.180</td> <td>   -0.000</td> <td>    0.002</td>
</tr>
<tr>
  <th>commodity_risk_material</th>   <td>-8.814e-05</td> <td>    0.001</td> <td>   -0.075</td> <td> 0.940</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>credit_risk</th>               <td>   -0.0013</td> <td>    0.001</td> <td>   -1.556</td> <td> 0.121</td> <td>   -0.003</td> <td>    0.000</td>
</tr>
<tr>
  <th>currency_risk</th>             <td>    0.0017</td> <td>    0.001</td> <td>    1.685</td> <td> 0.093</td> <td>   -0.000</td> <td>    0.004</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>123.217</td> <th>  Durbin-Watson:     </th> <td>   2.001</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 415.489</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.567</td>  <th>  Prob(JB):          </th> <td>5.99e-91</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.320</td>  <th>  Cond. No.          </th> <td>    32.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sm_ols('M02230323mret ~ commodity_risk_energy + commodity_risk_government+commodity_risk_material+credit_risk+currency_risk',
       data=final_01).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>M02230323mret</td>  <th>  R-squared:         </th> <td>   0.133</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.120</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   10.54</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 05 Apr 2022</td> <th>  Prob (F-statistic):</th> <td>1.99e-09</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:23:09</td>     <th>  Log-Likelihood:    </th> <td>  174.38</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   350</td>      <th>  AIC:               </th> <td>  -336.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   344</td>      <th>  BIC:               </th> <td>  -313.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>   -0.3709</td> <td>    0.018</td> <td>  -21.133</td> <td> 0.000</td> <td>   -0.405</td> <td>   -0.336</td>
</tr>
<tr>
  <th>commodity_risk_energy</th>     <td>   -0.0051</td> <td>    0.001</td> <td>   -4.651</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.003</td>
</tr>
<tr>
  <th>commodity_risk_government</th> <td>    0.0043</td> <td>    0.001</td> <td>    4.422</td> <td> 0.000</td> <td>    0.002</td> <td>    0.006</td>
</tr>
<tr>
  <th>commodity_risk_material</th>   <td>    0.0013</td> <td>    0.002</td> <td>    0.664</td> <td> 0.507</td> <td>   -0.002</td> <td>    0.005</td>
</tr>
<tr>
  <th>credit_risk</th>               <td>   -0.0026</td> <td>    0.001</td> <td>   -1.838</td> <td> 0.067</td> <td>   -0.005</td> <td>    0.000</td>
</tr>
<tr>
  <th>currency_risk</th>             <td>    0.0006</td> <td>    0.002</td> <td>    0.366</td> <td> 0.714</td> <td>   -0.003</td> <td>    0.004</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.286</td> <th>  Durbin-Watson:     </th> <td>   1.937</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.016</td> <th>  Jarque-Bera (JB):  </th> <td>  13.742</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.063</td> <th>  Prob(JB):          </th> <td> 0.00104</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.963</td> <th>  Cond. No.          </th> <td>    32.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
sm_ols('D0324ret ~ commodity_risk_energy + commodity_risk_government+commodity_risk_material+credit_risk+currency_risk',
       data=final_01).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>D0324ret</td>     <th>  R-squared:         </th> <td>   0.050</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.036</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.642</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 05 Apr 2022</td> <th>  Prob (F-statistic):</th>  <td>0.00318</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:23:09</td>     <th>  Log-Likelihood:    </th> <td>  433.05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   350</td>      <th>  AIC:               </th> <td>  -854.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   344</td>      <th>  BIC:               </th> <td>  -830.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>    0.1185</td> <td>    0.008</td> <td>   14.137</td> <td> 0.000</td> <td>    0.102</td> <td>    0.135</td>
</tr>
<tr>
  <th>commodity_risk_energy</th>     <td>    0.0004</td> <td>    0.001</td> <td>    0.833</td> <td> 0.405</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>commodity_risk_government</th> <td>   -0.0017</td> <td>    0.000</td> <td>   -3.705</td> <td> 0.000</td> <td>   -0.003</td> <td>   -0.001</td>
</tr>
<tr>
  <th>commodity_risk_material</th>   <td>    0.0004</td> <td>    0.001</td> <td>    0.475</td> <td> 0.635</td> <td>   -0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>credit_risk</th>               <td>    0.0012</td> <td>    0.001</td> <td>    1.758</td> <td> 0.080</td> <td>   -0.000</td> <td>    0.002</td>
</tr>
<tr>
  <th>currency_risk</th>             <td>   -0.0007</td> <td>    0.001</td> <td>   -0.870</td> <td> 0.385</td> <td>   -0.002</td> <td>    0.001</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>53.612</td> <th>  Durbin-Watson:     </th> <td>   1.923</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  88.470</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.906</td> <th>  Prob(JB):          </th> <td>6.15e-20</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.668</td> <th>  Cond. No.          </th> <td>    32.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### It does not change too much about the results by the regression 


```python

```
