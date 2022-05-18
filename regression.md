## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important


```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col
housing_train = pd.read_csv('input_data2/housing_train.csv')
```


```python
housing_train.shape
```




    (1941, 81)




```python
housing_train.columns
```




    Index(['parcel', 'v_MS_SubClass', 'v_MS_Zoning', 'v_Lot_Frontage',
           'v_Lot_Area', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour',
           'v_Utilities', 'v_Lot_Config', 'v_Land_Slope', 'v_Neighborhood',
           'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
           'v_Overall_Qual', 'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add',
           'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd',
           'v_Mas_Vnr_Type', 'v_Mas_Vnr_Area', 'v_Exter_Qual', 'v_Exter_Cond',
           'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
           'v_BsmtFin_Type_1', 'v_BsmtFin_SF_1', 'v_BsmtFin_Type_2',
           'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_Heating',
           'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_1st_Flr_SF',
           'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
           'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
           'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_Kitchen_Qual',
           'v_TotRms_AbvGrd', 'v_Functional', 'v_Fireplaces', 'v_Fireplace_Qu',
           'v_Garage_Type', 'v_Garage_Yr_Blt', 'v_Garage_Finish', 'v_Garage_Cars',
           'v_Garage_Area', 'v_Garage_Qual', 'v_Garage_Cond', 'v_Paved_Drive',
           'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch',
           'v_Screen_Porch', 'v_Pool_Area', 'v_Pool_QC', 'v_Fence',
           'v_Misc_Feature', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_Sale_Type',
           'v_Sale_Condition', 'v_SalePrice'],
          dtype='object')




```python
housing_train.describe().T.head()
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
      <th>v_MS_SubClass</th>
      <td>1941.0</td>
      <td>58.088614</td>
      <td>42.946015</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>50.0</td>
      <td>70.0</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>v_Lot_Frontage</th>
      <td>1620.0</td>
      <td>69.301235</td>
      <td>23.978101</td>
      <td>21.0</td>
      <td>58.0</td>
      <td>68.0</td>
      <td>80.0</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>v_Lot_Area</th>
      <td>1941.0</td>
      <td>10284.770222</td>
      <td>7832.295527</td>
      <td>1470.0</td>
      <td>7420.0</td>
      <td>9450.0</td>
      <td>11631.0</td>
      <td>164660.0</td>
    </tr>
    <tr>
      <th>v_Overall_Qual</th>
      <td>1941.0</td>
      <td>6.113344</td>
      <td>1.401594</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>v_Overall_Cond</th>
      <td>1941.0</td>
      <td>5.568264</td>
      <td>1.087465</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def outlier_report(df,vars_to_examine=None,color='red',thres=4,
                   return_df=False,no_print=False):
    '''
    Parameters
    ----------
    df : DATAFRAME
        Input dataframe
    vars_to_examine : LIST, optional
        List of variables to examine from dataframe. The default is df.columns.
    color : STRING, optional
        Color for cell highlighting. The default is 'red'.
    thres : int, optional
        Highlight cells where z score is above thres. The default is 4.
    return_df : Boolean, optional
        If true, will return the df obj (without styling) for further use. 
        The default is False.
    no_print : Boolean, optional
        If true, will not print. 
        The default is False.
    
    Displays (if no_print=False)
    -------
    Table with distribution of z-scores of variables of interest. 
    
    Returns (if return_df=True)
    -------
    Table with distribution of z-scores of variables of interest (without styling).     
    '''
        
    def highlight_extreme(s):
        '''
        Highlight extreme values in a series.
        '''
        is_extreme = abs(s) > thres
        return ['background-color: '+color if v else '' for v in is_extreme]
    
    if vars_to_examine==None:
        vars_to_examine=df.columns
    
    _tab = (
            # compute z scores
            ((df[vars_to_examine] - df[vars_to_examine].mean())/df[vars_to_examine].std())
            # output dist of z   
            .describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
            # add a new column = highest of min and max column
            .assign(max_z_abs = lambda x: x[['min','max']].abs().max(axis=1))
            # now sort on it
            .sort_values('max_z_abs',ascending = False)
    )
    
    if no_print == False:
        
        fdict = { c:('{:,.2f}' if c != 'count' else  '{:,.0f}') for c in _tab.columns   }

        display(_tab
             .style.format(fdict)
                   .apply(highlight_extreme, 
                          subset=['mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%','99%', 'max', 'max_z_abs'])
        ) 
    
    if return_df == True:
        return _tab
```


```python
e=housing_train.drop(columns=['parcel', 'v_MS_Zoning', 'v_Street', 'v_Alley', 'v_Lot_Shape',
       'v_Land_Contour', 'v_Utilities', 'v_Lot_Config', 'v_Land_Slope',
       'v_Neighborhood', 'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type',
       'v_House_Style', 'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st',
       'v_Exterior_2nd', 'v_Mas_Vnr_Type', 'v_Exter_Qual', 'v_Exter_Cond',
       'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
       'v_BsmtFin_Type_1', 'v_BsmtFin_Type_2', 'v_Heating', 'v_Heating_QC',
       'v_Central_Air', 'v_Electrical', 'v_Kitchen_Qual', 'v_Functional',
       'v_Fireplace_Qu', 'v_Garage_Type', 'v_Garage_Finish', 'v_Garage_Qual',
       'v_Garage_Cond', 'v_Paved_Drive', 'v_Pool_QC', 'v_Fence',
       'v_Misc_Feature', 'v_Sale_Type', 'v_Sale_Condition'])
e.columns
```




    Index(['v_MS_SubClass', 'v_Lot_Frontage', 'v_Lot_Area', 'v_Overall_Qual',
           'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add', 'v_Mas_Vnr_Area',
           'v_BsmtFin_SF_1', 'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF',
           'v_1st_Flr_SF', 'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
           'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
           'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_TotRms_AbvGrd', 'v_Fireplaces',
           'v_Garage_Yr_Blt', 'v_Garage_Cars', 'v_Garage_Area', 'v_Wood_Deck_SF',
           'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch', 'v_Screen_Porch',
           'v_Pool_Area', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_SalePrice'],
          dtype='object')




```python
vars_to_check = ['v_MS_SubClass', 'v_Lot_Frontage', 'v_Lot_Area', 'v_Overall_Qual',
       'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add', 'v_Mas_Vnr_Area',
       'v_BsmtFin_SF_1', 'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF',
       'v_1st_Flr_SF', 'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
       'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
       'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_TotRms_AbvGrd', 'v_Fireplaces',
       'v_Garage_Yr_Blt', 'v_Garage_Cars', 'v_Garage_Area', 'v_Wood_Deck_SF',
       'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch', 'v_Screen_Porch',
       'v_Pool_Area', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_SalePrice']

outlier_report(e,vars_to_check,thres=4)
```

<table id="T_69029_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >count</th>
      <th class="col_heading level0 col1" >mean</th>
      <th class="col_heading level0 col2" >std</th>
      <th class="col_heading level0 col3" >min</th>
      <th class="col_heading level0 col4" >1%</th>
      <th class="col_heading level0 col5" >5%</th>
      <th class="col_heading level0 col6" >25%</th>
      <th class="col_heading level0 col7" >50%</th>
      <th class="col_heading level0 col8" >75%</th>
      <th class="col_heading level0 col9" >95%</th>
      <th class="col_heading level0 col10" >99%</th>
      <th class="col_heading level0 col11" >max</th>
      <th class="col_heading level0 col12" >max_z_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_69029_level0_row0" class="row_heading level0 row0" >v_Misc_Val</th>
      <td id="T_69029_row0_col0" class="data row0 col0" >1,941</td>
      <td id="T_69029_row0_col1" class="data row0 col1" >-0.00</td>
      <td id="T_69029_row0_col2" class="data row0 col2" >1.00</td>
      <td id="T_69029_row0_col3" class="data row0 col3" >-0.09</td>
      <td id="T_69029_row0_col4" class="data row0 col4" >-0.09</td>
      <td id="T_69029_row0_col5" class="data row0 col5" >-0.09</td>
      <td id="T_69029_row0_col6" class="data row0 col6" >-0.09</td>
      <td id="T_69029_row0_col7" class="data row0 col7" >-0.09</td>
      <td id="T_69029_row0_col8" class="data row0 col8" >-0.09</td>
      <td id="T_69029_row0_col9" class="data row0 col9" >-0.09</td>
      <td id="T_69029_row0_col10" class="data row0 col10" >1.38</td>
      <td id="T_69029_row0_col11" class="data row0 col11" >27.51</td>
      <td id="T_69029_row0_col12" class="data row0 col12" >27.51</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row1" class="row_heading level0 row1" >v_Lot_Area</th>
      <td id="T_69029_row1_col0" class="data row1 col0" >1,941</td>
      <td id="T_69029_row1_col1" class="data row1 col1" >-0.00</td>
      <td id="T_69029_row1_col2" class="data row1 col2" >1.00</td>
      <td id="T_69029_row1_col3" class="data row1 col3" >-1.13</td>
      <td id="T_69029_row1_col4" class="data row1 col4" >-1.10</td>
      <td id="T_69029_row1_col5" class="data row1 col5" >-0.86</td>
      <td id="T_69029_row1_col6" class="data row1 col6" >-0.37</td>
      <td id="T_69029_row1_col7" class="data row1 col7" >-0.11</td>
      <td id="T_69029_row1_col8" class="data row1 col8" >0.17</td>
      <td id="T_69029_row1_col9" class="data row1 col9" >0.96</td>
      <td id="T_69029_row1_col10" class="data row1 col10" >3.55</td>
      <td id="T_69029_row1_col11" class="data row1 col11" >19.71</td>
      <td id="T_69029_row1_col12" class="data row1 col12" >19.71</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row2" class="row_heading level0 row2" >v_Pool_Area</th>
      <td id="T_69029_row2_col0" class="data row2 col0" >1,941</td>
      <td id="T_69029_row2_col1" class="data row2 col1" >0.00</td>
      <td id="T_69029_row2_col2" class="data row2 col2" >1.00</td>
      <td id="T_69029_row2_col3" class="data row2 col3" >-0.08</td>
      <td id="T_69029_row2_col4" class="data row2 col4" >-0.08</td>
      <td id="T_69029_row2_col5" class="data row2 col5" >-0.08</td>
      <td id="T_69029_row2_col6" class="data row2 col6" >-0.08</td>
      <td id="T_69029_row2_col7" class="data row2 col7" >-0.08</td>
      <td id="T_69029_row2_col8" class="data row2 col8" >-0.08</td>
      <td id="T_69029_row2_col9" class="data row2 col9" >-0.08</td>
      <td id="T_69029_row2_col10" class="data row2 col10" >-0.08</td>
      <td id="T_69029_row2_col11" class="data row2 col11" >18.23</td>
      <td id="T_69029_row2_col12" class="data row2 col12" >18.23</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row3" class="row_heading level0 row3" >v_3Ssn_Porch</th>
      <td id="T_69029_row3_col0" class="data row3 col0" >1,941</td>
      <td id="T_69029_row3_col1" class="data row3 col1" >0.00</td>
      <td id="T_69029_row3_col2" class="data row3 col2" >1.00</td>
      <td id="T_69029_row3_col3" class="data row3 col3" >-0.10</td>
      <td id="T_69029_row3_col4" class="data row3 col4" >-0.10</td>
      <td id="T_69029_row3_col5" class="data row3 col5" >-0.10</td>
      <td id="T_69029_row3_col6" class="data row3 col6" >-0.10</td>
      <td id="T_69029_row3_col7" class="data row3 col7" >-0.10</td>
      <td id="T_69029_row3_col8" class="data row3 col8" >-0.10</td>
      <td id="T_69029_row3_col9" class="data row3 col9" >-0.10</td>
      <td id="T_69029_row3_col10" class="data row3 col10" >4.82</td>
      <td id="T_69029_row3_col11" class="data row3 col11" >18.06</td>
      <td id="T_69029_row3_col12" class="data row3 col12" >18.06</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row4" class="row_heading level0 row4" >v_Low_Qual_Fin_SF</th>
      <td id="T_69029_row4_col0" class="data row4 col0" >1,941</td>
      <td id="T_69029_row4_col1" class="data row4 col1" >0.00</td>
      <td id="T_69029_row4_col2" class="data row4 col2" >1.00</td>
      <td id="T_69029_row4_col3" class="data row4 col3" >-0.10</td>
      <td id="T_69029_row4_col4" class="data row4 col4" >-0.10</td>
      <td id="T_69029_row4_col5" class="data row4 col5" >-0.10</td>
      <td id="T_69029_row4_col6" class="data row4 col6" >-0.10</td>
      <td id="T_69029_row4_col7" class="data row4 col7" >-0.10</td>
      <td id="T_69029_row4_col8" class="data row4 col8" >-0.10</td>
      <td id="T_69029_row4_col9" class="data row4 col9" >-0.10</td>
      <td id="T_69029_row4_col10" class="data row4 col10" >2.50</td>
      <td id="T_69029_row4_col11" class="data row4 col11" >16.13</td>
      <td id="T_69029_row4_col12" class="data row4 col12" >16.13</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row5" class="row_heading level0 row5" >v_Enclosed_Porch</th>
      <td id="T_69029_row5_col0" class="data row5 col0" >1,941</td>
      <td id="T_69029_row5_col1" class="data row5 col1" >0.00</td>
      <td id="T_69029_row5_col2" class="data row5 col2" >1.00</td>
      <td id="T_69029_row5_col3" class="data row5 col3" >-0.35</td>
      <td id="T_69029_row5_col4" class="data row5 col4" >-0.35</td>
      <td id="T_69029_row5_col5" class="data row5 col5" >-0.35</td>
      <td id="T_69029_row5_col6" class="data row5 col6" >-0.35</td>
      <td id="T_69029_row5_col7" class="data row5 col7" >-0.35</td>
      <td id="T_69029_row5_col8" class="data row5 col8" >-0.35</td>
      <td id="T_69029_row5_col9" class="data row5 col9" >2.41</td>
      <td id="T_69029_row5_col10" class="data row5 col10" >3.66</td>
      <td id="T_69029_row5_col11" class="data row5 col11" >15.16</td>
      <td id="T_69029_row5_col12" class="data row5 col12" >15.16</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row6" class="row_heading level0 row6" >v_Total_Bsmt_SF</th>
      <td id="T_69029_row6_col0" class="data row6 col0" >1,940</td>
      <td id="T_69029_row6_col1" class="data row6 col1" >-0.00</td>
      <td id="T_69029_row6_col2" class="data row6 col2" >1.00</td>
      <td id="T_69029_row6_col3" class="data row6 col3" >-2.40</td>
      <td id="T_69029_row6_col4" class="data row6 col4" >-2.40</td>
      <td id="T_69029_row6_col5" class="data row6 col5" >-1.30</td>
      <td id="T_69029_row6_col6" class="data row6 col6" >-0.59</td>
      <td id="T_69029_row6_col7" class="data row6 col7" >-0.15</td>
      <td id="T_69029_row6_col8" class="data row6 col8" >0.55</td>
      <td id="T_69029_row6_col9" class="data row6 col9" >1.65</td>
      <td id="T_69029_row6_col10" class="data row6 col10" >2.47</td>
      <td id="T_69029_row6_col11" class="data row6 col11" >11.53</td>
      <td id="T_69029_row6_col12" class="data row6 col12" >11.53</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row7" class="row_heading level0 row7" >v_BsmtFin_SF_1</th>
      <td id="T_69029_row7_col0" class="data row7 col0" >1,940</td>
      <td id="T_69029_row7_col1" class="data row7 col1" >-0.00</td>
      <td id="T_69029_row7_col2" class="data row7 col2" >1.00</td>
      <td id="T_69029_row7_col3" class="data row7 col3" >-0.95</td>
      <td id="T_69029_row7_col4" class="data row7 col4" >-0.95</td>
      <td id="T_69029_row7_col5" class="data row7 col5" >-0.95</td>
      <td id="T_69029_row7_col6" class="data row7 col6" >-0.95</td>
      <td id="T_69029_row7_col7" class="data row7 col7" >-0.16</td>
      <td id="T_69029_row7_col8" class="data row7 col8" >0.65</td>
      <td id="T_69029_row7_col9" class="data row7 col9" >1.77</td>
      <td id="T_69029_row7_col10" class="data row7 col10" >2.54</td>
      <td id="T_69029_row7_col11" class="data row7 col11" >11.37</td>
      <td id="T_69029_row7_col12" class="data row7 col12" >11.37</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row8" class="row_heading level0 row8" >v_Wood_Deck_SF</th>
      <td id="T_69029_row8_col0" class="data row8 col0" >1,941</td>
      <td id="T_69029_row8_col1" class="data row8 col1" >0.00</td>
      <td id="T_69029_row8_col2" class="data row8 col2" >1.00</td>
      <td id="T_69029_row8_col3" class="data row8 col3" >-0.73</td>
      <td id="T_69029_row8_col4" class="data row8 col4" >-0.73</td>
      <td id="T_69029_row8_col5" class="data row8 col5" >-0.73</td>
      <td id="T_69029_row8_col6" class="data row8 col6" >-0.73</td>
      <td id="T_69029_row8_col7" class="data row8 col7" >-0.73</td>
      <td id="T_69029_row8_col8" class="data row8 col8" >0.59</td>
      <td id="T_69029_row8_col9" class="data row8 col9" >1.79</td>
      <td id="T_69029_row8_col10" class="data row8 col10" >3.33</td>
      <td id="T_69029_row8_col11" class="data row8 col11" >10.48</td>
      <td id="T_69029_row8_col12" class="data row8 col12" >10.48</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row9" class="row_heading level0 row9" >v_Lot_Frontage</th>
      <td id="T_69029_row9_col0" class="data row9 col0" >1,620</td>
      <td id="T_69029_row9_col1" class="data row9 col1" >-0.00</td>
      <td id="T_69029_row9_col2" class="data row9 col2" >1.00</td>
      <td id="T_69029_row9_col3" class="data row9 col3" >-2.01</td>
      <td id="T_69029_row9_col4" class="data row9 col4" >-2.01</td>
      <td id="T_69029_row9_col5" class="data row9 col5" >-1.47</td>
      <td id="T_69029_row9_col6" class="data row9 col6" >-0.47</td>
      <td id="T_69029_row9_col7" class="data row9 col7" >-0.05</td>
      <td id="T_69029_row9_col8" class="data row9 col8" >0.45</td>
      <td id="T_69029_row9_col9" class="data row9 col9" >1.57</td>
      <td id="T_69029_row9_col10" class="data row9 col10" >2.77</td>
      <td id="T_69029_row9_col11" class="data row9 col11" >10.16</td>
      <td id="T_69029_row9_col12" class="data row9 col12" >10.16</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row10" class="row_heading level0 row10" >v_1st_Flr_SF</th>
      <td id="T_69029_row10_col0" class="data row10 col0" >1,941</td>
      <td id="T_69029_row10_col1" class="data row10 col1" >0.00</td>
      <td id="T_69029_row10_col2" class="data row10 col2" >1.00</td>
      <td id="T_69029_row10_col3" class="data row10 col3" >-2.08</td>
      <td id="T_69029_row10_col4" class="data row10 col4" >-1.65</td>
      <td id="T_69029_row10_col5" class="data row10 col5" >-1.23</td>
      <td id="T_69029_row10_col6" class="data row10 col6" >-0.69</td>
      <td id="T_69029_row10_col7" class="data row10 col7" >-0.19</td>
      <td id="T_69029_row10_col8" class="data row10 col8" >0.56</td>
      <td id="T_69029_row10_col9" class="data row10 col9" >1.68</td>
      <td id="T_69029_row10_col10" class="data row10 col10" >2.81</td>
      <td id="T_69029_row10_col11" class="data row10 col11" >9.91</td>
      <td id="T_69029_row10_col12" class="data row10 col12" >9.91</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row11" class="row_heading level0 row11" >v_Screen_Porch</th>
      <td id="T_69029_row11_col0" class="data row11 col0" >1,941</td>
      <td id="T_69029_row11_col1" class="data row11 col1" >-0.00</td>
      <td id="T_69029_row11_col2" class="data row11 col2" >1.00</td>
      <td id="T_69029_row11_col3" class="data row11 col3" >-0.29</td>
      <td id="T_69029_row11_col4" class="data row11 col4" >-0.29</td>
      <td id="T_69029_row11_col5" class="data row11 col5" >-0.29</td>
      <td id="T_69029_row11_col6" class="data row11 col6" >-0.29</td>
      <td id="T_69029_row11_col7" class="data row11 col7" >-0.29</td>
      <td id="T_69029_row11_col8" class="data row11 col8" >-0.29</td>
      <td id="T_69029_row11_col9" class="data row11 col9" >2.57</td>
      <td id="T_69029_row11_col10" class="data row11 col10" >4.36</td>
      <td id="T_69029_row11_col11" class="data row11 col11" >9.86</td>
      <td id="T_69029_row11_col12" class="data row11 col12" >9.86</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row12" class="row_heading level0 row12" >v_Open_Porch_SF</th>
      <td id="T_69029_row12_col0" class="data row12 col0" >1,941</td>
      <td id="T_69029_row12_col1" class="data row12 col1" >0.00</td>
      <td id="T_69029_row12_col2" class="data row12 col2" >1.00</td>
      <td id="T_69029_row12_col3" class="data row12 col3" >-0.70</td>
      <td id="T_69029_row12_col4" class="data row12 col4" >-0.70</td>
      <td id="T_69029_row12_col5" class="data row12 col5" >-0.70</td>
      <td id="T_69029_row12_col6" class="data row12 col6" >-0.70</td>
      <td id="T_69029_row12_col7" class="data row12 col7" >-0.30</td>
      <td id="T_69029_row12_col8" class="data row12 col8" >0.32</td>
      <td id="T_69029_row12_col9" class="data row12 col9" >1.99</td>
      <td id="T_69029_row12_col10" class="data row12 col10" >3.51</td>
      <td id="T_69029_row12_col11" class="data row12 col11" >9.86</td>
      <td id="T_69029_row12_col12" class="data row12 col12" >9.86</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row13" class="row_heading level0 row13" >v_Garage_Yr_Blt</th>
      <td id="T_69029_row13_col0" class="data row13 col0" >1,834</td>
      <td id="T_69029_row13_col1" class="data row13 col1" >-0.00</td>
      <td id="T_69029_row13_col2" class="data row13 col2" >1.00</td>
      <td id="T_69029_row13_col3" class="data row13 col3" >-3.23</td>
      <td id="T_69029_row13_col4" class="data row13 col4" >-2.42</td>
      <td id="T_69029_row13_col5" class="data row13 col5" >-1.93</td>
      <td id="T_69029_row13_col6" class="data row13 col6" >-0.71</td>
      <td id="T_69029_row13_col7" class="data row13 col7" >0.07</td>
      <td id="T_69029_row13_col8" class="data row13 col8" >0.93</td>
      <td id="T_69029_row13_col9" class="data row13 col9" >1.12</td>
      <td id="T_69029_row13_col10" class="data row13 col10" >1.16</td>
      <td id="T_69029_row13_col11" class="data row13 col11" >8.89</td>
      <td id="T_69029_row13_col12" class="data row13 col12" >8.89</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row14" class="row_heading level0 row14" >v_BsmtFin_SF_2</th>
      <td id="T_69029_row14_col0" class="data row14 col0" >1,940</td>
      <td id="T_69029_row14_col1" class="data row14 col1" >-0.00</td>
      <td id="T_69029_row14_col2" class="data row14 col2" >1.00</td>
      <td id="T_69029_row14_col3" class="data row14 col3" >-0.29</td>
      <td id="T_69029_row14_col4" class="data row14 col4" >-0.29</td>
      <td id="T_69029_row14_col5" class="data row14 col5" >-0.29</td>
      <td id="T_69029_row14_col6" class="data row14 col6" >-0.29</td>
      <td id="T_69029_row14_col7" class="data row14 col7" >-0.29</td>
      <td id="T_69029_row14_col8" class="data row14 col8" >-0.29</td>
      <td id="T_69029_row14_col9" class="data row14 col9" >2.28</td>
      <td id="T_69029_row14_col10" class="data row14 col10" >4.92</td>
      <td id="T_69029_row14_col11" class="data row14 col11" >8.40</td>
      <td id="T_69029_row14_col12" class="data row14 col12" >8.40</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row15" class="row_heading level0 row15" >v_Mas_Vnr_Area</th>
      <td id="T_69029_row15_col0" class="data row15 col0" >1,923</td>
      <td id="T_69029_row15_col1" class="data row15 col1" >-0.00</td>
      <td id="T_69029_row15_col2" class="data row15 col2" >1.00</td>
      <td id="T_69029_row15_col3" class="data row15 col3" >-0.57</td>
      <td id="T_69029_row15_col4" class="data row15 col4" >-0.57</td>
      <td id="T_69029_row15_col5" class="data row15 col5" >-0.57</td>
      <td id="T_69029_row15_col6" class="data row15 col6" >-0.57</td>
      <td id="T_69029_row15_col7" class="data row15 col7" >-0.57</td>
      <td id="T_69029_row15_col8" class="data row15 col8" >0.34</td>
      <td id="T_69029_row15_col9" class="data row15 col9" >1.99</td>
      <td id="T_69029_row15_col10" class="data row15 col10" >3.73</td>
      <td id="T_69029_row15_col11" class="data row15 col11" >8.08</td>
      <td id="T_69029_row15_col12" class="data row15 col12" >8.08</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row16" class="row_heading level0 row16" >v_Gr_Liv_Area</th>
      <td id="T_69029_row16_col0" class="data row16 col0" >1,941</td>
      <td id="T_69029_row16_col1" class="data row16 col1" >-0.00</td>
      <td id="T_69029_row16_col2" class="data row16 col2" >1.00</td>
      <td id="T_69029_row16_col3" class="data row16 col3" >-2.23</td>
      <td id="T_69029_row16_col4" class="data row16 col4" >-1.55</td>
      <td id="T_69029_row16_col5" class="data row16 col5" >-1.22</td>
      <td id="T_69029_row16_col6" class="data row16 col6" >-0.74</td>
      <td id="T_69029_row16_col7" class="data row16 col7" >-0.13</td>
      <td id="T_69029_row16_col8" class="data row16 col8" >0.47</td>
      <td id="T_69029_row16_col9" class="data row16 col9" >1.89</td>
      <td id="T_69029_row16_col10" class="data row16 col10" >2.90</td>
      <td id="T_69029_row16_col11" class="data row16 col11" >7.88</td>
      <td id="T_69029_row16_col12" class="data row16 col12" >7.88</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row17" class="row_heading level0 row17" >v_Bsmt_Half_Bath</th>
      <td id="T_69029_row17_col0" class="data row17 col0" >1,939</td>
      <td id="T_69029_row17_col1" class="data row17 col1" >-0.00</td>
      <td id="T_69029_row17_col2" class="data row17 col2" >1.00</td>
      <td id="T_69029_row17_col3" class="data row17 col3" >-0.26</td>
      <td id="T_69029_row17_col4" class="data row17 col4" >-0.26</td>
      <td id="T_69029_row17_col5" class="data row17 col5" >-0.26</td>
      <td id="T_69029_row17_col6" class="data row17 col6" >-0.26</td>
      <td id="T_69029_row17_col7" class="data row17 col7" >-0.26</td>
      <td id="T_69029_row17_col8" class="data row17 col8" >-0.26</td>
      <td id="T_69029_row17_col9" class="data row17 col9" >3.67</td>
      <td id="T_69029_row17_col10" class="data row17 col10" >3.67</td>
      <td id="T_69029_row17_col11" class="data row17 col11" >7.59</td>
      <td id="T_69029_row17_col12" class="data row17 col12" >7.59</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row18" class="row_heading level0 row18" >v_SalePrice</th>
      <td id="T_69029_row18_col0" class="data row18 col0" >1,941</td>
      <td id="T_69029_row18_col1" class="data row18 col1" >-0.00</td>
      <td id="T_69029_row18_col2" class="data row18 col2" >1.00</td>
      <td id="T_69029_row18_col3" class="data row18 col3" >-2.10</td>
      <td id="T_69029_row18_col4" class="data row18 col4" >-1.46</td>
      <td id="T_69029_row18_col5" class="data row18 col5" >-1.15</td>
      <td id="T_69029_row18_col6" class="data row18 col6" >-0.65</td>
      <td id="T_69029_row18_col7" class="data row18 col7" >-0.25</td>
      <td id="T_69029_row18_col8" class="data row18 col8" >0.41</td>
      <td id="T_69029_row18_col9" class="data row18 col9" >1.96</td>
      <td id="T_69029_row18_col10" class="data row18 col10" >3.37</td>
      <td id="T_69029_row18_col11" class="data row18 col11" >7.13</td>
      <td id="T_69029_row18_col12" class="data row18 col12" >7.13</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row19" class="row_heading level0 row19" >v_Bedroom_AbvGr</th>
      <td id="T_69029_row19_col0" class="data row19 col0" >1,941</td>
      <td id="T_69029_row19_col1" class="data row19 col1" >0.00</td>
      <td id="T_69029_row19_col2" class="data row19 col2" >1.00</td>
      <td id="T_69029_row19_col3" class="data row19 col3" >-3.46</td>
      <td id="T_69029_row19_col4" class="data row19 col4" >-2.25</td>
      <td id="T_69029_row19_col5" class="data row19 col5" >-1.05</td>
      <td id="T_69029_row19_col6" class="data row19 col6" >-1.05</td>
      <td id="T_69029_row19_col7" class="data row19 col7" >0.16</td>
      <td id="T_69029_row19_col8" class="data row19 col8" >0.16</td>
      <td id="T_69029_row19_col9" class="data row19 col9" >1.37</td>
      <td id="T_69029_row19_col10" class="data row19 col10" >2.58</td>
      <td id="T_69029_row19_col11" class="data row19 col11" >6.20</td>
      <td id="T_69029_row19_col12" class="data row19 col12" >6.20</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row20" class="row_heading level0 row20" >v_TotRms_AbvGrd</th>
      <td id="T_69029_row20_col0" class="data row20 col0" >1,941</td>
      <td id="T_69029_row20_col1" class="data row20 col1" >0.00</td>
      <td id="T_69029_row20_col2" class="data row20 col2" >1.00</td>
      <td id="T_69029_row20_col3" class="data row20 col3" >-2.83</td>
      <td id="T_69029_row20_col4" class="data row20 col4" >-1.56</td>
      <td id="T_69029_row20_col5" class="data row20 col5" >-1.56</td>
      <td id="T_69029_row20_col6" class="data row20 col6" >-0.93</td>
      <td id="T_69029_row20_col7" class="data row20 col7" >-0.29</td>
      <td id="T_69029_row20_col8" class="data row20 col8" >0.34</td>
      <td id="T_69029_row20_col9" class="data row20 col9" >1.61</td>
      <td id="T_69029_row20_col10" class="data row20 col10" >2.87</td>
      <td id="T_69029_row20_col11" class="data row20 col11" >5.41</td>
      <td id="T_69029_row20_col12" class="data row20 col12" >5.41</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row21" class="row_heading level0 row21" >v_Fireplaces</th>
      <td id="T_69029_row21_col0" class="data row21 col0" >1,941</td>
      <td id="T_69029_row21_col1" class="data row21 col1" >0.00</td>
      <td id="T_69029_row21_col2" class="data row21 col2" >1.00</td>
      <td id="T_69029_row21_col3" class="data row21 col3" >-0.93</td>
      <td id="T_69029_row21_col4" class="data row21 col4" >-0.93</td>
      <td id="T_69029_row21_col5" class="data row21 col5" >-0.93</td>
      <td id="T_69029_row21_col6" class="data row21 col6" >-0.93</td>
      <td id="T_69029_row21_col7" class="data row21 col7" >0.63</td>
      <td id="T_69029_row21_col8" class="data row21 col8" >0.63</td>
      <td id="T_69029_row21_col9" class="data row21 col9" >2.19</td>
      <td id="T_69029_row21_col10" class="data row21 col10" >2.19</td>
      <td id="T_69029_row21_col11" class="data row21 col11" >5.30</td>
      <td id="T_69029_row21_col12" class="data row21 col12" >5.30</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row22" class="row_heading level0 row22" >v_Kitchen_AbvGr</th>
      <td id="T_69029_row22_col0" class="data row22 col0" >1,941</td>
      <td id="T_69029_row22_col1" class="data row22 col1" >-0.00</td>
      <td id="T_69029_row22_col2" class="data row22 col2" >1.00</td>
      <td id="T_69029_row22_col3" class="data row22 col3" >-5.15</td>
      <td id="T_69029_row22_col4" class="data row22 col4" >-0.19</td>
      <td id="T_69029_row22_col5" class="data row22 col5" >-0.19</td>
      <td id="T_69029_row22_col6" class="data row22 col6" >-0.19</td>
      <td id="T_69029_row22_col7" class="data row22 col7" >-0.19</td>
      <td id="T_69029_row22_col8" class="data row22 col8" >-0.19</td>
      <td id="T_69029_row22_col9" class="data row22 col9" >-0.19</td>
      <td id="T_69029_row22_col10" class="data row22 col10" >4.76</td>
      <td id="T_69029_row22_col11" class="data row22 col11" >4.76</td>
      <td id="T_69029_row22_col12" class="data row22 col12" >5.15</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row23" class="row_heading level0 row23" >v_Garage_Area</th>
      <td id="T_69029_row23_col0" class="data row23 col0" >1,940</td>
      <td id="T_69029_row23_col1" class="data row23 col1" >-0.00</td>
      <td id="T_69029_row23_col2" class="data row23 col2" >1.00</td>
      <td id="T_69029_row23_col3" class="data row23 col3" >-2.18</td>
      <td id="T_69029_row23_col4" class="data row23 col4" >-2.18</td>
      <td id="T_69029_row23_col5" class="data row23 col5" >-2.18</td>
      <td id="T_69029_row23_col6" class="data row23 col6" >-0.71</td>
      <td id="T_69029_row23_col7" class="data row23 col7" >0.02</td>
      <td id="T_69029_row23_col8" class="data row23 col8" >0.48</td>
      <td id="T_69029_row23_col9" class="data row23 col9" >1.78</td>
      <td id="T_69029_row23_col10" class="data row23 col10" >2.62</td>
      <td id="T_69029_row23_col11" class="data row23 col11" >4.68</td>
      <td id="T_69029_row23_col12" class="data row23 col12" >4.68</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row24" class="row_heading level0 row24" >v_Overall_Cond</th>
      <td id="T_69029_row24_col0" class="data row24 col0" >1,941</td>
      <td id="T_69029_row24_col1" class="data row24 col1" >0.00</td>
      <td id="T_69029_row24_col2" class="data row24 col2" >1.00</td>
      <td id="T_69029_row24_col3" class="data row24 col3" >-4.20</td>
      <td id="T_69029_row24_col4" class="data row24 col4" >-2.36</td>
      <td id="T_69029_row24_col5" class="data row24 col5" >-1.44</td>
      <td id="T_69029_row24_col6" class="data row24 col6" >-0.52</td>
      <td id="T_69029_row24_col7" class="data row24 col7" >-0.52</td>
      <td id="T_69029_row24_col8" class="data row24 col8" >0.40</td>
      <td id="T_69029_row24_col9" class="data row24 col9" >2.24</td>
      <td id="T_69029_row24_col10" class="data row24 col10" >3.16</td>
      <td id="T_69029_row24_col11" class="data row24 col11" >3.16</td>
      <td id="T_69029_row24_col12" class="data row24 col12" >4.20</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row25" class="row_heading level0 row25" >v_2nd_Flr_SF</th>
      <td id="T_69029_row25_col0" class="data row25 col0" >1,941</td>
      <td id="T_69029_row25_col1" class="data row25 col1" >-0.00</td>
      <td id="T_69029_row25_col2" class="data row25 col2" >1.00</td>
      <td id="T_69029_row25_col3" class="data row25 col3" >-0.79</td>
      <td id="T_69029_row25_col4" class="data row25 col4" >-0.79</td>
      <td id="T_69029_row25_col5" class="data row25 col5" >-0.79</td>
      <td id="T_69029_row25_col6" class="data row25 col6" >-0.79</td>
      <td id="T_69029_row25_col7" class="data row25 col7" >-0.79</td>
      <td id="T_69029_row25_col8" class="data row25 col8" >0.87</td>
      <td id="T_69029_row25_col9" class="data row25 col9" >1.84</td>
      <td id="T_69029_row25_col10" class="data row25 col10" >2.45</td>
      <td id="T_69029_row25_col11" class="data row25 col11" >3.97</td>
      <td id="T_69029_row25_col12" class="data row25 col12" >3.97</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row26" class="row_heading level0 row26" >v_Overall_Qual</th>
      <td id="T_69029_row26_col0" class="data row26 col0" >1,941</td>
      <td id="T_69029_row26_col1" class="data row26 col1" >-0.00</td>
      <td id="T_69029_row26_col2" class="data row26 col2" >1.00</td>
      <td id="T_69029_row26_col3" class="data row26 col3" >-3.65</td>
      <td id="T_69029_row26_col4" class="data row26 col4" >-2.22</td>
      <td id="T_69029_row26_col5" class="data row26 col5" >-1.51</td>
      <td id="T_69029_row26_col6" class="data row26 col6" >-0.79</td>
      <td id="T_69029_row26_col7" class="data row26 col7" >-0.08</td>
      <td id="T_69029_row26_col8" class="data row26 col8" >0.63</td>
      <td id="T_69029_row26_col9" class="data row26 col9" >1.35</td>
      <td id="T_69029_row26_col10" class="data row26 col10" >2.77</td>
      <td id="T_69029_row26_col11" class="data row26 col11" >2.77</td>
      <td id="T_69029_row26_col12" class="data row26 col12" >3.65</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row27" class="row_heading level0 row27" >v_Bsmt_Unf_SF</th>
      <td id="T_69029_row27_col0" class="data row27 col0" >1,940</td>
      <td id="T_69029_row27_col1" class="data row27 col1" >-0.00</td>
      <td id="T_69029_row27_col2" class="data row27 col2" >1.00</td>
      <td id="T_69029_row27_col3" class="data row27 col3" >-1.29</td>
      <td id="T_69029_row27_col4" class="data row27 col4" >-1.29</td>
      <td id="T_69029_row27_col5" class="data row27 col5" >-1.29</td>
      <td id="T_69029_row27_col6" class="data row27 col6" >-0.78</td>
      <td id="T_69029_row27_col7" class="data row27 col7" >-0.21</td>
      <td id="T_69029_row27_col8" class="data row27 col8" >0.56</td>
      <td id="T_69029_row27_col9" class="data row27 col9" >2.09</td>
      <td id="T_69029_row27_col10" class="data row27 col10" >2.75</td>
      <td id="T_69029_row27_col11" class="data row27 col11" >3.61</td>
      <td id="T_69029_row27_col12" class="data row27 col12" >3.61</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row28" class="row_heading level0 row28" >v_Year_Built</th>
      <td id="T_69029_row28_col0" class="data row28 col0" >1,941</td>
      <td id="T_69029_row28_col1" class="data row28 col1" >-0.00</td>
      <td id="T_69029_row28_col2" class="data row28 col2" >1.00</td>
      <td id="T_69029_row28_col3" class="data row28 col3" >-3.29</td>
      <td id="T_69029_row28_col4" class="data row28 col4" >-2.36</td>
      <td id="T_69029_row28_col5" class="data row28 col5" >-1.83</td>
      <td id="T_69029_row28_col6" class="data row28 col6" >-0.61</td>
      <td id="T_69029_row28_col7" class="data row28 col7" >0.06</td>
      <td id="T_69029_row28_col8" class="data row28 col8" >0.98</td>
      <td id="T_69029_row28_col9" class="data row28 col9" >1.15</td>
      <td id="T_69029_row28_col10" class="data row28 col10" >1.18</td>
      <td id="T_69029_row28_col11" class="data row28 col11" >1.21</td>
      <td id="T_69029_row28_col12" class="data row28 col12" >3.29</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row29" class="row_heading level0 row29" >v_Half_Bath</th>
      <td id="T_69029_row29_col0" class="data row29 col0" >1,941</td>
      <td id="T_69029_row29_col1" class="data row29 col1" >0.00</td>
      <td id="T_69029_row29_col2" class="data row29 col2" >1.00</td>
      <td id="T_69029_row29_col3" class="data row29 col3" >-0.76</td>
      <td id="T_69029_row29_col4" class="data row29 col4" >-0.76</td>
      <td id="T_69029_row29_col5" class="data row29 col5" >-0.76</td>
      <td id="T_69029_row29_col6" class="data row29 col6" >-0.76</td>
      <td id="T_69029_row29_col7" class="data row29 col7" >-0.76</td>
      <td id="T_69029_row29_col8" class="data row29 col8" >1.25</td>
      <td id="T_69029_row29_col9" class="data row29 col9" >1.25</td>
      <td id="T_69029_row29_col10" class="data row29 col10" >1.25</td>
      <td id="T_69029_row29_col11" class="data row29 col11" >3.25</td>
      <td id="T_69029_row29_col12" class="data row29 col12" >3.25</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row30" class="row_heading level0 row30" >v_Bsmt_Full_Bath</th>
      <td id="T_69029_row30_col0" class="data row30 col0" >1,939</td>
      <td id="T_69029_row30_col1" class="data row30 col1" >-0.00</td>
      <td id="T_69029_row30_col2" class="data row30 col2" >1.00</td>
      <td id="T_69029_row30_col3" class="data row30 col3" >-0.81</td>
      <td id="T_69029_row30_col4" class="data row30 col4" >-0.81</td>
      <td id="T_69029_row30_col5" class="data row30 col5" >-0.81</td>
      <td id="T_69029_row30_col6" class="data row30 col6" >-0.81</td>
      <td id="T_69029_row30_col7" class="data row30 col7" >-0.81</td>
      <td id="T_69029_row30_col8" class="data row30 col8" >1.13</td>
      <td id="T_69029_row30_col9" class="data row30 col9" >1.13</td>
      <td id="T_69029_row30_col10" class="data row30 col10" >3.07</td>
      <td id="T_69029_row30_col11" class="data row30 col11" >3.07</td>
      <td id="T_69029_row30_col12" class="data row30 col12" >3.07</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row31" class="row_heading level0 row31" >v_MS_SubClass</th>
      <td id="T_69029_row31_col0" class="data row31 col0" >1,941</td>
      <td id="T_69029_row31_col1" class="data row31 col1" >0.00</td>
      <td id="T_69029_row31_col2" class="data row31 col2" >1.00</td>
      <td id="T_69029_row31_col3" class="data row31 col3" >-0.89</td>
      <td id="T_69029_row31_col4" class="data row31 col4" >-0.89</td>
      <td id="T_69029_row31_col5" class="data row31 col5" >-0.89</td>
      <td id="T_69029_row31_col6" class="data row31 col6" >-0.89</td>
      <td id="T_69029_row31_col7" class="data row31 col7" >-0.19</td>
      <td id="T_69029_row31_col8" class="data row31 col8" >0.28</td>
      <td id="T_69029_row31_col9" class="data row31 col9" >2.37</td>
      <td id="T_69029_row31_col10" class="data row31 col10" >3.07</td>
      <td id="T_69029_row31_col11" class="data row31 col11" >3.07</td>
      <td id="T_69029_row31_col12" class="data row31 col12" >3.07</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row32" class="row_heading level0 row32" >v_Garage_Cars</th>
      <td id="T_69029_row32_col0" class="data row32 col0" >1,940</td>
      <td id="T_69029_row32_col1" class="data row32 col1" >-0.00</td>
      <td id="T_69029_row32_col2" class="data row32 col2" >1.00</td>
      <td id="T_69029_row32_col3" class="data row32 col3" >-2.32</td>
      <td id="T_69029_row32_col4" class="data row32 col4" >-2.32</td>
      <td id="T_69029_row32_col5" class="data row32 col5" >-2.32</td>
      <td id="T_69029_row32_col6" class="data row32 col6" >-1.01</td>
      <td id="T_69029_row32_col7" class="data row32 col7" >0.30</td>
      <td id="T_69029_row32_col8" class="data row32 col8" >0.30</td>
      <td id="T_69029_row32_col9" class="data row32 col9" >1.61</td>
      <td id="T_69029_row32_col10" class="data row32 col10" >1.61</td>
      <td id="T_69029_row32_col11" class="data row32 col11" >2.92</td>
      <td id="T_69029_row32_col12" class="data row32 col12" >2.92</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row33" class="row_heading level0 row33" >v_Full_Bath</th>
      <td id="T_69029_row33_col0" class="data row33 col0" >1,941</td>
      <td id="T_69029_row33_col1" class="data row33 col1" >-0.00</td>
      <td id="T_69029_row33_col2" class="data row33 col2" >1.00</td>
      <td id="T_69029_row33_col3" class="data row33 col3" >-2.83</td>
      <td id="T_69029_row33_col4" class="data row33 col4" >-1.03</td>
      <td id="T_69029_row33_col5" class="data row33 col5" >-1.03</td>
      <td id="T_69029_row33_col6" class="data row33 col6" >-1.03</td>
      <td id="T_69029_row33_col7" class="data row33 col7" >0.78</td>
      <td id="T_69029_row33_col8" class="data row33 col8" >0.78</td>
      <td id="T_69029_row33_col9" class="data row33 col9" >0.78</td>
      <td id="T_69029_row33_col10" class="data row33 col10" >2.59</td>
      <td id="T_69029_row33_col11" class="data row33 col11" >2.59</td>
      <td id="T_69029_row33_col12" class="data row33 col12" >2.83</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row34" class="row_heading level0 row34" >v_Mo_Sold</th>
      <td id="T_69029_row34_col0" class="data row34 col0" >1,941</td>
      <td id="T_69029_row34_col1" class="data row34 col1" >0.00</td>
      <td id="T_69029_row34_col2" class="data row34 col2" >1.00</td>
      <td id="T_69029_row34_col3" class="data row34 col3" >-1.98</td>
      <td id="T_69029_row34_col4" class="data row34 col4" >-1.98</td>
      <td id="T_69029_row34_col5" class="data row34 col5" >-1.61</td>
      <td id="T_69029_row34_col6" class="data row34 col6" >-0.52</td>
      <td id="T_69029_row34_col7" class="data row34 col7" >-0.16</td>
      <td id="T_69029_row34_col8" class="data row34 col8" >0.57</td>
      <td id="T_69029_row34_col9" class="data row34 col9" >1.66</td>
      <td id="T_69029_row34_col10" class="data row34 col10" >2.03</td>
      <td id="T_69029_row34_col11" class="data row34 col11" >2.03</td>
      <td id="T_69029_row34_col12" class="data row34 col12" >2.03</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row35" class="row_heading level0 row35" >v_Year_Remod/Add</th>
      <td id="T_69029_row35_col0" class="data row35 col0" >1,941</td>
      <td id="T_69029_row35_col1" class="data row35 col1" >0.00</td>
      <td id="T_69029_row35_col2" class="data row35 col2" >1.00</td>
      <td id="T_69029_row35_col3" class="data row35 col3" >-1.64</td>
      <td id="T_69029_row35_col4" class="data row35 col4" >-1.64</td>
      <td id="T_69029_row35_col5" class="data row35 col5" >-1.64</td>
      <td id="T_69029_row35_col6" class="data row35 col6" >-0.92</td>
      <td id="T_69029_row35_col7" class="data row35 col7" >0.43</td>
      <td id="T_69029_row35_col8" class="data row35 col8" >0.96</td>
      <td id="T_69029_row35_col9" class="data row35 col9" >1.10</td>
      <td id="T_69029_row35_col10" class="data row35 col10" >1.15</td>
      <td id="T_69029_row35_col11" class="data row35 col11" >1.20</td>
      <td id="T_69029_row35_col12" class="data row35 col12" >1.64</td>
    </tr>
    <tr>
      <th id="T_69029_level0_row36" class="row_heading level0 row36" >v_Yr_Sold</th>
      <td id="T_69029_row36_col0" class="data row36 col0" >1,941</td>
      <td id="T_69029_row36_col1" class="data row36 col1" >-0.00</td>
      <td id="T_69029_row36_col2" class="data row36 col2" >1.00</td>
      <td id="T_69029_row36_col3" class="data row36 col3" >-1.25</td>
      <td id="T_69029_row36_col4" class="data row36 col4" >-1.25</td>
      <td id="T_69029_row36_col5" class="data row36 col5" >-1.25</td>
      <td id="T_69029_row36_col6" class="data row36 col6" >-1.25</td>
      <td id="T_69029_row36_col7" class="data row36 col7" >0.00</td>
      <td id="T_69029_row36_col8" class="data row36 col8" >1.25</td>
      <td id="T_69029_row36_col9" class="data row36 col9" >1.25</td>
      <td id="T_69029_row36_col10" class="data row36 col10" >1.25</td>
      <td id="T_69029_row36_col11" class="data row36 col11" >1.25</td>
      <td id="T_69029_row36_col12" class="data row36 col12" >1.25</td>
    </tr>
  </tbody>
</table>




```python
(
    ( # these lines do the calculation - what % of missing values are there for each var
        housing_train.isna()      # ccm.isna() TURNS every obs/variable = 1 when its missing and 0 else
       .sum(axis=0)     # count the number of na for each variable (now data is 1 obs per column = # missing)
        /len(housing_train)       # convert # missing to % missing 
        *100            # report as percentage
    ) 
    # you can stop here and report this...
    # but I wanted to format it a bit...
    .sort_values(ascending=False)[:13]
    .to_frame(name='% missing') # the next line only works on a frame, and because pandas sees only 1 variable at this pt
    .style.format("{:.1f}")     # in the code, it calls this a "series" type object, so convert it to dataframe type object
)
#
```


<table id="T_d919a_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >% missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d919a_level0_row0" class="row_heading level0 row0" >v_Pool_QC</th>
      <td id="T_d919a_row0_col0" class="data row0 col0" >99.3</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row1" class="row_heading level0 row1" >v_Misc_Feature</th>
      <td id="T_d919a_row1_col0" class="data row1 col0" >96.8</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row2" class="row_heading level0 row2" >v_Alley</th>
      <td id="T_d919a_row2_col0" class="data row2 col0" >93.0</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row3" class="row_heading level0 row3" >v_Fence</th>
      <td id="T_d919a_row3_col0" class="data row3 col0" >81.2</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row4" class="row_heading level0 row4" >v_Fireplace_Qu</th>
      <td id="T_d919a_row4_col0" class="data row4 col0" >48.4</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row5" class="row_heading level0 row5" >v_Lot_Frontage</th>
      <td id="T_d919a_row5_col0" class="data row5 col0" >16.5</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row6" class="row_heading level0 row6" >v_Garage_Cond</th>
      <td id="T_d919a_row6_col0" class="data row6 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row7" class="row_heading level0 row7" >v_Garage_Finish</th>
      <td id="T_d919a_row7_col0" class="data row7 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row8" class="row_heading level0 row8" >v_Garage_Yr_Blt</th>
      <td id="T_d919a_row8_col0" class="data row8 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row9" class="row_heading level0 row9" >v_Garage_Qual</th>
      <td id="T_d919a_row9_col0" class="data row9 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row10" class="row_heading level0 row10" >v_Garage_Type</th>
      <td id="T_d919a_row10_col0" class="data row10 col0" >5.4</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row11" class="row_heading level0 row11" >v_Bsmt_Exposure</th>
      <td id="T_d919a_row11_col0" class="data row11 col0" >2.7</td>
    </tr>
    <tr>
      <th id="T_d919a_level0_row12" class="row_heading level0 row12" >v_Bsmt_Qual</th>
      <td id="T_d919a_row12_col0" class="data row12 col0" >2.6</td>
    </tr>
  </tbody>
</table>




#### Data Summary

(1)This dataset has 1941 observations and 81 variables.

(2)Time spans

- Time spans of 'v_Yr_Sold'   are three years that are from 2006 to 2008
- Time spans of 'v_Year_Built' are 137 years that are from 1872 to 2008
- Time spans of 'v_Year_Remod/Add' are 60 years that are from 1950 to 2009
- Time spans of 'v_Garage_Yr_Blt' are 114 years that are from 1895 to 2008. There is a value 2207 in this column that seems a much larger future year. I conclude 114 years without 2207. 

(3)Outliers and missing values

Besides object type of data, there are 25 kinds of numerical data which have outliers among 37 kinds of numerical data.
Among the whole data set, there are 27 kinds of data which have missing values. Among the 27, 6 kinds of data have over 10% missing values in their columns.

(4)Data type

These data type is object and they are also categorical data. For better understanding, they can be divided into nominal or ordinal type. The categorical ordering is meaningful.
Nominal:
'parcel', 'v_MS_Zoning', 'v_Street', 'v_Alley', 'v_Land_Contour', 'v_Lot_Config', 'v_Neighborhood', 'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type','v_House_Style', 'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd', 'v_Mas_Vnr_Type',       'v_Foundation', 'v_Heating', 'v_Central_Air', 'v_Functional','v_Garage_Type', 'v_Misc_Feature', 'v_Sale_Type', 'v_Sale_Condition'
Ordinal:
'v_Lot_Shape'(ordinal),'v_Utilities'(ordinal), 'v_Land_Slope'(ordinal),'v_Exter_Qual'(ordinal), 'v_Exter_Cond'(ordinal),'v_Bsmt_Qual'(ordinal),'v_Bsmt_Cond'(ordinal),'v_Bsmt_Exposure'(ordinal),
'v_BsmtFin_Type_1'(ordinal), 'v_BsmtFin_Type_2'(ordinal), 'v_Heating_QC'(ordinal),'v_Electrical'(ordinal), 'v_Kitchen_Qual'(ordinal),'v_Fireplace_Qu'(ordinal),'v_Garage_Finish'(ordinal),'v_Garage_Qual'(ordinal),'v_Garage_Cond'(ordinal),'v_Paved_Drive'(ordinal), 'v_Pool_QC'(ordinal), 'v_Fence'(ordinal),

These data type is int and float and they are also numerical data. For better understanding, they can be divided into discrete or continuous type.

Numerical: 
Nominal：'v_MS_SubClass'
Continuous:'v_Lot_Frontage', 'v_Lot_Area',
 'v_Overall_Qual'(ordinal), 'v_Overall_Cond'(ordinal), 'v_Mas_Vnr_Area','v_BsmtFin_SF_1', 'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF','v_1st_Flr_SF', 'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area','v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 
'v_Garage_Area', 'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch', 'v_Screen_Porch','v_Pool_Area', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_SalePrice'

Discrete:'v_Year_Built', 'v_Year_Remod/Add', 'v_Full_Bath', 'v_Half_Bath','v_Bedroom_AbvGr',
'v_Kitchen_AbvGr', 'v_TotRms_AbvGrd', 'v_Fireplaces','v_Garage_Yr_Blt', 'v_Garage_Cars','v_Mo_Sold', 'v_Yr_Sold'

(5)Visualization and notes

In the relations between sale price and continuous variables, their scatter plots shows that 
Some of them has linear relations, some of them does not have obvious relations with sales price, and the discrete variables show the same type of relations as the categorical variables. For example, Total square feet of basement area and sale price has positive linear relations.

According to the visualization of categorical variables, the house with following characters may have higher mean price. For example,'floating village residential', 'pave street',' hillside',' CulDSac lot configuration',' adjacent to positive off-site feature',' within 200' of North-South railroad',' townhouse end unit type of dwelling' ,'two story or two',' one-half story: 2nd level finished',' flat roof style' ,' wood shingles of roof materials',' stone exterior cover',' poured concrete or wood foundations' ,'typical functionality',' new',' GasA(Gas forced warm air furnace )',' GasW(Gas hot water or steam heat)' ,' central air conditioning',' Built-In (Garage part of house - typically has room above garage)' ,' partial home was not completed when last assessed (associated with New Homes)',' moderately irregular(general shape of property)','   severe Slope',' excellent (evaluates the quality of the material on the exterior )',' excellent(evaluates the present condition of the material on the exterior)',' evaluates the height of the basement excellent (100+ inches)','  good evaluates (the general condition of the basement)',' walkout or garden level walls Gd Good Exposure',' rating of basement finished area GLQGood living quarters',' excellent heating quality and condition', 'excellent electrical system','standard circuit breakers & romex','excellent kitchen','excellent fireplace quality','excellent - exceptional masonry fireplace','interior finish of the garage',' excellent garage quality',' good garage condition',' paved driveway' ,' pool quality(excellent)','fence quality good privacy',' 2-STORY 1946 & NEWER Identifies the type of dwelling involved in the sale'.

To my surprise, house with severe slope may have higher mean price and people may not care too much about the basement quality when they pick the houses. The mean price of house with good basement quality is even higher than house with excellent conditions.

(6)Data issues before regression
   - some columns have large numbers of missing values ,such as 'v_Pool_QC', which cannot have enough data to explain sale price.
   - some columns have no relations with the dependent variable, such as 'v_BsmtFin_SF_2 (Continuous): Type 2 finished square feet', which cannot obviously influence the sale price.
   - Be careful about some variables that share similar slope but they do not have reasonable relations in real. These kinds of variable will misguide researcher to conclude wrong correlation or reasons of another variable.
 








```python
# categorical visualization code
#ax = sns.boxplot(x="v_MS_Zoning", y="v_SalePrice", data=housing_train)
#ax1= sns.boxplot(x="v_Street", y="v_SalePrice", data=housing_train)
#ax2= sns.boxplot(x="v_Alley", y="v_SalePrice", data=housing_train)
#ax3= sns.boxplot(x="v_Land_Contour", y="v_SalePrice", data=housing_train)
#ax4= sns.boxplot(x="v_Lot_Config", y="v_SalePrice", data=housing_train)
#ax5= sns.boxplot(x="v_Neighborhood", y="v_SalePrice", data=housing_train)
#ax6= sns.boxplot(x="v_Condition_1", y="v_SalePrice", data=housing_train)
#ax7= sns.boxplot(x="v_Condition_2", y="v_SalePrice", data=housing_train)
#ax8= sns.boxplot(x="v_Bldg_Type", y="v_SalePrice", data=housing_train)
#ax9= sns.boxplot(x="v_House_Style", y="v_SalePrice", data=housing_train)
#ax10= sns.boxplot(x="v_Roof_Style", y="v_SalePrice", data=housing_train)
#ax11= sns.boxplot(x="v_Roof_Matl", y="v_SalePrice", data=housing_train)
#ax12= sns.boxplot(x="v_Exterior_1st", y="v_SalePrice", data=housing_train)
#ax13= sns.boxplot(x="v_Exterior_2nd", y="v_SalePrice", data=housing_train)
#ax14= sns.boxplot(x="v_Mas_Vnr_Type", y="v_SalePrice", data=housing_train)
#ax15= sns.boxplot(x="v_Foundation", y="v_SalePrice", data=housing_train)
#ax16= sns.boxplot(x="v_Heating", y="v_SalePrice", data=housing_train)
#ax17= sns.boxplot(x="v_Central_Air", y="v_SalePrice", data=housing_train)
#ax18= sns.boxplot(x="v_Functional", y="v_SalePrice", data=housing_train)
#ax19= sns.boxplot(x="v_Garage_Type", y="v_SalePrice", data=housing_train)
#ax20= sns.boxplot(x="v_Misc_Feature", y="v_SalePrice", data=housing_train)
#ax21= sns.boxplot(x="v_Sale_Type", y="v_SalePrice", data=housing_train)
#ax22= sns.boxplot(x="v_Sale_Condition", y="v_SalePrice", data=housing_train)
```


```python
#ordinal visualization code
#ax23= sns.boxplot(x="v_Lot_Shape", y="v_SalePrice", data=housing_train)
#ax24= sns.boxplot(x="v_Utilities", y="v_SalePrice", data=housing_train)
#ax25= sns.boxplot(x="v_Land_Slope", y="v_SalePrice", data=housing_train).set(title='sale price & land slope')
#ax26= sns.boxplot(x="v_Exter_Qual", y="v_SalePrice", data=housing_train)
#ax27= sns.boxplot(x="v_Exter_Cond", y="v_SalePrice", data=housing_train)
#ax28= sns.boxplot(x="v_Bsmt_Qual", y="v_SalePrice", data=housing_train)
#ax29= sns.boxplot(x="v_Bsmt_Cond", y="v_SalePrice", data=housing_train).set(title='sale price & basement condition')
#ax30= sns.boxplot(x="v_Bsmt_Exposure", y="v_SalePrice", data=housing_train)
#ax31= sns.boxplot(x="v_BsmtFin_Type_1", y="v_SalePrice", data=housing_train)
#ax32= sns.boxplot(x="v_BsmtFin_Type_2", y="v_SalePrice", data=housing_train)
#ax33= sns.boxplot(x="v_Heating_QC", y="v_SalePrice", data=housing_train)
#ax34= sns.boxplot(x="v_Electrical", y="v_SalePrice", data=housing_train)
#ax35= sns.boxplot(x="v_Kitchen_Qual", y="v_SalePrice", data=housing_train)
#ax36= sns.boxplot(x="v_Fireplace_Qu", y="v_SalePrice", data=housing_train)
#ax37= sns.boxplot(x="v_Garage_Finish", y="v_SalePrice", data=housing_train)
#ax38= sns.boxplot(x="v_Garage_Qual", y="v_SalePrice", data=housing_train)
#ax39= sns.boxplot(x="v_Garage_Cond", y="v_SalePrice", data=housing_train)
#ax40= sns.boxplot(x="v_Paved_Drive", y="v_SalePrice", data=housing_train)
#ax41= sns.boxplot(x="v_Pool_QC", y="v_SalePrice", data=housing_train)
#ax42= sns.boxplot(x="v_Fence", y="v_SalePrice", data=housing_train)
```


```python
#continuous variable visualization code
#cs1=sns.boxplot(data=housing_train, x="v_MS_SubClass", y="v_SalePrice")
#cs2=sns.scatterplot(data=housing_train, x="v_Lot_Frontage", y="v_SalePrice")
#cs3=sns.scatterplot(data=housing_train, x="v_Lot_Area", y="v_SalePrice")
#cs4=sns.boxplot(data=housing_train, x="v_Overall_Qual", y="v_SalePrice")
#cs5=sns.boxplot(data=housing_train, x="v_Overall_Cond", y="v_SalePrice")
#cs6=sns.scatterplot(data=housing_train, x="v_Mas_Vnr_Area", y="v_SalePrice")
#cs7=sns.scatterplot(data=housing_train, x="v_BsmtFin_SF_1", y="v_SalePrice")
#cs8=sns.scatterplot(data=housing_train, x="v_BsmtFin_SF_2", y="v_SalePrice")
#cs9=sns.scatterplot(data=housing_train, x="v_Bsmt_Unf_SF", y="v_SalePrice")
#cs10=sns.scatterplot(data=housing_train, x="v_Total_Bsmt_SF", y="v_SalePrice")
#cs11=sns.scatterplot(data=housing_train, x="v_1st_Flr_SF", y="v_SalePrice")
#cs12=sns.scatterplot(data=housing_train, x="v_2nd_Flr_SF", y="v_SalePrice")
#cs13=sns.scatterplot(data=housing_train, x="v_Low_Qual_Fin_SF", y="v_SalePrice")
#cs14=sns.scatterplot(data=housing_train, x="v_Gr_Liv_Area", y="v_SalePrice")
#cs15=sns.scatterplot(data=housing_train, x="v_Bsmt_Full_Bath", y="v_SalePrice")
#cs16=sns.scatterplot(data=housing_train, x="v_Bsmt_Half_Bath", y="v_SalePrice")
#cs17=sns.scatterplot(data=housing_train, x="v_Garage_Area", y="v_SalePrice")
#cs18=sns.scatterplot(data=housing_train, x="v_Wood_Deck_SF", y="v_SalePrice")
#cs19=sns.scatterplot(data=housing_train, x="v_Open_Porch_SF", y="v_SalePrice")
#cs20=sns.scatterplot(data=housing_train, x="v_Enclosed_Porch", y="v_SalePrice")
#cs21=sns.scatterplot(data=housing_train, x="v_3Ssn_Porch", y="v_SalePrice")
#cs22=sns.scatterplot(data=housing_train, x="v_Screen_Porch", y="v_SalePrice")
#cs23=sns.scatterplot(data=housing_train, x="v_Pool_Area", y="v_SalePrice")
#cs24=sns.regplot(data=housing_train, x='v_Misc_Val', y='v_SalePrice')
#ax = sns.regplot(x="v_Misc_Val", y="v_SalePrice", data=housing_train)
# discrete variable visualization code
#cs25=sns.barplot(data=housing_train, x="v_Mo_Sold", y="v_SalePrice")
#cs26=sns.barplot(data=housing_train, x="v_Yr_Sold", y="v_SalePrice")
#cs27=sns.barplot(data=housing_train, x="v_Garage_Cars", y="v_SalePrice")
#cs28=sns.barplot(data=housing_train, x="v_Garage_Yr_Blt", y="v_SalePrice")
#cs29=sns.barplot(data=housing_train, x="v_Fireplaces", y="v_SalePrice")
#cs30=sns.barplot(data=housing_train, x="v_TotRms_AbvGrd", y="v_SalePrice")
#cs31=sns.barplot(data=housing_train, x="v_Kitchen_AbvGr", y="v_SalePrice")
#cs32=sns.barplot(data=housing_train, x="v_Bedroom_AbvGr", y="v_SalePrice")
#cs33=sns.barplot(data=housing_train, x="v_Half_Bath", y="v_SalePrice")
#cs34=sns.barplot(data=housing_train, x="v_Full_Bath", y="v_SalePrice")
#cs35=sns.barplot(data=housing_train, x="v_Year_Remod/Add", y="v_SalePrice")
#cs36=sns.barplot(data=housing_train, x="v_Year_Built", y="v_SalePrice")
```

<img width="591" alt="image" src="https://user-images.githubusercontent.com/98285249/161427152-dad26a32-bc5e-40af-8393-d3fccde6a80d.png">

<img width="591" alt="image" src="https://user-images.githubusercontent.com/98285249/161427346-acbabc82-18e6-4388-824b-856a019c4806.png">

## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that i is indexing a given house, and t indexes the year of sale._ 

<img width="782" alt="image" src="https://user-images.githubusercontent.com/98285249/169089282-0792e363-f1c0-4fcc-a509-09979fe56d43.png">

    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
housing_train = (housing_train
                  # create variables
                  .assign(l_vLA = np.log(housing_train['v_Lot_Area']),
                          l_SP = np.log(housing_train['v_SalePrice']),
                         )
                  .rename(columns={'v_SalePrice':'Price'})
                 )
```


```python
reg1 = sm_ols('Price ~  v_Lot_Area', data=housing_train).fit()

reg2= sm_ols('Price ~  l_vLA  ',  data=housing_train).fit()

reg3= sm_ols('l_SP ~ v_Lot_Area',  data=housing_train).fit()

reg4= sm_ols('l_SP ~  l_vLA  ',  data=housing_train).fit()

reg5 = sm_ols('l_SP ~ v_Yr_Sold',  data=housing_train).fit()

reg6 = sm_ols('l_SP ~ C(v_Yr_Sold)',  data=housing_train).fit()

reg7 = sm_ols('l_SP ~ C(v_Neighborhood)+v_Overall_Qual+v_Gr_Liv_Area+v_Bldg_Type+v_Roof_Matl',
       data=housing_train).fit()
```


```python
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y =  sale price if not specified, log(sale price else)')
print(summary_col(results=[reg1,reg2,reg3,reg4,reg5,reg6,reg7], # list the result obj here
                  float_format='%0.5f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 (log)','4 (log)','5 (log)','6(log)','7(log)'], 
                  info_dict=info_dict,
                  regressor_order=[ 'Intercept','v_Lot_Area','l_vLA','v_Yr_Sold',
                                  'C(v_Yr_Sold)[2007]','C(v_Yr_Sold)[2008]']
                  )
     )
```

    ============================================================================================================
                      y =  sale price if not specified, log(sale price else)
    
    =======================================================================================================================
                                        1               2            3 (log)   4 (log)    5 (log)      6(log)      7(log)  
    -----------------------------------------------------------------------------------------------------------------------
    Intercept                    154789.55021*** -327915.80232*** 11.89407*** 9.40505*** 22.29321   12.02287*** 9.45152*** 
                                 (2911.59058)    (30221.34714)    (0.01463)   (0.15108)  (22.93682) (0.01614)   (0.17582)  
    v_Lot_Area                   2.64894***                       0.00001***                                               
                                 (0.22525)                        (0.00000)                                                
    l_vLA                                        56028.16996***               0.28826***                                   
                                                 (3315.13919)                 (0.01657)                                    
    v_Yr_Sold                                                                            -0.00511                          
                                                                                         (0.01143)                         
    C(v_Neighborhood)[T.Blueste]                                                                                -0.08008   
                                                                                                                (0.09026)  
    v_Bldg_Type[T.Twnhs]                                                                                        -0.18521***
                                                                                                                (0.02780)  
    C(v_Neighborhood)[T.Timber]                                                                                 0.04482    
                                                                                                                (0.04454)  
    C(v_Neighborhood)[T.Veenker]                                                                                0.02890    
                                                                                                                (0.05183)  
    C(v_Yr_Sold)[T.2007]                                                                            0.02559                
                                                                                                    (0.02225)              
    C(v_Yr_Sold)[T.2008]                                                                            -0.01028               
                                                                                                    (0.02285)              
    v_Bldg_Type[T.2fmCon]                                                                                       -0.00841   
                                                                                                                (0.02475)  
    v_Bldg_Type[T.Duplex]                                                                                       -0.12928***
                                                                                                                (0.02142)  
    v_Bldg_Type[T.TwnhsE]                                                                                       -0.05162***
                                                                                                                (0.01660)  
    v_Gr_Liv_Area                                                                                               0.00028*** 
                                                                                                                (0.00001)  
    C(v_Neighborhood)[T.Somerst]                                                                                -0.00465   
                                                                                                                (0.03990)  
    v_Overall_Qual                                                                                              0.12006*** 
                                                                                                                (0.00444)  
    v_Roof_Matl[T.CompShg]                                                                                      1.55843*** 
                                                                                                                (0.16556)  
    v_Roof_Matl[T.Membran]                                                                                      1.74420*** 
                                                                                                                (0.23333)  
    v_Roof_Matl[T.Metal]                                                                                        1.67432*** 
                                                                                                                (0.23389)  
    v_Roof_Matl[T.Roll]                                                                                         1.50886*** 
                                                                                                                (0.23130)  
    v_Roof_Matl[T.Tar&Grv]                                                                                      1.67766*** 
                                                                                                                (0.17017)  
    v_Roof_Matl[T.WdShake]                                                                                      1.44690*** 
                                                                                                                (0.17579)  
    v_Roof_Matl[T.WdShngl]                                                                                      1.72329*** 
                                                                                                                (0.17746)  
    C(v_Neighborhood)[T.BrDale]                                                                                 -0.31137***
                                                                                                                (0.05436)  
    C(v_Neighborhood)[T.StoneBr]                                                                                0.08214*   
                                                                                                                (0.04549)  
    C(v_Neighborhood)[T.Sawyer]                                                                                 -0.11868***
                                                                                                                (0.04231)  
    C(v_Neighborhood)[T.SawyerW]                                                                                -0.08899** 
                                                                                                                (0.04290)  
    C(v_Neighborhood)[T.BrkSide]                                                                                -0.29030***
                                                                                                                (0.04351)  
    C(v_Neighborhood)[T.ClearCr]                                                                                -0.01944   
                                                                                                                (0.05004)  
    C(v_Neighborhood)[T.CollgCr]                                                                                -0.04718   
                                                                                                                (0.04004)  
    C(v_Neighborhood)[T.Crawfor]                                                                                -0.04568   
                                                                                                                (0.04234)  
    C(v_Neighborhood)[T.Edwards]                                                                                -0.23353***
                                                                                                                (0.04142)  
    C(v_Neighborhood)[T.Gilbert]                                                                                -0.09866** 
                                                                                                                (0.04150)  
    C(v_Neighborhood)[T.Greens]                                                                                 0.00537    
                                                                                                                (0.08205)  
    C(v_Neighborhood)[T.GrnHill]                                                                                0.33982*** 
                                                                                                                (0.11915)  
    C(v_Neighborhood)[T.IDOTRR]                                                                                 -0.41340***
                                                                                                                (0.04386)  
    C(v_Neighborhood)[T.Landmrk]                                                                                -0.08337   
                                                                                                                (0.16681)  
    C(v_Neighborhood)[T.MeadowV]                                                                                -0.24402***
                                                                                                                (0.05150)  
    C(v_Neighborhood)[T.Mitchel]                                                                                -0.05766   
                                                                                                                (0.04315)  
    C(v_Neighborhood)[T.NAmes]                                                                                  -0.13603***
                                                                                                                (0.04016)  
    C(v_Neighborhood)[T.NPkVill]                                                                                -0.12200*  
                                                                                                                (0.06560)  
    C(v_Neighborhood)[T.NWAmes]                                                                                 -0.09981** 
                                                                                                                (0.04239)  
    C(v_Neighborhood)[T.NoRidge]                                                                                0.01920    
                                                                                                                (0.04546)  
    C(v_Neighborhood)[T.NridgHt]                                                                                0.11413*** 
                                                                                                                (0.04058)  
    C(v_Neighborhood)[T.OldTown]                                                                                -0.33278***
                                                                                                                (0.04128)  
    C(v_Neighborhood)[T.SWISU]                                                                                  -0.30032***
                                                                                                                (0.04995)  
    R-squared                    0.06658         0.12840          0.06459     0.13497    0.00010    0.00144     0.84480    
    R-squared Adj.               0.06610         0.12795          0.06411     0.13453    -0.00041   0.00041     0.84153    
    R-squared                    0.07            0.13             0.06        0.13       0.00       0.00        0.84       
    Adj R-squared                0.07            0.13             0.06        0.13       -0.00      0.00        0.84       
    No. observations             1941            1941             1941        1941       1941       1941        1941       
    =======================================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that i is indexing a given house, and t indexes the year of sale._ 

<img width="756" alt="image" src="https://user-images.githubusercontent.com/98285249/169089552-50960cc1-caa3-450d-9c29-5c66c0d34700.png">



The increase is based on the mean lot size (180284.77 square feet) and the mean sale price (182033.23 dollars)

- Q2-model 2

A 1% increase in lot size is associated with a 560.28 dollars increase in price, holding all other X constant.
if lot size goes from mean to 182087 square feet, the price will increase from mean to 182593.51 dollars.

- Q3-model 3

A 1 square feet increase in lot size is associated with a 0.0013% increase in price, holding all other X constant.
if lot size goes from mean to 182087 square feet( 1802 units increase), the price will increase from mean to 186347.85 dollars.

- Q4-model 4

I think model 4 best explain the data because the value of adjusted R-squared is the largest.

- Q5-model 5

A 1 year increase is associated with a 0.51% decrease in price, holding all other X constant.

- Q6-model 6

$\alpha$ means that the average log price of 2006 is 12.02 dollars

- Q7-model 6

The average log price for 2007 is 0.025 dollars higher than log price of 2006.

- Q8-model 6

R2 of model 6 is higher than that of the model 5 because model 6 can better explain the linear regression relations of log price and sale year. Model 6 is an multiple regression but model 5 is a simple linear regression

- Q9-model 7

These five variables, including 'v_Neighborhood','v_Overall_Qual','v_Gr_Liv_Area','v_Bldg_Type','v_Roof_Matl', are involved in the model 7

- Q10-model 7

The R2 of model 7 is 0.84


```python

```
