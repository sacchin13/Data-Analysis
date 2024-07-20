#!/usr/bin/env python
# coding: utf-8

# # Data import and Basic Exploration
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


app = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/DATA/application_data.csv")
prev_app = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/DATA/previous_application.csv")


# In[3]:


app.head()


# # Feature Selection 

# In[5]:


app.columns


# In[6]:


app.shape


# In[9]:


msng_info = pd.DataFrame(app.isnull().sum().sort_values()).reset_index()
msng_info.rename(columns={'index':'col_name',0:'null_count'},inplace=True)
msng_info.head()


# In[14]:


msng_info['msng_pct'] = msng_info['null_count']/app.shape[0]*100
msng_info.to_excel("C:/Users/Lenovo/OneDrive/Desktop/DATA/missing_info.xlsx",index=False)
msng_info.head()


# In[15]:


msng_col = msng_info[msng_info['msng_pct']>=40]['col_name'].to_list()
app_msng_rmvd = app.drop(labels=msng_col,axis=1)
app_msng_rmvd.shape


# In[16]:


app_msng_rmvd.head()


# In[17]:


flag_col = []

for col in app_msng_rmvd.columns:
    if col.startswith("FLAG_"):
        flag_col.append(col)


# In[18]:


len(flag_col)


# In[19]:


flag_tgt_col = app_msng_rmvd[flag_col+['TARGET']]
flag_tgt_col.head()


# In[21]:


plt.figure(figsize=(20,25))

for i, col in enumerate(flag_col):
    plt.subplot(7,4,i+1)
    sns.countplot(data=flag_tgt_col,x=col,hue='TARGET')


# In[27]:


# List of columns to include in the correlation matrix
flg_corr = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
            'FLAG_PHONE', 'FLAG_EMAIL', 'TARGET']

# Subset the DataFrame to include only the relevant columns
flag_corr_df = app_msng_rmvd[flg_corr].copy()

# Convert binary categorical columns to numeric
binary_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
for col in binary_columns:
    flag_corr_df.loc[:, col] = flag_corr_df[col].map({'Y': 1, 'N': 0})

# Calculate the correlation matrix and round to 2 decimal places
corr_df = round(flag_corr_df.corr(), 2)

# Plot the heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(corr_df, cmap='coolwarm', linewidths=.5, annot=True)
plt.show()


# In[28]:


app_flag_rmvd = app_msng_rmvd.drop(labels =flag_col,axis=1)
app_flag_rmvd.shape


# In[30]:


app_flag_rmvd.head()


# In[31]:


sns.heatmap(data=round(app_flag_rmvd[['EXT_SOURCE_2','EXT_SOURCE_3','TARGET']].corr(),2),cmap='coolwarm',linewidths=.5,annot=True)


# In[32]:


app_score_col_rmvd = app_flag_rmvd.drop(['EXT_SOURCE_2','EXT_SOURCE_3'],axis=1)
app_score_col_rmvd.shape


# # Feature Engineering

# In[33]:


app_score_col_rmvd.isnull().sum().sort_values()/app_score_col_rmvd.shape[0]


# # Missing Imputation

# In[38]:


app_score_col_rmvd['CNT_FAM_MEMBERS'] = app_score_col_rmvd['CNT_FAM_MEMBERS'].fillna((app_score_col_rmvd['CNT_FAM_MEMBERS'].mode()[0]))
app_score_col_rmvd['CNT_FAM_MEMBERS'].isnull().sum()


# In[42]:


app_score_col_rmvd['OCCUPATION_TYPE'] = app_score_col_rmvd['OCCUPATION_TYPE'].fillna((app_score_col_rmvd['OCCUPATION_TYPE'].mode()[0]))
app_score_col_rmvd['OCCUPATION_TYPE'].isnull().sum()


# In[43]:


app_score_col_rmvd['NAME_TYPE_SUITE'] = app_score_col_rmvd['NAME_TYPE_SUITE'].fillna((app_score_col_rmvd['NAME_TYPE_SUITE'].mode()[0]))
app_score_col_rmvd['NAME_TYPE_SUITE'].isnull().sum()


# In[44]:


app_score_col_rmvd['AMT_ANNUITY'] = app_score_col_rmvd['AMT_ANNUITY'].fillna((app_score_col_rmvd['AMT_ANNUITY'].mean()))
app_score_col_rmvd['AMT_ANNUITY'].isnull().sum()


# In[47]:


amt_req_col = []

for col in app_score_col_rmvd.columns:
    if col.startswith("AMT_REQ_CREDIT_BUREAU"):
        amt_req_col.append(col)
amt_req_col


# In[48]:


for col in amt_req_col:
    app_score_col_rmvd[col] = app_score_col_rmvd[col].fillna((app_score_col_rmvd[col].median()))


# In[56]:


app_score_col_rmvd.isnull().sum().sort_values()


# In[53]:


app_score_col_rmvd['AMT_GOODS_PRICE'] = app_score_col_rmvd['AMT_GOODS_PRICE'].fillna((app_score_col_rmvd['AMT_GOODS_PRICE'].median()))


# In[54]:


app_score_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# # Value Modifiction

# In[55]:


days_col = []
for col in app_score_col_rmvd.columns:
    if col.startswith("DAYS"):
        days_col.append(col)
days_col


# In[57]:


for col in days_col:
    app_score_col_rmvd[col] = abs(app_score_col_rmvd[col])


# In[58]:


app_score_col_rmvd.head()


# In[60]:


app_score_col_rmvd.nunique().sort_values()


# In[61]:


app_score_col_rmvd['OBS_30_CNT_SOCIAL_CIRCLE'].unique()


# # Outlier detection & treatment

# In[62]:


app_score_col_rmvd['AMT_GOODS_PRICE'].agg(['min','max','median'])


# In[64]:


sns.kdeplot(data=app_score_col_rmvd,x='AMT_GOODS_PRICE')


# In[65]:


sns.boxenplot(data=app_score_col_rmvd,x='AMT_GOODS_PRICE')


# In[66]:


app_score_col_rmvd['AMT_GOODS_PRICE'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[67]:


bins = [0,100000,200000,300000,400000,500000,600000,700000,800000,900000,4050000]
ranges = ['0-100K','100k-200K','200K-300K','300K-400K','400K-500K','500K-600K','600K-700K'
          ,'700K-800K','800K-900K','Above 900K']

app_score_col_rmvd['AMT_GOODS_PRICE_RANGE'] = pd.cut(app_score_col_rmvd['AMT_GOODS_PRICE'],bins,labels=ranges)


# In[99]:


app_score_col_rmvd.groupby(['AMT_GOODS_PRICE_RANGE']).size()


# In[98]:


app_score_col_rmvd['AMT_INCOME_TOTAL'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[70]:


app_score_col_rmvd['AMT_INCOME_TOTAL'].max()


# In[71]:


bins = [0,100000,150000,200000,250000,300000,350000,400000,117000000]
ranges = ['0-100K','100K-150K','150K-200K','200K-250K','250K-300K','300K-350K','350K-400K'
          ,'Above 400K']

app_score_col_rmvd['AMT_INCOME_TOTAL_RANGE'] = pd.cut(app_score_col_rmvd['AMT_INCOME_TOTAL'],bins,labels=ranges)


# In[72]:


app_score_col_rmvd.groupby(['AMT_INCOME_TOTAL_RANGE']).size()


# In[73]:


app_score_col_rmvd['AMT_INCOME_TOTAL_RANGE'].isnull().sum()


# In[74]:


app_score_col_rmvd['AMT_CREDIT'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[75]:


app_score_col_rmvd['AMT_CREDIT'].max()


# In[76]:


bins = [0,200000,400000,600000,800000,900000,1000000,2000000,3000000,4050000]
ranges = ['0-200K','200K-400K','400K-600K','600K-800K','800K-900K','900K-1M','1M-2M','2M-3M','Above 3M']

app_score_col_rmvd['AMT_CREDIT_RANGE'] = pd.cut(app_score_col_rmvd['AMT_CREDIT'],bins,labels=ranges)


# In[77]:


app_score_col_rmvd.groupby(['AMT_CREDIT_RANGE']).size()


# In[78]:


app_score_col_rmvd['AMT_CREDIT'].isnull().sum()


# In[79]:


app_score_col_rmvd['AMT_ANNUITY'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[80]:


app_score_col_rmvd['AMT_ANNUITY'].max()


# In[81]:


bins = [0,25000,50000,100000,150000,200000,258025.5]
ranges = ['0-25K','25K-50K','50K-100K','100K-150K','150K-200K','Above 200K']

app_score_col_rmvd['AMT_ANNUITY_RANGE'] = pd.cut(app_score_col_rmvd['AMT_ANNUITY'],bins,labels=ranges)


# In[82]:


app_score_col_rmvd.groupby(['AMT_ANNUITY_RANGE']).size()


# In[83]:


app_score_col_rmvd['AMT_ANNUITY_RANGE'].isnull().sum()


# In[84]:


app_score_col_rmvd['DAYS_EMPLOYED'].agg(['min','max','median'])


# In[85]:


app_score_col_rmvd['DAYS_EMPLOYED'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[86]:


app_score_col_rmvd[app_score_col_rmvd['DAYS_EMPLOYED']<app_score_col_rmvd['DAYS_EMPLOYED'].max()].max()['DAYS_EMPLOYED']


# In[87]:


app_score_col_rmvd['DAYS_EMPLOYED'].max()


# In[88]:


bins = [0,1825,3650,5475,7300,9125,10950,12775,14600,16425,18250,23691,365243]

ranges = ['0-5Y','5Y-10Y','10Y-15Y','15Y-20Y','20Y-25Y','25Y-30Y','30Y-35Y','35Y-40Y','40Y-45Y','45Y-50Y'
          ,'50Y-65Y','Above 65Y']

app_score_col_rmvd['DAYS_EMPLOYED_RANGE'] = pd.cut(app_score_col_rmvd['DAYS_EMPLOYED'],bins,labels=ranges)


# In[95]:


grouped_data = app_score_col_rmvd.groupby(['DAYS_EMPLOYED_RANGE'], observed=False).size()
print(grouped_data)


# In[90]:


app_score_col_rmvd['DAYS_BIRTH'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[91]:


app_score_col_rmvd['DAYS_BIRTH'].min()


# In[92]:


bins = [0,7300,10950,14600,18250,21900,25229]
ranges = ['20Y','20Y-30Y','30Y-40Y','40Y-50Y','50Y-60Y','Above 60Y']
app_score_col_rmvd['DAYS_BIRTH_RANGE'] = pd.cut(app_score_col_rmvd['DAYS_BIRTH'],bins,labels=ranges)


# In[96]:


grouped_data = app_score_col_rmvd.groupby(['DAYS_BIRTH_RANGE'], observed=True).size()
print(grouped_data)


# In[94]:


app_score_col_rmvd['DAYS_BIRTH'].isnull().sum()


# # Data Analysis

# In[100]:


app_score_col_rmvd.dtypes.value_counts()


# In[101]:


obj_var = app_score_col_rmvd.select_dtypes(include=['object']).columns
obj_var


# In[102]:


app_score_col_rmvd.groupby(['NAME_CONTRACT_TYPE']).size()


# In[103]:


sns.countplot(data=app_score_col_rmvd,x='NAME_CONTRACT_TYPE',hue='TARGET')


# In[105]:


data_pct = app_score_col_rmvd[['NAME_CONTRACT_TYPE','TARGET']].groupby(['NAME_CONTRACT_TYPE'], as_index=False).mean().sort_values(by='TARGET',ascending=False)


# In[107]:


data_pct['PCT'] = data_pct['TARGET']*100


# In[108]:


data_pct


# In[109]:


sns.barplot(data=data_pct,x='NAME_CONTRACT_TYPE',y='PCT')


# In[111]:


obj_var


# In[130]:


plt.figure(figsize=(25, 60))

for i, var in enumerate(obj_var):
    data_pct = app_score_col_rmvd[[var, 'TARGET']].groupby([var], as_index=False).mean().sort_values(by='TARGET', ascending=False)
    data_pct['PCT'] = data_pct['TARGET'] * 100

    plt.subplot(10, 2, i + i + 1)
    plt.subplots_adjust(wspace=0.1, hspace=1)
    sns.countplot(data=app_score_col_rmvd, x=var, hue='TARGET')
    plt.xticks(rotation=90)

    plt.subplot(10, 2, i + i + 2)
    sns.barplot(data=data_pct, x=var, y='PCT', hue='TARGET', palette='coolwarm', legend=False)
    plt.xticks(rotation=90)

plt.show()


# In[131]:


app_score_col_rmvd['NAME_EDUCATION_TYPE'].unique()


# In[132]:


app_score_col_rmvd.dtypes.value_counts()


# In[133]:


num_var = app_score_col_rmvd.select_dtypes(include=['float64','int64']).columns
num_cat_var = app_score_col_rmvd.select_dtypes(include=['float64','int64','category']).columns
len(num_var)


# In[134]:


num_data = app_score_col_rmvd[num_var]
defaulters = num_data[num_data['TARGET']==1]
repayers = num_data[num_data['TARGET']==0]
repayers.head()


# In[135]:


defaulters[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL']].corr()


# In[141]:


import numpy as np
import pandas as pd

# Compute the correlation matrix
defaulter_corr = defaulters.corr()

# Create a boolean mask for the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(defaulter_corr, dtype=bool), k=1)

# Apply the mask to get only the upper triangle values
defaulter_corr_unstack = defaulter_corr.where(mask).stack().reset_index()
defaulter_corr_unstack.columns = ['var1', 'var2', 'corr']

# Compute the absolute values of correlations
defaulter_corr_unstack['corr'] = abs(defaulter_corr_unstack['corr'])

# Drop NaN values and get the top 10 correlations
top_corrs = defaulter_corr_unstack.dropna(subset=['corr']).sort_values(by='corr', ascending=False).head(10)

print(top_corrs)


# In[143]:


import numpy as np
import pandas as pd

# Compute the correlation matrix
repayers_corr = repayers.corr()

# Create a boolean mask for the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(repayers_corr, dtype=bool), k=1)

# Apply the mask to get only the upper triangle values
repayers_corr_unstack = repayers_corr.where(mask).stack().reset_index()
repayers_corr_unstack.columns = ['var1', 'var2', 'corr']

# Compute the absolute values of correlations
repayers_corr_unstack['corr'] = abs(repayers_corr_unstack['corr'])

# Drop NaN values and get the top 10 correlations
top_corrs = repayers_corr_unstack.dropna(subset=['corr']).sort_values(by='corr', ascending=False).head(10)

print(top_corrs)


# In[144]:


num_data.head()


# In[145]:


amt_var = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']


# In[146]:


sns.kdeplot(data=num_data,x='AMT_CREDIT',hue='TARGET')


# In[147]:


plt.figure(figsize=(10,5))

for i, col in enumerate(amt_var):
    plt.subplot(2,2,i+1)
    sns.kdeplot(data=num_data,x=col,hue='TARGET')
    plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[148]:


num_data.head()


# In[159]:


sns.scatterplot(data=num_data,x='AMT_CREDIT',y='CNT_CHILDREN',hue='TARGET')


# In[160]:


amt_var = num_data[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','TARGET']]


# In[161]:


sns.pairplot(data=amt_var,hue='TARGET')


# In[163]:


null_count = pd.DataFrame(prev_app.isnull().sum().sort_values(ascending=False) / prev_app.shape[0] * 100).reset_index().rename(columns={'index': 'var', 0: 'count_pct'})
var_msng_ge_40 = list(null_count[null_count['count_pct'] >= 40]['var'])
var_msng_ge_40


# In[164]:


nva_cols = var_msng_ge_40+['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']
len(nva_cols)


# In[165]:


len(prev_app.columns)


# In[166]:


prev_app_nva_col_rmvd = prev_app.drop(labels=nva_cols,axis=1)
len(prev_app_nva_col_rmvd.columns)


# In[167]:


prev_app_nva_col_rmvd.columns


# In[168]:


prev_app_nva_col_rmvd.head()


# In[169]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)/prev_app_nva_col_rmvd.shape[0]*100


# In[170]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].agg(func=['mean','median'])


# In[172]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MEDIAN'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].median())


# In[173]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MEAN'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].mean())


# In[174]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MODE'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].mode()[0])


# In[175]:


gp_cols = ['AMT_GOODS_PRICE','AMT_GOODS_PRICE_MEDIAN','AMT_GOODS_PRICE_MEAN','AMT_GOODS_PRICE_MODE']


# In[183]:


plt.figure(figsize=(10,5))

for i, col in enumerate(gp_cols):
    plt.subplot(2,2,i+1)
    sns.kdeplot(data=prev_app_nva_col_rmvd,x=col)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[177]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].median())


# In[178]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[179]:


prev_app_nva_col_rmvd['AMT_ANNUITY'].agg(func=['mean','median','max'])


# In[180]:


prev_app_nva_col_rmvd['AMT_ANNUITY'] = prev_app_nva_col_rmvd['AMT_ANNUITY'].fillna(prev_app_nva_col_rmvd['AMT_ANNUITY'].median())


# In[182]:


prev_app_nva_col_rmvd['PRODUCT_COMBINATION'] = prev_app_nva_col_rmvd['PRODUCT_COMBINATION'].fillna(prev_app_nva_col_rmvd['PRODUCT_COMBINATION'].mode()[0])


# In[184]:


prev_app_nva_col_rmvd['CNT_PAYMENT'].agg(func=['mean','median','max'])


# In[185]:


prev_app_nva_col_rmvd[prev_app_nva_col_rmvd['CNT_PAYMENT'].isnull()].groupby(['NAME_CONTRACT_STATUS']).size().sort_values(ascending=False)


# In[186]:


prev_app_nva_col_rmvd['CNT_PAYMENT'] = prev_app_nva_col_rmvd['CNT_PAYMENT'].fillna(0)


# In[187]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)


# In[188]:


prev_app_nva_col_rmvd = prev_app_nva_col_rmvd.drop(labels=['AMT_GOODS_PRICE_MEDIAN','AMT_GOODS_PRICE_MEAN','AMT_GOODS_PRICE_MODE'],axis=1)


# In[191]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)


# In[190]:


len(prev_app_nva_col_rmvd.columns)


# In[192]:


prev_app_nva_col_rmvd.head()


# In[193]:


merged_df = pd.merge(app_score_col_rmvd,prev_app_nva_col_rmvd,how='inner',on='SK_ID_CURR')
merged_df.head()


# In[194]:


plt.figure(figsize=(15,5))

sns.countplot(data=merged_df,x='NAME_CASH_LOAN_PURPOSE',hue='NAME_CONTRACT_STATUS')
plt.xticks(rotation=90)
plt.yscale('log')


# In[195]:


sns.countplot(data=merged_df,x='NAME_CONTRACT_STATUS',hue='TARGET')


# In[196]:


merged_agg = merged_df.groupby(['NAME_CONTRACT_STATUS','TARGET']).size().reset_index().rename(columns={0:'counts'})
sum_df  = merged_agg.groupby(['NAME_CONTRACT_STATUS'])['counts'].sum().reset_index()

merged_agg_2 = pd.merge(merged_agg,sum_df,how='left',on='NAME_CONTRACT_STATUS')
merged_agg_2['pct'] = round(merged_agg_2['counts_x']/merged_agg_2['counts_y']*100,2)
merged_agg_2


# In[198]:


sns.lineplot(data=merged_df, x='NAME_CONTRACT_STATUS', y='AMT_INCOME_TOTAL', errorbar=None, hue='TARGET')


# In[199]:


len(merged_df.columns)


# # Conclusion/Insights

# In[ ]:


#Overview
This analysis focuses on understanding customer behavior and default rates to help the bank target safer segments for loan distribution.
The insights derived from the data encompass various demographic, occupational, and financial factors influencing loan defaults.

#Key Findings
1.	Loan Types
       •	Most customers have taken cash loans.
       •	Customers with cash loans are less likely to default.
2.	Demographic Insights
       •	Gender: Most loans are taken by females, who have a lower default rate of ~7%.
       •	Family Status: Married individuals are safer to target, with an 8% default rate.
       •	Housing: People with houses/apartments have a default rate of ~8%.
       •	Education: Higher education correlates with a lower default rate, less than 5%.
       •	Accompaniment: Unaccompanied individuals have a default rate of ~8.5%.
3.	Income and Occupation
       •	Income: Safest segments include working individuals, commercial associates, and pensioners.
       •	Occupation:
       •	Low-skill laborers and drivers have the highest default rates.
       •	Accountants have lower default rates.
       •	Core staff, managers, and laborers have default rates between 7.5% to 10%.
4.	Organization Type
       •	Transport type 3 organizations have the highest default rate.
       •	Others, Business Entity Type 3, and self-employed individuals have around a 10% default rate.
Numeric Variable Analysis
       •	Loan Amount: Most loans range between 0 to 1 million.
       •	Credit Amount: Most loans given are for credit amounts between 0 to 1 million.
       •	Annuity Payments: Most customers pay an annuity between 0 to 50K.
       •	Income: Customers typically have an income between 0 to 1 million.
Bivariate Analysis
       •	Credit and Goods Price: These are linearly correlated. As AMT_CREDIT increases, the number of defaulters decreases.
       •	Income and Loan Amount: People with income ≤ 1 million are more likely to take loans. Those taking loans < 1.5 million could be defaulters.
       •	Family Size: Individuals with 1 to 4 children are safer to give loans to.
       •	Annuity: Those paying an annuity of 100K are likely to get loans, particularly up to 2 million.
Merged Data Analysis
       •	Previous Applications: Most previous applications for repair purposes were canceled.
       •	Repayment Patterns: 80-90% of previously canceled or refused applications are now repayers.
       •	Default Rates: Previously unused offers now have the highest default rates despite high-income customers.

#Recommendations
Target Customers
       •	Income: Below 1 million.
       •	Occupation: Accountants, core staff, managers, and laborers.
       •	Organization Type: Others, Business Entity Type 3, self-employed.
       •	Family and Housing: Married individuals with children (not more than 5) and having houses/apartments.
       •	Education: Highly educated.
       •	Gender: Preferably female.
       •	Accompaniment: Unaccompanied individuals (default rate ~8.5%).
Loan Amount and Annuity
       •	Credit Amount: Should not exceed 1 million.
       •	Annuity: Can be set at 50K depending on eligibility.
       •	Income Bracket: Below 1 million.
Precautions
       •	Avoid customers associated with:
       •	Organization: Transport type 3.
       •	Occupation: Low-skill laborers and drivers.
       •	Avoid previously unused offers and high-income customers with prior defaults.
#Conclusion
The bank should focus on targeting safer customer segments based on demographic, occupational, and financial data to minimize default rates and maximize repayment success. 
By adhering to the recommended precautions and targeting criteria, the bank can optimize its loan distribution strategy effectively.



# In[ ]:





# In[ ]:




