# %% [markdown]
# #### Clustering analysis and customer segmentation for Retail data

# %% [markdown]
# ##### Step1: import Required Libraries

# %%
import pandas as pd
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# %% [markdown]
# ##### Step2: Import Dataset

# %%
df = pd.read_csv('OnlineRetail.csv',encoding='unicode_escape')
df.head()

# %% [markdown]
# #### Step 3: Data Exploration and Data Cleaning

# %%
df.info()

# %%
#1: Drop Nulls
df.dropna(inplace=True)
df.isnull().sum()


# %%
df.describe(include='all')

# %%
df[['Quantity','UnitPrice']].hist(figsize=(10,5))
plt.show()


# %% [markdown]
# ##### We can see that Quantity has negative values, which indicates sum returns. This might affect the data, so we will remove them

# %%
df=df[df['Quantity']>0]
df=df[df['UnitPrice']>0]

# %%
df.describe(include='all')

# %%

df.dtypes

# %%
#Change Invoice to date time
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')


# %%
#Customer Total amount spend
df['Total_Amount_Spent']= df['Quantity'] * df['UnitPrice']

total_amount = df['Total_Amount_Spent'].groupby(df['CustomerID']).sum()
total_amount = pd.DataFrame(total_amount).reset_index()
total_amount.head()

# %%
#df.drop(columns=['Day_of_Week_Name', 'Day_of_Week_Integer'], inplace=True)


# %%
# Create additional column 'Day_of_Week_Integer'
df['Day_of_Week_Integer'] = df['InvoiceDate'].dt.dayofweek  # Adding 1 to start from 1 for Sunday
df['Month'] = df['InvoiceDate'].dt.month
df['Day_of_Week'] = df['InvoiceDate'].dt.day_name()


df.head(2)

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df['Total_Amount_Spent'], shade=True, color='blue')
plt.title('Kernel Density Estimate (KDE) Plot for Total Amount Spent')
plt.xlabel('Total Amount Spent')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df['Day_of_Week_Integer'], shade=True, color='blue')
plt.title('Kernel Density Estimate (KDE) Plot for Day of week')
plt.xlabel('Total Amount Spent')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# %%
distinct_day_of_week = df[['Day_of_Week_Integer', 'Day_of_Week']].drop_duplicates().reset_index(drop=True)
print(distinct_day_of_week)

# %%
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
yearly_counts = df['InvoiceDate'].dt.year.value_counts().sort_index()
month_counts = df['InvoiceDate'].dt.month.value_counts().sort_index()

month_counts.plot(kind='bar')
plt.title('Monthly Distribution')
plt.xlabel('Month')
plt.ylabel('Distribution ')
plt.show()

# %%
# Line plot for Date and Amount Spent

Daily_spend = df.groupby(df['InvoiceDate'].dt.date)['Total_Amount_Spent'].sum().reset_index()

Daily_spend = pd.DataFrame(Daily_spend).reset_index()
Daily_spend = Daily_spend.sort_values(by='InvoiceDate')

plt.figure(figsize=(20, 6))
#plt.plot(Daily_spend['InvoiceDate'], Daily_spend['Total_Amount_Spent'], marker='o', color='blue', linestyle='-')
plt.plot(Daily_spend['InvoiceDate'], Daily_spend['Total_Amount_Spent'], color='green', linestyle='-')

plt.title('Line Plot for Date and Amount Spent')
plt.xlabel('Date')
plt.ylabel('Amount Spent')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
### Country Group Spend
total_spending = df['Total_Amount_Spent'].sum()

Country_spend = df['Total_Amount_Spent'].groupby(df['Country']).sum()
Country_spend = pd.DataFrame(Country_spend).reset_index()

Country_spend['Percent_spend'] = ((Country_spend['Total_Amount_Spent'] / total_spending) * 100).round(2).astype(str) + '%'
#Country_spend = pd.DataFrame(Country_spend, Percent_spend).reset_index()

Country_spend.head()

# %%
# Bar plots
plt.figure(figsize=(6, 5))
top_10_Countries = df.groupby('Country')['Total_Amount_Spent'].sum().nlargest(5).reset_index()
sns.barplot(data=top_10_Countries, x='Country', y='Total_Amount_Spent', color='skyblue')

#sns.barplot(data=Country_spend, x='Country', y='Percent_spend', color='green')

plt.title('Top 10 countries by Amount Spent')
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Total Amount Spent')
plt.tight_layout()
plt.show()


# %%
# Sort the DataFrame by 'Total_Amount_Spent' in descending order
df_sorted = df.sort_values(by='Total_Amount_Spent', ascending=False)

# Group by StockCode and sum Total_Amount_Spent, including Description
top_10_stock_codes = df_sorted.groupby(['StockCode', 'Description'])['Total_Amount_Spent'].sum().nlargest(5).reset_index()

# Bar plot for top 10 stock codes
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_stock_codes, x='StockCode', y='Total_Amount_Spent', hue='Description', dodge=False)

plt.title('Top 5 stocks')
plt.xlabel('StockCode')
plt.ylabel('Total Amount Spent')
plt.xticks(rotation=45)

# Customize legend to show in a separate box
plt.legend(title='Description', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
df['CustomerID'] = df['CustomerID'].astype(int)


# %% [markdown]
# #### Step 4: Recency frequency Analysis
# ###### Recency, Frequency, Monetary model (RFM), is a behavior based analysis technique used to segment customers by examining their transaction history

# %%
df.info()

# %% [markdown]
# ###### Step 4a: Recency

# %%
last_transaction_date = df.groupby('CustomerID')['InvoiceDate'].max()

reference_date = max(df['InvoiceDate'])
last_transaction_date = pd.to_datetime(last_transaction_date)

days_difference = pd.DataFrame((reference_date - last_transaction_date).dt.days).reset_index()
days_difference = days_difference.rename(columns={'InvoiceDate': 'recency'})


days_difference.head()

# %% [markdown]
# ###### Step 4b: Frequency

# %%
frequency = df.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency = frequency.rename(columns={'InvoiceNo': 'frequency'})

frequency.head()

# %% [markdown]
# ###### Step 4c: Spend

# %%
#### Spend
Spend = df.groupby('CustomerID')['Total_Amount_Spent'].sum()
Spend = Spend.reset_index()
Spend = Spend.rename(columns={'Total_Amount_Spent': 'Spend'})

Spend.head()


# %%
#COmbining Recency frequency and Spend 
merged_rfs = days_difference.merge(frequency, on='CustomerID', how='inner')

merged_rfs = merged_rfs.merge(Spend, on='CustomerID', how='inner')

print(merged_rfs.head())

# %% [markdown]
# ##### Step 5: Outllier Analysis for RFS

# %%
#merged_rfs[['recency', 'frequency', 'Spend']].boxplot()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot box plots for each feature
merged_rfs['recency'].plot(kind='box', ax=axes[0])
axes[0].set_title('Recency')

merged_rfs['frequency'].plot(kind='box', ax=axes[1])
axes[1].set_title('Frequency')

merged_rfs['Spend'].plot(kind='box', ax=axes[2])
axes[2].set_title('Spend')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Creating subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot box plots for each rfs
axes[0].scatter(merged_rfs['recency'], merged_rfs['CustomerID'])
axes[0].set_title('Recency')

axes[1].scatter(merged_rfs['frequency'], merged_rfs['CustomerID'])
axes[1].set_title('Frequency')

axes[2].scatter(merged_rfs['Spend'], merged_rfs['CustomerID'])
axes[2].set_title('Spend')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Calculate z-scores for each feature
z_scores = (merged_rfs[['recency', 'frequency', 'Spend']] - merged_rfs[['recency', 'frequency', 'Spend']].mean()) / merged_rfs[['recency', 'frequency', 'Spend']].std()

# First defined thresholds for identifying outliers
threshold_recency = 3
threshold_frequency = 3
threshold_spend = 3

# Find outliers for each feature
outliers_recency = z_scores['recency'].abs() > threshold_recency
outliers_frequency = z_scores['frequency'].abs() > threshold_frequency
outliers_spend = z_scores['Spend'].abs() > threshold_spend

# Combining outlier flags across RFS features
outliers = outliers_recency | outliers_frequency | outliers_spend

# Printing the indices of the outliers
print("Indices of outliers:")
print(merged_rfs[outliers].index)

# Print the number of outliers per feature
print("\nNumber of outliers per feature:")
print("Recency:", outliers_recency.sum())
print("Frequency:", outliers_frequency.sum())
print("Spend:", outliers_spend.sum())



# %%
#Removing Outliers
rfs= merged_rfs[~outliers]
rfs.describe()


# %% [markdown]
# ##### Step 6: Feature Scaling before Clustering

# %%
from sklearn.preprocessing import StandardScaler
X=rfs.iloc[:,1:]

scaler = StandardScaler()
X = scaler.fit_transform(X)
#scaled_RFS.head(5)
X

# %%
from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# %% [markdown]
# ##### Step 7: K-means Clustering

# %%
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Instantiate the KMeans model
model = KMeans()

# Instantiate the KElbowVisualizer with the KMeans model
visualizer = KElbowVisualizer(model, k=(2,11)) # Range of k values to try

# Fitting  visualizer to the data
visualizer.fit(X)

# Finalize and render the figure
visualizer.show()


# %%

kmeans= KMeans(n_clusters=4,n_init='auto',random_state=42)

kmeans.fit(X)

# %%
#print("Inertia:", kmeans.inertia_)


# %%
kmeans.predict(X)

# %%
rfs["Cluster"]=kmeans.labels_
rfs.head()

# %%
import matplotlib.pyplot as plt

# Group the DataFrame by the 'Cluster' column
grouped = rfs.groupby('Cluster')

#cmap = plt.get_cmap('tab10')  # You can use any colormap available in Matplotlib

# Create a scatter plot for each cluster
for cluster, data in grouped:
    plt.scatter(data['recency'], data['Spend'], label=f'Cluster {cluster}')

# Add labels and title
plt.xlabel('Recency(Days last Spend)')
plt.ylabel('Spend (Amount)')
plt.title('Scatter Plot of Spend  and Recency')

# Add legend
plt.legend()

# Show the plot
plt.show()


# %%
# Create a scatter plot for each cluster
for cluster, data in grouped:
    plt.scatter(data['frequency'], data['Spend'], label=f'Cluster {cluster}')

# Add labels and title
plt.xlabel('Frequency (How often they Spend)')
plt.ylabel('Spend (Amount)')
plt.title('Scatter Plot Frequency and Spend')

# Add legend
plt.legend()

# Show the plot
plt.show()


# %% [markdown]
# Summary
# ##### Cluster 0: High recency, low frequency , low Spend
# ##### Cluster 1: Low recency, High Frequency, moderate Spend
# ##### Cluster 2: Low Recency, Low Frequency, Low Spend
# ##### Cluster 3: Low Recency,  High Frequency, High Spend

# %%
final_df = pd.merge(df, rfs, on='CustomerID')
final_df.head(2)


# %%
grouped = final_df.groupby('Cluster')

# Initialize a figure and axis for the plot
fig, ax = plt.subplots()

# Iterate over each cluster
for cluster, cluster_data in grouped:
    # Calculate the total amount spent for each item in the cluster
    item_amounts = cluster_data.groupby('Description')['Total_Amount_Spent'].sum()
    
    # Sort the items by total amount spent and select the top 3 items
    top_items = item_amounts.sort_values(ascending=False).head(1)
    
    # Plot the top items for the cluster
    ax.barh([f'Cluster {cluster}: {item}' for item in top_items.index], top_items.values, label=f'Cluster {cluster}')

# Add labels and title
ax.set_xlabel('Amount Spent')
ax.set_ylabel('Item')
ax.set_title('Top 3 Items by Amount Spent per Cluster')

# Add legend
ax.legend()

# Show the plot
plt.show()

# %%
final_df.columns

# %%
# Group the DataFrame by 'Cluster' and 'day_of_week' and sum the amount spent
grouped = final_df.groupby(['Cluster', 'Day_of_Week'])['Total_Amount_Spent'].sum()

# Find the day of the week with the maximum total amount spent for each cluster
max_spending_day = grouped.groupby('Cluster').idxmax()

# Convert the result to a DataFrame
max_spending_day_df = pd.DataFrame(max_spending_day.values.tolist(), columns=['Cluster', 'Day of Week'])

# Count the occurrences of each day of the week
day_counts = max_spending_day_df['Day of Week'].value_counts()

# Plot the bar chart
day_counts.plot(kind='bar', color='skyblue')

# Add labels and title
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.title('Most Spending Day of the Week per Cluster')

# Show the plot
plt.show()


# Print the result
for cluster, day in max_spending_day:
    print(f"Cluster {cluster}: {day}")


