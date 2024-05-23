# Retail Customer Behaviour Analysis 
This project Details the analysis conducted to Identify customer behaviour in an ecommerce data set.
The project showcases the adoption of Recency, frequency and spend in understanding customer behaviour in the data set

Customer segementation was also identified using KMeans Clustering algorithm.

## Customer Behaviour analysis and segementation using KMeans.

**Project description:**  This is a data analysis project that involves reading, cleaning, exploring, and performing advanced analyses on a retail dataset.
The project uses Kmeans Clustering algorithm for identifying .

This project covers data cleaning, exploratory data analysis (EDA), data visualization, and customer segmentation using machine learning techniques. 

### 1. Importing libraries and Loading the data

 I used a dataset named OnlineRetail.csv, which contains transactional data from an online retail store. The primary goal was to explore the dataset, clean it, visualize important patterns, and perform customer segmentation
![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/1dd5cc1a-05f8-4f53-82ab-93ed3cef801a)\ ![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/ef198a92-0413-4d6f-9b09-b7abd3f70b30)


### 2. Data Visualization

After removing the nulll values from the dataset, I created two addditonal columsn 'Month' and 'Day of Week' based of tdh invoice date column. 
The goal was to further understand if there were any pointers to the actual day of week, or some other insights from the date
Below  are some visualizatiions showing trend and distributions.

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/12ea5d21-e2d4-4d01-9b60-3e779eadfd20)

The Barplots show that there were no transactions on saturday, and even though the highest transactions were in the 11th month, the total amount spent was highest in the first month in January

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/39f1f7ee-a0ab-4aa9-874f-a921c7b192af)


In addition this visuals show for teh top 5 countries and the tiop stock items purchased 
![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/fa2c6cc6-828c-4e81-a884-c5253ae4185c)



### 3.Analysis using Recency Frequency and Spend (RFS)

Recency, Frequency, Monetary model (RFM), is a behavior based analysis technique used to segment customers by examining their transaction history. Recency is calculated as the number of days since the last purchase
Recency is calculated, frequency is the number of transactions per customer, and Spend is the total amount spent per customer.

```python
last_transaction_date = df.groupby('CustomerID')['InvoiceDate'].max()
reference_date = max(df['InvoiceDate'])
days_difference = (reference_date - last_transaction_date).dt.days
days_difference = days_difference.reset_index().rename(columns={'InvoiceDate': 'recency'})
```
These 2 metrics are then merged into a single dataframe.

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/0edc2421-bd39-4d44-919c-a4229c163394)

And these are boxplots to show the Outliers in the RFS data distributions

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/a58e8814-c381-4661-8e36-f87844461b16)

The outliers were mostly in the Frequency and spend coluumns and were removed before applying Kmeans algorthm



### 4. Feature scaling

The outliers were removed in X, and then feature scaling 

```python
from sklearn.preprocessing import StandardScaler
X=rfs.iloc[:,1:]
scaler = StandardScaler()
X = scaler.fit_transform(X)
```


### 4. Clustering customers using KMeans

I adopted the Yellowbirck cluster for visualizing the Kmeans distorion score Elbow. As shown below, the k elbows at 4, indicating 4 clusters.
![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/1398c04e-8d19-407a-a8d1-6653e02ad672)

Then I fitted the Kmeans and then updated the RFS dataframe with teh clusters identified.

```python
kmeans= KMeans(n_clusters=4,n_init='auto',random_state=42)
kmeans.fit(X)
```

### 4. Results and findings : recency and Spend

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/50c76e6a-514d-4ffd-bbab-df17a4b89970)

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/9dac9e13-6710-4373-ae05-4e5ab83c8d22)

These are the four clusters identified from the results

##### Cluster 0: High recency, low frequency , low Spend
##### Cluster 1: Low recency, High Frequency, moderate Spend
##### Cluster 2: Low Recency, Low Frequency, Low Spend
##### Cluster 3: Low Recency,  High Frequency, High Spend


In addition, these are the top items by Amount spent.

![image](https://github.com/moyinajayi/moyinajayi.github.io/assets/9222400/828789d9-8209-4292-956d-362db448f755)


For more details see [Repository](https://github.com/moyinajayi/Retail_customer_Clustering)
