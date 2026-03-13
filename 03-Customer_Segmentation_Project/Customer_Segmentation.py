import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("ifood_df.csv")

print("Shape os dataset", df.shape)

print(df.columns)
print(df.head())
print(df.info())
print(df.describe())

features = df[[
    "Income",
    "Recency",
    "MntTotal",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumCatalogPurchases"
]]

print(features.head())

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(scaled_features[:5])

wcss = []  # Within Cluster Sum of Squares : cluster مقدار الخطأ داخل كل

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(scaled_features)

df["Cluster"] = clusters

print(df[["Income", "MntTotal", "Cluster"]].head())

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=df["Income"],
    y=df["MntTotal"],
    hue=df["Cluster"],
    palette="Set1"
)

plt.title("Customer Segmentation")
plt.xlabel("Income")
plt.ylabel("Total Spending")
plt.show()


# Insights:
# 1- Customers were segmented into 3 groups based on income and spending behavior.
# 2- High-income customers tend to spend significantly more.
# 3- Medium-value customers represent a growth opportunity for targeted marketing.
