import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("AB_NYC_2019.csv")

print(df.head())
print(df.info())
print(df.describe())

# Display the number of missing values ​​in each column
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# تعويض name و host_name
# df["column"] = df["column"].fillna(...)
df["name"] = df["name"].fillna("unKnown")
df["host_name"] = df["host_name"].fillna("unKnown")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert last_review to datetime
print("\n")
df["last_review"] = pd.to_datetime(df["last_review"])
print(df["last_review"].dtype)

# Calculating the number of duplicate rows
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# Outlier
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["price"])
plt.title("Boxplot of price")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["minimum_nights"])
plt.title("Boxplot of Minimum Nights")
plt.show()

# Price Outliers
Q1_price = df['price'].quantile(0.25)
Q3_price = df['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price

df = df[(df['price'] >= Q1_price - 1.5*IQR_price) &
        (df['price'] <= Q3_price + 1.5*IQR_price)]

# Minimum Nights Outliers
Q1_nights = df['minimum_nights'].quantile(0.25)
Q3_nights = df['minimum_nights'].quantile(0.75)
IQR_nights = Q3_nights - Q1_nights
df = df[(df["minimum_nights"] >= Q1_nights - 1.5*IQR_nights) &
        (df["minimum_nights"] <= Q3_nights + 1.5*IQR_nights)]


plt.figure(figsize=(10, 5))
sns.boxplot(x=df["price"])
plt.title("Boxplot of price")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["minimum_nights"])
plt.title("Boxplot of Minimum Nights")
plt.show()

print("Memory usage before optimization:")
print(df.memory_usage(deep=True).sum())

df["neighbourhood_group"] = df["neighbourhood_group"].astype("category")
df["neighbourhood"] = df["neighbourhood"].astype("category")
df["room_type"] = df["room_type"].astype("category")

print("\n Memory usage after optimization:")
print(df.memory_usage(deep=True).sum())

df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

df["review_year"] = df["last_review"].dt.year
df["review_month"] = df["last_review"].dt.month
df["review_day"] = df["last_review"].dt.day

print(df[["last_review", "review_year", "review_month", "review_day"]].head())

df["review_year"] = df["review_year"].astype("Int64")
df["review_month"] = df["review_month"].astype("Int64")
df["review_day"] = df["review_day"].astype("Int64")

print(df.dtypes)

df.to_csv("AB_NYC_2019_Cleaned.csv", index=False)
