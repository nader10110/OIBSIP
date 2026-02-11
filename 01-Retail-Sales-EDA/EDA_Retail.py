import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Reading and Previewing Data
sns.set_style("whitegrid")
df = pd.read_csv("retail_sales_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------------------
# Step 2: Data Cleaning

df["Date"] = pd.to_datetime(df["Date"])
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# ---------------------------------------------------------------
# Step 3: Descriptive Statistics and Basic Analysis

# Total sales revenue per product category
category_sales = df.groupby('Product Category')['Total Amount'].sum()
print(category_sales)

# Top 10 customers based on total spending
top_customers = df.groupby('Customer ID')['Total Amount'].sum(
).sort_values(ascending=False).head(10)
print(top_customers)

# Detailed statistics for specific numerical features
print(df[['Quantity', 'Price per Unit', 'Total Amount', 'Age']].describe())

# ---------------------------------------------------------------------
# Step 4: Time Series Analysis [Analyze sales trends over time]

df["Month"] = df['Date'].dt.to_period("M")

# Calculating total sales per month
Monthly_sales = df.groupby("Month")["Total Amount"].sum()
print(Monthly_sales)

Monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# ---------------------------------------------------------------------
# Step 5: Customer Analysis [Demographics and purchasing behavior]

# Total sales breakdown by Gender
Gender_sales = df.groupby("Gender")["Total Amount"].sum()
print(Gender_sales)

Gender_sales.plot(kind="barh")
plt.title("Total Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")
plt.show()

# Segmenting customers into Age Groups
df["Age Group"] = pd.cut(
    df["Age"],
    bins=[18, 30, 45, 60, 70],
    labels=["18-30", "31-45", "46-60", "60+"]
)

# Average spending per age group
# [Goal: Identify which age group spends more on average]
age_sales = df.groupby("Age Group")["Total Amount"].mean()
print(age_sales)

# Visualizing average spending by Age Group
ax = age_sales.plot(kind="bar")
plt.title("Average Spending by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Total Amount")
plt.xticks(rotation=0)

# Adding data labels for better readability
ax.bar_label(ax.containers[0], padding=3, fmt='%.2f')
plt.ylim(0, age_sales.max() + 50)
plt.show()

# ---------------------------------------------------------------------
# Step 6: Product Analysis

#  Which Product Category generates the most revenue?
#  Which product has the highest demand (Quantity)?
#  Do prices vary significantly between categories?

# Total sales per product category
category_sales = df.groupby("Product Category")["Total Amount"].sum()
print(category_sales)

ay = category_sales.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.ylim(0, category_sales.max() * 1.2)
ay.bar_label(ay.containers[0], padding=5, fontsize=9, fmt='%.0f')
plt.show()

# Average price per unit across categories
avg_price = df.groupby("Product Category")["Price per Unit"].mean()
print(avg_price)

a = avg_price.plot(kind="bar")
plt.title("Average Price per Unit by Category")
plt.xlabel("Product Category")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.ylim(0, avg_price.max() * 1.2)
a.bar_label(a.containers[0], padding=5, fontsize=9, fmt='%.0f')
plt.show()

# Most purchased products based on quantity
top_Products = df.groupby("Product Category")["Quantity"].sum()
print(top_Products)

# ---------------------------------------------------------------------
# Step 7: Correlation Heatmap [Identifying relationships between variables]

numeric_df = df[["Age", "Quantity", "Price per Unit", "Total Amount"]]
correlation = numeric_df.corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------------------
# Step 8: Recommendations [Actionable insights based on the EDA]
# print("\n--- Recommendations ---")

print("1. Boost marketing and inventory in May 2023, the peak sales month.")
print("2. Offer promotions in January 2024, the lowest sales month.")
print("3. Target campaigns to female customers, the highest spending group.")
print("4. Focus on 18-30 age group, highest average spenders.")
print("5. Prioritize Electronics category, top-selling and profitable segment.")
print("6. Adjust pricing strategy carefully; price strongly affects total sales.")
print("7. Ensure sufficient inventory for popular product categories.")
