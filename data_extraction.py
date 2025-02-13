# Step 1: Import necessary libraries for Extraction, Transformation, and DuckDB
import pandas as pd
import duckdb
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from matplotlib_venn import venn2  # Import the Venn diagram module

# Step 2: Data Extraction with Pandas
file_path = 'C:/Users/POST LAB/ecommerce_project/ecommerce_data.csv'

# Load the dataset using pandas
df = pd.read_csv(file_path)

# Display the first 5 rows to inspect the dataset
print("First 5 rows of the dataset (Pandas):")
print(df.head())

# Get the shape of the dataset (rows and columns)
print(f"\nShape of the dataset: {df.shape}")

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Show basic statistics for numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Step 3: Data Transformation using PySpark
# Initialize PySpark session
spark = SparkSession.builder \
    .appName("DataTransformation") \
    .getOrCreate()

# Read the dataset into a Spark DataFrame (same file path as before)
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Show the first few rows to check if data is loaded correctly
print("\nFirst 5 rows from PySpark DataFrame:")
df_spark.show(5)

# Remove rows with missing values
df_spark_clean = df_spark.dropna()

# Show cleaned data
print("\nCleaned data (no missing values) from PySpark:")
df_spark_clean.show(5)

# Check the schema (data types) of the cleaned data
print("\nSchema of the cleaned PySpark DataFrame:")
df_spark_clean.printSchema()

# Step 4: Convert PySpark DataFrame to Pandas DataFrame
df_clean_pandas = df_spark_clean.toPandas()

# Step 5: Load Data into DuckDB and Perform Aggregation
# Connect to DuckDB and create a table for the cleaned data
con = duckdb.connect(database=':memory:', read_only=False)

# Register the DataFrame as a table in DuckDB
con.execute("CREATE TABLE ecommerce_sales AS SELECT * FROM df_clean_pandas")

# Perform an aggregation (e.g., total sales and commission by car make)
query = """
    SELECT
        "Car Make",
        SUM("Sale Price") AS total_sales,
        SUM("Commission Earned") AS total_commission
    FROM ecommerce_sales
    GROUP BY "Car Make"
    ORDER BY total_sales DESC
"""

# Execute the query and fetch the result
result = con.execute(query).fetchall()

# Display the result of the aggregation
print("\nTotal Sales and Commission Earned by Car Make:")
for row in result:
    print(row)

# Step 6: Visualize the Results
# Extracting the car make, total sales, and total commission from the result
car_makes = [row[0] for row in result]
total_sales = [row[1] for row in result]
total_commission = [row[2] for row in result]

# Create a figure with two subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot total sales as a bar chart
ax1.bar(car_makes, total_sales, color='skyblue')
ax1.set_xlabel('Car Make')
ax1.set_ylabel('Total Sales ($)', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Create a second y-axis to plot the total commission
ax2 = ax1.twinx()
ax2.plot(car_makes, total_commission, color='orange', marker='o', linestyle='dashed')
ax2.set_ylabel('Total Commission Earned ($)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Set the title
plt.title('Total Sales and Commission Earned by Car Make')
plt.show()

# Pie Chart: Visualizing the sales distribution by car make
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(total_sales, labels=car_makes, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.title('Sales Distribution by Car Make')
plt.show()

# Stacked Bar Chart: Comparing total sales and commission by car make
fig, ax = plt.subplots(figsize=(10, 6))

# Plot stacked bar chart
ax.bar(car_makes, total_sales, label='Total Sales', color='skyblue')
ax.bar(car_makes, total_commission, bottom=total_sales, label='Commission Earned', color='orange')

# Set labels and title
ax.set_xlabel('Car Make')
ax.set_ylabel('Amount ($)')
ax.set_title('Stacked Bar Chart: Total Sales and Commission by Car Make')

# Display legend
ax.legend()

# Display the plot
plt.xticks(rotation=90)
plt.show()

# Area Chart: Visualizing total sales and commission with an area chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot area chart
ax.fill_between(car_makes, total_sales, color='skyblue', alpha=0.5, label='Total Sales')
ax.fill_between(car_makes, total_commission, color='orange', alpha=0.5, label='Commission Earned')

# Set labels and title
ax.set_xlabel('Car Make')
ax.set_ylabel('Amount ($)')
ax.set_title('Area Chart: Total Sales and Commission by Car Make')

# Display legend
ax.legend()

# Display the plot
plt.xticks(rotation=90)
plt.show()

# Step 7: Venn Diagram to visualize the intersection between sales and commission
# Let's assume a simple set-up: Set A (car makes with sales) and Set B (car makes with commission earned)
# We'll create a Venn diagram for the intersection of these two sets.

# Defining car makes for each set
set_sales = set(car_makes[:10])  # Top 10 car makes by sales
set_commission = set(car_makes[5:15])  # Top 10 car makes by commission (assuming overlap)

# Create a Venn diagram showing the intersection of car makes with sales and commission
plt.figure(figsize=(8, 8))
venn2([set_sales, set_commission], set_labels=('Sales', 'Commission'))
plt.title('Venn Diagram: Car Makes with Sales and Commission')
plt.show()

# Close the DuckDB connection
con.close()
