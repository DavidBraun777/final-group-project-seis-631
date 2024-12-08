#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os


# In[3]:


# Define file paths
files = {
    "inventory": "./Metro_invt_fs_uc_sfrcondo_sm_month.csv",
    "market_heat_index": "./Metro_market_temp_index_uc_sfrcondo_month.csv",
    "days_to_pending": "./Metro_mean_doz_pending_uc_sfrcondo_sm_month.csv",
    "new_construction_sales": "./Metro_new_con_sales_count_raw_uc_sfrcondo_month.csv",
    "sales_count": "./Metro_sales_count_now_uc_sfrcondo_month.csv",
    "zhvf_growth": "./Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zhvi": "./Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zori": "./Metro_zori_uc_sfrcondomfr_sm_month.csv",
    "zorf_growth": "./National_zorf_growth_uc_sfr_sm_month.csv"
}

# Load all files into dataframes
dataframes = {key: pd.read_csv(path) for key, path in files.items()}

# Display first few rows of each dataset
for name, df in dataframes.items():
    print(f"Dataset: {name}")
    print(df.head(), "\n")


# In[4]:


# Example of cleaning
for name, df in dataframes.items():
    # Drop rows with excessive missing data
    df.dropna(thresh=len(df.columns) * 0.8, inplace=True)
    # Convert date columns to datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    dataframes[name] = df

print("Data cleaning completed.")


# In[8]:


# Identify date columns (they usually look like "YYYY-MM-DD")
date_columns = inventory_data.columns[5:]  # Assuming first 5 columns are metadata

# Reshape the dataset to long format (melt only date columns)
inventory_data_long = pd.melt(
    inventory_data,
    id_vars=["RegionName", "StateName"],  # Keep these columns as identifiers
    value_vars=date_columns,  # Only melt date columns
    var_name="Date",
    value_name="Inventory"
)

# Convert the 'Date' column to datetime
inventory_data_long["Date"] = pd.to_datetime(inventory_data_long["Date"])

# Group by region and plot inventory trends
regions_to_plot = inventory_data_long["RegionName"].unique()[:5]  # Limit to first 5 regions for clarity
for region in regions_to_plot:
    region_data = inventory_data_long[inventory_data_long["RegionName"] == region]
    plt.plot(region_data["Date"], region_data["Inventory"], label=region)

plt.title("Inventory Trends")
plt.xlabel("Date")
plt.ylabel("Inventory Count")
plt.legend()
plt.show()


# In[12]:


print(sales_data.columns)


# In[20]:


# Identify the latest date column
latest_date = sales_data.columns[sales_data.columns.str.match(r"\d{4}-\d{2}-\d{2}")].max()


# In[21]:


# Identify the correct growth column
growth_column = sales_data.columns[sales_data.columns.str.match(r"2024-11-30")].max()  # Adjust date as per data


# In[22]:


# Ensure numeric values for calculations
sales_data[latest_date] = pd.to_numeric(sales_data[latest_date], errors="coerce")
sales_data[growth_column] = pd.to_numeric(sales_data[growth_column], errors="coerce")


# In[23]:


purchase_price = sales_data[latest_date]


# In[24]:


# Identify latest date and growth columns
latest_date = sales_data.columns[sales_data.columns.str.match(r"\d{4}-\d{2}-\d{2}")].max()
growth_column = sales_data.columns[sales_data.columns.str.match(r"2024-11-30")].max()

# Convert columns to numeric
sales_data[latest_date] = pd.to_numeric(sales_data[latest_date], errors="coerce")
sales_data[growth_column] = pd.to_numeric(sales_data[growth_column], errors="coerce")

# Calculate purchase price and projected price
purchase_price = sales_data[latest_date]
sales_data["Projected_Price"] = purchase_price * (1 + sales_data[growth_column] / 100)

# Mortgage calculations
down_payment = purchase_price * 0.20
loan_amount = purchase_price - down_payment
annual_interest_rate = 0.03
loan_term_years = 30
monthly_mortgage = (
    loan_amount * annual_interest_rate / 12 /
    (1 - (1 + annual_interest_rate / 12) ** (-loan_term_years * 12))
)
total_mortgage_payments = monthly_mortgage * loan_term_years * 12

# Profit and ROI calculations
sales_data["Profit"] = sales_data["Projected_Price"] - purchase_price - total_mortgage_payments
sales_data["Sales_ROI (%)"] = (sales_data["Profit"] / purchase_price) * 100

# Display top regions by Sales ROI
top_sales_roi = sales_data.sort_values(by="Sales_ROI (%)", ascending=False).head(10)
print(top_sales_roi[["RegionName", "Sales_ROI (%)", "Profit"]])


# In[29]:


# Confirm latest_rent_date exists in the zori dataset
print("Rental Data Columns in ZORI:", zori.columns)


# In[30]:


# Merge rental data with sales data
rental_data = sales_data.merge(zori, on="RegionName", suffixes=("_sales", "_rent"))

# Confirm the column name after the merge
print("Rental Data Columns After Merge:", rental_data.columns)


# In[31]:


# Use the renamed column for rental income
latest_rent_date_renamed = f"{latest_rent_date}_rent"

# Ensure the rental income column is numeric
rental_data[latest_rent_date_renamed] = pd.to_numeric(rental_data[latest_rent_date_renamed], errors="coerce")

# Calculate monthly and annual rental income
rental_data["Monthly_Rental_Income"] = rental_data[latest_rent_date_renamed]
rental_data["Annual_Rental_Income"] = rental_data["Monthly_Rental_Income"] * 12


# In[32]:


# Assumptions for costs
maintenance_costs = 200  # $200/month for maintenance
annual_taxes = 0.01 * purchase_price  # Example: 1% of purchase price annually

# Net annual rental income
rental_data["Net_Annual_Rental_Income"] = (
    rental_data["Annual_Rental_Income"]
    - (maintenance_costs * 12)
    - annual_taxes
    - total_mortgage_payments / loan_term_years
)

# Calculate rental ROI
rental_data["Rental_ROI (%)"] = (rental_data["Net_Annual_Rental_Income"] / purchase_price) * 100

# Display top regions by Rental ROI
top_rental_roi = rental_data.sort_values(by="Rental_ROI (%)", ascending=False).head(10)
print(top_rental_roi[["RegionName", "Rental_ROI (%)", "Net_Annual_Rental_Income"]])


# In[33]:


# Rental ROI Visualization
plt.figure(figsize=(10, 6))
plt.bar(top_rental_roi["RegionName"], top_rental_roi["Rental_ROI (%)"])
plt.title("Top Regions by Rental ROI")
plt.xlabel("Region")
plt.ylabel("Rental ROI (%)")
plt.xticks(rotation=45, ha='right')
plt.show()


# In[40]:


combined_roi = combined_roi.replace([np.inf, -np.inf], np.nan)
combined_roi = combined_roi.dropna(subset=["Sales_ROI (%)", "Rental_ROI (%)"])
combined_roi = combined_roi[(combined_roi["Rental_ROI (%)"] > -1e6) & (combined_roi["Rental_ROI (%)"] < 1e6)]
print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].describe())


# In[41]:


scaler = MinMaxScaler()
combined_roi[["Sales_ROI_norm", "Rental_ROI_norm"]] = scaler.fit_transform(
    combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]]
)


# In[42]:


print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].describe())
print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].isna().sum())
print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].isin([np.inf, -np.inf]).sum())


# In[45]:


print(combined_roi.columns)


# In[48]:


combined_roi = combined_roi.replace([np.inf, -np.inf], np.nan)
combined_roi = combined_roi.dropna(subset=["Sales_ROI (%)", "Rental_ROI (%)"])


# In[49]:


print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].describe())
print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].isna().sum())
print(combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]].isin([np.inf, -np.inf]).sum())


# In[50]:


from sklearn.preprocessing import MinMaxScaler

# Re-apply MinMaxScaler to ensure normalized columns exist
scaler = MinMaxScaler()
combined_roi[["Sales_ROI_norm", "Rental_ROI_norm"]] = scaler.fit_transform(
    combined_roi[["Sales_ROI (%)", "Rental_ROI (%)"]]
)

# Verify normalized columns
print(combined_roi[["Sales_ROI_norm", "Rental_ROI_norm"]].head())


# In[53]:


# Calculate overall ROI score (weighted equally for sales and rental)
combined_roi["Overall_ROI_Score"] = (
    combined_roi["Sales_ROI_norm"] + combined_roi["Rental_ROI_norm"]
) / 2

# Rank regions based on the overall ROI score
top_overall_regions = combined_roi.sort_values(by="Overall_ROI_Score", ascending=False).head(10)

# Display the top regions
print(top_overall_regions[["RegionName", "Sales_ROI (%)", "Rental_ROI (%)", "Overall_ROI_Score"]])


# In[52]:


import matplotlib.pyplot as plt

# Plot top regions
top_overall_regions.plot(
    x="RegionName", y="Overall_ROI_Score", kind="barh", legend=False, figsize=(10, 6)
)
plt.title("Top 10 Regions by Overall ROI Score")
plt.xlabel("Overall ROI Score")
plt.ylabel("Region Name")
plt.gca().invert_yaxis()  # To display the highest score at the top
plt.show()


# # Sales ROI
# Question: How much profit can I expect from sales after paying a mortgage?
# 
# - Profit Formula: 

# In[54]:


# TODO:

# 5. Sales ROI
# Question: How much profit can I expect from sales after paying a mortgage?

# Covered: Fully calculated in the Sales ROI section:
# Profit Formula: 
# Profit
# =
# Projected Sale Price
# −
# Purchase Price
# −
# Mortgage Payments
# Profit=Projected Sale Price−Purchase Price−Mortgage Payments
# ROI Formula: 
# Sales ROI
# =
# Profit
# Purchase Price
# ×
# 100
# Sales ROI= 
# Purchase Price
# Profit
# ​
#  ×100
# Top regions with high ROI identified.
# 6. Rental ROI
# Question: How much monthly income can I expect from renting after upgrades, taxes, and mortgage?

# Covered: Fully calculated in the Rental ROI section:
# Net Annual Rental Income: 
# Annual Rental Income
# −
# Costs (taxes, maintenance, mortgage payments)
# Annual Rental Income−Costs (taxes, maintenance, mortgage payments)
# ROI Formula: 
# Rental ROI
# =
# Net Annual Rental Income
# Purchase Price
# ×
# 100
# Rental ROI= 
# Purchase Price
# Net Annual Rental Income
# ​
#  ×100
# Top regions with high rental ROI identified.
# BONUS: Visualizations
# Visualizations: Suggested plots for both Sales ROI and Rental ROI.
# Additional Insight: Regions with both high Sales and Rental ROI can be combined into a recommendation chart.
# Final Deliverable
# You are now ready to:

# Prepare a written summary: Highlight key insights from Sales and Rental ROI, with recommendations for investment.
# Generate plots: Use the provided code to create visualizations.
# Combine findings: Create a concise recommendation for the investor based on the data.


# In[ ]:




