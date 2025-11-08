import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# ===========================================
# STEP 1: Load & Inspect Data
# ===========================================
df = pd.read_csv("data/generation_data.csv")
print("✅ Data Loaded Successfully!")
print(df.head())

# Clean column names
df.columns = df.columns.str.strip().str.lower()
print("\n🧹 Cleaned Columns:", df.columns.tolist())

# Drop unnecessary columns if they exist
for col in ['entity', 'code']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Handle missing values
print("\n🔍 Missing values before cleaning:\n", df.isna().sum())
df.fillna(0, inplace=True)
print("\n✨ Missing values after cleaning:\n", df.isna().sum())

# Ensure numeric columns are numeric
for col in df.columns:
    if col != 'year':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print("\n✅ Data cleaned and ready!")

# ===========================================
# STEP 2: Load Emission Factors
# ===========================================
em_factors = pd.read_csv("data/emission_factors.csv")
print("\n✅ Emission factors loaded!")
print(em_factors)

# Convert emission factors to a dictionary
# (match your CSV column names: fuel_type, kgCO2_per_MWh)
em_dict = dict(zip(em_factors['fuel_type'].str.lower(), em_factors['kgCO2_per_MWh']))

# ===========================================
# STEP 3: Renewable Share Calculation
# ===========================================
renewables = ['solar', 'wind', 'hydro', 'bioenergy']
df['renewable_generation_twh'] = df[renewables].sum(axis=1)

# Total generation (sum of all sources)
total_cols = ['bioenergy', 'solar', 'wind', 'hydro', 'nuclear', 'oil', 'gas', 'coal']
df['total_generation_twh'] = df[total_cols].sum(axis=1)

# Calculate renewable share (%)
df['renewable_share_percent'] = (df['renewable_generation_twh'] / df['total_generation_twh']) * 100

print("\n✅ Renewable share calculated!")
print(df[['year', 'renewable_share_percent']].tail())

# ===========================================
# STEP 4: Renewable Share Forecast (Linear Regression)
# ===========================================
X = df[['year']]
y = df['renewable_share_percent']

model = LinearRegression()
model.fit(X, y)

future_years = pd.DataFrame({'year': np.arange(df['year'].max() + 1, df['year'].max() + 6)})
future_years['forecasted_renewable_share'] = model.predict(future_years)

print("\n📈 Forecasted Renewable Share (next 5 years):")
print(future_years)

# Plot renewable share
plt.figure(figsize=(8, 5))
plt.plot(df['year'], df['renewable_share_percent'], label='Historical', color='green')
plt.plot(future_years['year'], future_years['forecasted_renewable_share'], '--', label='Forecast', color='orange')
plt.xlabel('Year')
plt.ylabel('Renewable Share (%)')
plt.title('Renewable Share Forecast')
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# STEP 5: CO₂ Emission Estimation
# ===========================================
# Calculate total emissions (MtCO₂)
df['total_emission_mtco2'] = 0
for fuel in total_cols:
    df['total_emission_mtco2'] += df[fuel] * em_dict[fuel] / 1000  # TWh * kg/MWh → MtCO₂

print("\n✅ CO₂ Emission Estimates (Historical):")
print(df[['year', 'total_emission_mtco2']].tail())

# Plot emissions
plt.figure(figsize=(8, 5))
plt.plot(df['year'], df['total_emission_mtco2'], color='red')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (Mt)')
plt.title('Total CO₂ Emissions from Electricity Generation')
plt.grid(True)
plt.show()

# ===========================================
# STEP 6: Save Results
# ===========================================
os.makedirs("results", exist_ok=True)
df.to_csv("results/forecast_and_emissions.csv", index=False)
print("\n💾 Results saved to results/forecast_and_emissions.csv")

print("\n🎉 WEEK 1 COMPLETE: Data collected, cleaned, and analyzed successfully!")


