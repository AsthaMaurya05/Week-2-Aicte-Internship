import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# =====================================================
# Load cleaned results
# =====================================================
df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\internship project\\results\\forecast_and_emissions.csv")

print("✅ Data Preview:")
print(df.head())

# =====================================================
# FORECAST 1: Renewable Share (%)
# =====================================================

# Prepare data for Prophet
renew_df = df[['year', 'renewable_share_percent']].rename(columns={'year': 'ds', 'renewable_share_percent': 'y'})
renew_df['ds'] = pd.to_datetime(renew_df['ds'], format='%Y')

# Create and fit model
renew_model = Prophet()
renew_model.fit(renew_df)

# Create future dataframe (next 10 years)
future_renew = renew_model.make_future_dataframe(periods=10, freq='Y')

# Forecast
forecast_renew = renew_model.predict(future_renew)

# Plot forecast
renew_model.plot(forecast_renew)
plt.title("Renewable Share Forecast (with Prophet)")
plt.xlabel("Year")
plt.ylabel("Renewable Share (%)")
plt.grid(True)
plt.show()

# Save results
os.makedirs("results", exist_ok=True)
forecast_renew[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("results/prophet_renewable_forecast.csv", index=False)
print("\n💾 Renewable share forecast saved to results/prophet_renewable_forecast.csv")

# =====================================================
# FORECAST 2: Total CO₂ Emissions (MtCO₂)
# =====================================================

# Prepare data for Prophet
emission_df = df[['year', 'total_emission_mtco2']].rename(columns={'year': 'ds', 'total_emission_mtco2': 'y'})
emission_df['ds'] = pd.to_datetime(emission_df['ds'], format='%Y')

# Create and fit model
emission_model = Prophet()
emission_model.fit(emission_df)

# Create future dataframe (next 10 years)
future_emission = emission_model.make_future_dataframe(periods=10, freq='Y')

# Forecast
forecast_emission = emission_model.predict(future_emission)

# Plot forecast
emission_model.plot(forecast_emission)
plt.title("Total CO₂ Emissions Forecast (with Prophet)")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (Mt)")
plt.grid(True)
plt.show()

# Save results
forecast_emission[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("results/prophet_emission_forecast.csv", index=False)
print("\n💾 Emission forecast saved to results/prophet_emission_forecast.csv")

print("\n🎯 Both forecasts completed successfully — Renewable Share & CO₂ Emissions!")
