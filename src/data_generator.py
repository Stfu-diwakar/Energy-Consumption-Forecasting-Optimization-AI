import numpy as np
import pandas as pd
import os

def generate_smart_home_data(start_date="2025-01-01", num_days=365, random_seed=42):
    """
    Generates realistic 1-year hourly time-series energy consumption data for a smart home.
    Features daily, weekly, and annual seasonality, outdoor weather correlation, sub-metered
    appliances (HVAC, EV charger, Water heater, Refrigeration, Lighting/Sockets), and Solar PV.
    """
    np.random.seed(random_seed)
    total_hours = num_days * 24
    timestamps = pd.date_range(start=start_date, periods=total_hours, freq="h")
    
    # 1. Temporal variables
    hours = timestamps.hour
    dayofweek = timestamps.dayofweek
    dayofyear = timestamps.dayofyear
    is_weekend = (dayofweek >= 5).astype(int)
    
    # 2. Weather simulation (Temperature & Humidity)
    # Seasonal outdoor temp: Summer peak (day 200), Winter low (day 15)
    annual_temp = 18 + 12 * np.sin(2 * np.pi * (dayofyear - 100) / 365)
    daily_temp = 5 * np.sin(2 * np.pi * (hours - 8) / 24)
    weather_noise = np.random.normal(0, 1.5, total_hours)
    temperature = annual_temp + daily_temp + weather_noise
    
    # Humidity inversely correlated with temp
    humidity = np.clip(80 - 1.2 * daily_temp + np.random.normal(0, 5, total_hours), 20, 95)
    
    # 3. Occupancy simulation
    base_occupancy = np.where((hours >= 6) & (hours <= 9), 2.5,
                     np.where((hours >= 17) & (hours <= 23), 3.0, 1.0))
    weekend_boost = is_weekend * 0.8
    occupancy = np.clip(base_occupancy + weekend_boost + np.random.normal(0, 0.4, total_hours), 0, 4)
    
    # 4. Appliance Sub-metering Load Simulation (in kW)
    
    # A. HVAC (Heating & Cooling)
    target_temp = 21.0
    cooling_demand = np.maximum(0, temperature - target_temp) * 0.28
    heating_demand = np.maximum(0, target_temp - temperature) * 0.22
    hvac = (cooling_demand + heating_demand) * (1 + 0.15 * occupancy) + np.random.normal(0, 0.1, total_hours)
    hvac = np.clip(hvac, 0.2, 4.5)
    
    # B. Refrigeration (Continuous baseline with cycling)
    refrigeration = 0.15 + 0.1 * np.sin(2 * np.pi * hours / 3) + np.random.normal(0, 0.02, total_hours)
    refrigeration = np.clip(refrigeration, 0.1, 0.4)
    
    # C. Water Heater (Spikes in morning 06-09 and evening 18-21)
    water_heater_activity = np.where((hours >= 6) & (hours <= 8), 1.8,
                           np.where((hours >= 19) & (hours <= 21), 1.5, 0.1))
    water_heater = water_heater_activity * (0.8 + 0.4 * np.random.rand(total_hours))
    water_heater = np.clip(water_heater, 0.05, 2.8)
    
    # D. EV Charger (Charges 3-4 days a week between 19:00 and 02:00)
    ev_days = (dayofweek % 2 == 0).astype(int) # Every other day
    ev_hours = ((hours >= 19) | (hours <= 2)).astype(int)
    ev_charger = ev_days * ev_hours * 7.2 * (0.9 + 0.2 * np.random.rand(total_hours))
    
    # E. Lighting & Sockets (Higher evening, occupancy based)
    lighting_activity = np.where((hours >= 17) & (hours <= 23), 0.8,
                        np.where((hours >= 6) & (hours <= 8), 0.4, 0.15))
    lighting_sockets = lighting_activity * (1 + 0.2 * occupancy) + np.random.normal(0, 0.05, total_hours)
    lighting_sockets = np.clip(lighting_sockets, 0.1, 1.8)
    
    # 5. Solar PV Generation (kW)
    # Peak solar output centered around 13:00, 0 during night
    solar_base = np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 14))
    seasonal_solar_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (dayofyear - 80) / 365)
    cloud_cover = np.clip(np.random.beta(2, 5, total_hours), 0, 0.8)
    solar_generation = solar_base * seasonal_solar_factor * 5.0 * (1 - cloud_cover)
    solar_generation = np.where((hours >= 6) & (hours <= 19), solar_generation, 0.0)
    
    # 6. Total Consumption (kW)
    total_consumption = hvac + refrigeration + water_heater + ev_charger + lighting_sockets
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "occupancy": np.round(occupancy, 1),
        "hvac_kw": np.round(hvac, 3),
        "refrigeration_kw": np.round(refrigeration, 3),
        "water_heater_kw": np.round(water_heater, 3),
        "ev_charger_kw": np.round(ev_charger, 3),
        "lighting_sockets_kw": np.round(lighting_sockets, 3),
        "solar_generation_kw": np.round(solar_generation, 3),
        "total_consumption_kw": np.round(total_consumption, 3)
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smart_home_energy.csv")
    
    print("Generating 1-year Smart Home time-series dataset...")
    df = generate_smart_home_data()
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to {out_path} with shape {df.shape}")
