import numpy as np
import pandas as pd

class SmartHomeEnergyOptimizer:
    """
    Home Energy Management System (HEMS) Optimization Engine.
    Implements Time-of-Use (ToU) tariff pricing, Peak Shaving & Load Shifting,
    and Solar PV + Battery Storage Optimization (BESS).
    """
    def __init__(self, battery_capacity_kwh=10.0, max_charge_kw=3.3, max_discharge_kw=3.3, efficiency=0.95):
        self.battery_capacity = battery_capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = efficiency
        
        # Default Time-of-Use (ToU) Tariffs ($/kWh)
        self.tariff_off_peak = 0.10  # 22:00 - 06:00
        self.tariff_mid_peak = 0.20  # 07:00 - 16:00
        self.tariff_on_peak = 0.32   # 17:00 - 21:00

    def get_hourly_tariff(self, hours):
        """
        Returns tariff rate ($/kWh) for given hours array.
        """
        hours = np.array(hours)
        tariffs = np.where((hours >= 17) & (hours <= 21), self.tariff_on_peak,
                  np.where((hours >= 7) & (hours <= 16), self.tariff_mid_peak,
                           self.tariff_off_peak))
        return tariffs

    def optimize_schedule(self, hourly_df, enable_load_shifting=True, enable_battery=True):
        """
        Simulates baseline vs optimized consumption profile, battery SOC, and financial savings.
        """
        df = hourly_df.copy()
        if "hour" not in df.columns:
            if "timestamp" in df.columns:
                df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            else:
                df["hour"] = np.tile(np.arange(24), len(df) // 24 + 1)[:len(df)]
                
        n_steps = len(df)
        hours = df["hour"].values
        tariffs = self.get_hourly_tariff(hours)
        
        baseline_demand = df["total_consumption_kw"].values.copy()
        solar_gen = df["solar_generation_kw"].values if "solar_generation_kw" in df.columns else np.zeros(n_steps)
        ev_load = df["ev_charger_kw"].values if "ev_charger_kw" in df.columns else np.zeros(n_steps)
        
        optimized_demand = baseline_demand.copy()
        
        # 1. Load Shifting Strategy
        shifted_ev_energy = 0.0
        if enable_load_shifting:
            for i in range(n_steps):
                hr = hours[i]
                # Shift EV charging away from peak hours (17-21)
                if 17 <= hr <= 21 and ev_load[i] > 0:
                    ev_to_shift = ev_load[i]
                    optimized_demand[i] -= ev_to_shift
                    shifted_ev_energy += ev_to_shift
                    
            # Re-distribute shifted EV energy evenly across off-peak window (01:00 - 05:00)
            off_peak_mask = (hours >= 1) & (hours <= 5)
            n_off_peak = np.sum(off_peak_mask)
            if n_off_peak > 0 and shifted_ev_energy > 0:
                add_per_step = shifted_ev_energy / n_off_peak
                optimized_demand[off_peak_mask] += add_per_step

        # 2. Solar PV + Battery Storage Optimization (BESS)
        battery_soc = np.zeros(n_steps) # State of Charge in kWh
        battery_flow = np.zeros(n_steps) # Positive = discharge (power to home), Negative = charge
        net_grid_import_baseline = np.maximum(0, baseline_demand - solar_gen)
        net_grid_import_optimized = np.maximum(0, optimized_demand - solar_gen)
        
        peak_grid_target = float(np.max(net_grid_import_baseline)) * 0.65 # Target max grid import ceiling
        
        if enable_battery:
            current_soc = self.battery_capacity * 0.3 # Initial 30% SOC
            for i in range(n_steps):
                hr = hours[i]
                net_load = net_grid_import_optimized[i]
                excess_solar = np.maximum(0, solar_gen[i] - optimized_demand[i])
                
                charge_amount = 0.0
                discharge_amount = 0.0
                
                # Priority 1: Charge battery from excess solar
                if excess_solar > 0 and current_soc < self.battery_capacity:
                    max_can_charge = min(self.max_charge_kw, (self.battery_capacity - current_soc) / self.efficiency)
                    charge_amount = min(excess_solar, max_can_charge)
                    current_soc += charge_amount * self.efficiency
                    battery_flow[i] = -charge_amount
                    
                # Priority 2: Charge battery from grid during off-peak if SOC low (<80%)
                elif (0 <= hr <= 4) and current_soc < (self.battery_capacity * 0.8):
                    max_can_charge = min(self.max_charge_kw, (self.battery_capacity - current_soc) / self.efficiency)
                    current_import = net_grid_import_optimized[i]
                    headroom = max(0.0, peak_grid_target - current_import)
                    charge_amount = min(max_can_charge, headroom)
                    if charge_amount > 0:
                        current_soc += charge_amount * self.efficiency
                        battery_flow[i] = -charge_amount
                        net_grid_import_optimized[i] += charge_amount
                    
                # Priority 3: Discharge battery during peak hours (17-21) to supply home load
                elif (17 <= hr <= 21) and net_load > 0 and current_soc > 0.5:
                    max_can_discharge = min(self.max_discharge_kw, current_soc)
                    discharge_amount = min(net_load, max_can_discharge)
                    current_soc -= discharge_amount
                    battery_flow[i] = discharge_amount
                    net_grid_import_optimized[i] -= discharge_amount
                    
                battery_soc[i] = current_soc

        # 3. Calculate Financial & Performance Impact
        baseline_cost = np.sum(net_grid_import_baseline * tariffs)
        optimized_cost = np.sum(net_grid_import_optimized * tariffs)
        cost_savings = baseline_cost - optimized_cost
        pct_cost_savings = (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        
        peak_demand_baseline = float(np.max(net_grid_import_baseline))
        peak_demand_optimized = float(np.max(net_grid_import_optimized))
        peak_shaving_kw = max(0.0, peak_demand_baseline - peak_demand_optimized)
        pct_peak_shaving = (peak_shaving_kw / peak_demand_baseline * 100) if peak_demand_baseline > 0 else 0.0
        
        solar_self_consumed = np.sum(np.minimum(solar_gen, optimized_demand + np.maximum(0, -battery_flow)))
        solar_total = np.sum(solar_gen)
        solar_self_consumption_ratio = (solar_self_consumed / solar_total * 100) if solar_total > 0 else 0.0
        
        result = {
            "baseline_cost": round(float(baseline_cost), 2),
            "optimized_cost": round(float(optimized_cost), 2),
            "cost_savings": round(float(cost_savings), 2),
            "pct_cost_savings": round(float(pct_cost_savings), 1),
            "peak_demand_baseline": round(peak_demand_baseline, 2),
            "peak_demand_optimized": round(peak_demand_optimized, 2),
            "peak_shaving_kw": round(peak_shaving_kw, 2),
            "pct_peak_shaving": round(float(pct_peak_shaving), 1),
            "solar_self_consumption_ratio": round(float(solar_self_consumption_ratio), 1),
            "tariffs": tariffs.tolist(),
            "baseline_demand": net_grid_import_baseline.tolist(),
            "optimized_demand": net_grid_import_optimized.tolist(),
            "battery_soc": battery_soc.tolist(),
            "solar_generation": solar_gen.tolist()
        }
        
        return result

