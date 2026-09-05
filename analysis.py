import pandas as pd
import numpy as np

# 1. Dataset Load
df = pd.read_csv("DL_PROJECT_BK.csv")

# 2. Duplicate rows remove 
df = df.drop_duplicates()

# 3. Missing values check 
print("Missing values before cleaning:\n", df.isnull().sum())

# 4. Text columns-la extra spaces / lowercase standardize 
text_cols = df.select_dtypes(include=["object", "str"]).columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# 5. Time format convert  
def convert_to_decimal_hours(value):
    try:
        hours, minutes = str(value).split(":")
        return float(hours) + float(minutes) / 60
    except:
        return None

df["delivery_time_hours"] = df["delivery_time_hours"].apply(convert_to_decimal_hours)
df["expected_time_hours"] = df["expected_time_hours"].apply(convert_to_decimal_hours)

# 6. Convert failed rows (NaN) remove 
df = df.dropna(subset=["delivery_time_hours", "expected_time_hours"])

# 7. Delay Analysis
df["delay_hours"] = df["delivery_time_hours"] - df["expected_time_hours"]

df["delay_status"] = df["delay_hours"].apply(
    lambda x: "Delayed" if x > 0 else "On Time"
)

# 8. Decimal hours 
def decimal_to_hhmm(value):
    if pd.isna(value):
        return None
    is_negative = value < 0
    value = abs(value)
    hours = int(value)
    minutes = int(round((value - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    sign = "-" if is_negative else ""
    return f"{sign}{hours:02d}:{minutes:02d}"

df["delivery_time_hhmm"] = df["delivery_time_hours"].apply(decimal_to_hhmm)
df["expected_time_hhmm"] = df["expected_time_hours"].apply(decimal_to_hhmm)
df["delay_hhmm"] = df["delay_hours"].apply(decimal_to_hhmm)

# 9. Extra useful columns for dashboard
df["distance_category"] = pd.cut(
    df["distance_km"],
    bins=[0, 100, 250, float("inf")],
    labels=["Short", "Medium", "Long"]
)

df["delay_percentage"] = np.where(
    df["expected_time_hours"] > 0,
    round((df["delay_hours"] / df["expected_time_hours"]) * 100, 2),
    None
)

df["cost_per_km"] = np.where(
    df["distance_km"] > 0,
    round(df["delivery_cost"] / df["distance_km"], 2),
    None
)

def rating_category(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 3.5:
        return "Good"
    elif rating >= 2.5:
        return "Average"
    else:
        return "Poor"

df["rating_category"] = df["delivery_rating"].apply(rating_category)

df["speed_kmph"] = np.where(
    df["delivery_time_hours"] > 0,
    round(df["distance_km"] / df["delivery_time_hours"], 2),
    None
)

df["weight_category"] = pd.cut(
    df["package_weight_kg"],
    bins=[0, 5, 15, float("inf")],
    labels=["Light", "Medium", "Heavy"]
)

# 10. Analysis Summary
total_deliveries = len(df)
delayed_deliveries = (df["delay_status"] == "Delayed").sum()
on_time_deliveries = (df["delay_status"] == "On Time").sum()
delayed_percentage = round((delayed_deliveries / total_deliveries) * 100, 2)
on_time_percentage = round((on_time_deliveries / total_deliveries) * 100, 2)
average_delay = round(df.loc[df["delay_hours"] > 0, "delay_hours"].mean(), 2)

print("\n--- Delivery Analysis Summary ---")
print("Total Deliveries:", total_deliveries)
print("Delayed Deliveries:", delayed_deliveries)
print("On-Time Deliveries:", on_time_deliveries)
print("Delayed %:", delayed_percentage)
print("On-Time %:", on_time_percentage)
print("Average Delay Hours:", average_delay)

# 11. Save final cleaned file
df.to_csv("Delivery_Logistics_Analyzed.csv", index=False)

print("\nAnalysis completed successfully!")
print("New file created: Delivery_Logistics_Analyzed.csv")
print("Total columns:", len(df.columns))

inf_check = 0
for c in df.select_dtypes(include="float64").columns:
    inf_check += np.isinf(df[c]).sum()
print("Remaining infinity values:", inf_check)