from flask import jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
import math
import os
import requests


# ── Load model artifacts ───────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
model      = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
target_enc = joblib.load(os.path.join(MODEL_DIR, 'target_label_encoder.pkl'))
feat_enc   = joblib.load(os.path.join(MODEL_DIR, 'feature_encoders.pkl'))
feat_cols  = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

# ── AQI category metadata ──────────────────────────────────────────────────
AQI_META = {
    'Good': {
        'color': '#2ecc71', 'icon': '😊',
        'health': 'Air quality is satisfactory. Enjoy outdoor activities freely.',
        'advice': ['All outdoor activities are safe', 'No precautions needed', 'Great day for a run or walk']
    },
    'Moderate': {
        'color': '#f1c40f', 'icon': '😐',
        'health': 'Acceptable air quality. Sensitive individuals may experience minor effects.',
        'advice': ['Outdoor activities generally fine', 'Sensitive groups may reduce prolonged outdoor exertion', 'Keep windows open for ventilation']
    },
    'Unhealthy for Sensitive Groups': {
        'color': '#e67e22', 'icon': '😷',
        'health': 'Sensitive groups (children, elderly, asthma patients) may experience health effects.',
        'advice': ['Sensitive groups should limit outdoor exertion', 'Keep medication handy if asthmatic', 'Avoid strenuous outdoor activity for 2+ hours']
    },
    'Unhealthy': {
        'color': '#e74c3c', 'icon': '🤧',
        'health': 'Everyone may begin to experience health effects. Limit prolonged outdoor exposure.',
        'advice': ['Wear an N95 mask outdoors', 'Reduce time spent outside', 'Keep windows closed and use air purifier indoors']
    },
    'Very Unhealthy': {
        'color': '#8e44ad', 'icon': '😨',
        'health': 'Health alert — everyone may experience serious health effects.',
        'advice': ['Avoid outdoor activities entirely', 'Wear N95 mask if going out is unavoidable', 'Run air purifier indoors at all times', 'Stay hydrated']
    },
    'Hazardous': {
        'color': '#2c3e50', 'icon': '☠️',
        'health': 'Health emergency. Everyone is likely to be affected seriously.',
        'advice': ['Do NOT go outside', 'Seal windows and doors', 'Wear N95 mask even indoors if possible', 'Seek medical attention if experiencing symptoms']
    }
}


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Get coordinates from city + state (Open-Meteo, no API key)
# ─────────────────────────────────────────────────────────────────────────
def get_coordinates(city, state):
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=5&language=en&format=json"
    )
    try:
        res     = requests.get(url, timeout=10).json()
        results = res.get('results', [])
        if not results:
            return None, None

        # Try to match state name for accuracy
        for r in results:
            admin = r.get('admin1', '').lower()
            if state.lower() in admin or admin in state.lower():
                return r['latitude'], r['longitude']

        # Fallback to first result if no state match found
        return results[0]['latitude'], results[0]['longitude']

    except Exception as e:
        print(f"[AQI] Geocoding error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Fetch weather data (Open-Meteo, no API key)
# ─────────────────────────────────────────────────────────────────────────
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"wind_direction_10m,surface_pressure,cloud_cover,precipitation"
        f"&timezone=Asia%2FKolkata"
    )
    try:
        res = requests.get(url, timeout=10).json()
        c   = res['current']
        return {
            'temperature':   c['temperature_2m'],
            'humidity':      c['relative_humidity_2m'],
            'pressure':      c['surface_pressure'],
            'wind_speed':    c['wind_speed_10m'],       # already in km/h
            'wind_dir':      c['wind_direction_10m'],
            'cloud_cover':   c['cloud_cover'],
            'precipitation': c['precipitation'],
        }
    except Exception as e:
        print(f"[AQI] Weather fetch error: {e}")
        return {
            'temperature': 25.0, 'humidity': 50.0, 'pressure': 1013.0,
            'wind_speed': 10.0,  'wind_dir': 180.0, 'cloud_cover': 20.0,
            'precipitation': 0.0
        }


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — Fetch air quality data (Open-Meteo, no API key)
#   Works for past dates, today, and future forecast
# ─────────────────────────────────────────────────────────────────────────
def fetch_air_quality(lat, lon, input_date):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,"
        f"carbon_monoxide,ammonia"
        f"&start_date={input_date}&end_date={input_date}"
        f"&timezone=Asia%2FKolkata"
    )
    try:
        res = requests.get(url, timeout=10).json()
        h   = res['hourly']

        # Use current hour for today, midday (12) for other dates
        idx = datetime.now().hour if input_date == date.today().isoformat() else 12

        def safe(lst, i, fallback=0.0):
            try:
                v = lst[i]
                return float(v) if v is not None else fallback
            except Exception:
                return fallback

        return {
            'pm25': safe(h['pm2_5'],             idx),
            'pm10': safe(h['pm10'],              idx),
            'no2':  safe(h['nitrogen_dioxide'],  idx),
            'o3':   safe(h['ozone'],             idx),
            'so2':  safe(h['sulphur_dioxide'],   idx),
            'co':   safe(h['carbon_monoxide'],   idx),
            'nh3':  safe(h.get('ammonia', []),   idx, 2.0),
        }
    except Exception as e:
        print(f"[AQI] Air quality fetch error: {e}")
        return {
            'pm25': 30.0, 'pm10': 50.0, 'no2': 20.0,
            'o3':   40.0, 'so2':  10.0, 'co': 500.0, 'nh3': 2.0
        }


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — Derive all engineered features (original, unchanged)
# ─────────────────────────────────────────────────────────────────────────
def derive_features(data):
    """Auto-derive all features that don't need user input."""
    hour = int(data.get('hour', 0))
    dt = datetime.strptime(f"{data['date']} {hour:02d}:00", "%Y-%m-%d %H:%M")

    hour       = dt.hour
    month      = dt.month
    day        = dt.day
    year       = dt.year
    weekday    = dt.weekday()
    week       = dt.isocalendar()[1]
    is_weekend = int(weekday >= 5)

    day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_name  = day_names[weekday]
    quarter   = (month - 1) // 3 + 1

    if month in [12, 1, 2]:      season = 'Winter'
    elif month in [3, 4, 5]:     season = 'Spring'
    elif month in [6, 7, 8, 9]:  season = 'Monsoon'
    else:                         season = 'Post_Monsoon'

    if   5 <= hour < 12:  time_of_day = 'Morning'
    elif 12 <= hour < 17: time_of_day = 'Afternoon'
    elif 17 <= hour < 20: time_of_day = 'Evening'
    elif 20 <= hour < 24: time_of_day = 'Night'
    else:                  time_of_day = 'Night_Late'

    is_daytime = int(6 <= hour < 19)

    humidity = float(data['humidity'])
    if   humidity < 30:  hum_cat = 'Low'
    elif humidity < 60:  hum_cat = 'Moderate'
    elif humidity < 80:  hum_cat = 'High'
    else:                hum_cat = 'Very_High'

    wind = float(data['wind_speed'])
    if   wind < 5:   wind_cat = 'Calm'
    elif wind < 15:  wind_cat = 'Light'
    elif wind < 30:  wind_cat = 'Moderate'
    elif wind < 50:  wind_cat = 'Strong'
    else:             wind_cat = 'Storm'

    wind_stagnation = int(wind < 5)
    temp      = float(data['temperature'])
    dew_point = temp - ((100 - humidity) / 5)

    pm25     = float(data['pm25'])
    pm10     = float(data['pm10'])
    pm_ratio = round(pm25 / pm10, 3) if pm10 > 0 else 0.5

    solar    = 600.0  if is_daytime else 0.0
    uv       = 5.0    if is_daytime else 0.0
    sunshine = 3600.0 if is_daytime else 0.0

    row = {
        'City':                  data['city'],
        'State':                 data['state'],
        'Latitude':              float(data.get('latitude',  20.0)),
        'Longitude':             float(data.get('longitude', 78.0)),
        'Year':                  year,
        'Month':                 month,
        'Day':                   day,
        'Hour':                  hour,
        'Day_of_Week':           weekday,
        'Day_Name':              day_name,
        'Week_of_Year':          week,
        'Is_Weekend':            is_weekend,
        'Quarter':               quarter,
        'Season':                season,
        'Time_of_Day':           time_of_day,
        'Temp_2m_C':             temp,
        'Temp_80m_C':            temp + 1.0,
        'Temp_120m_C':           temp + 1.5,
        'Temp_180m_C':           temp + 2.0,
        'Humidity_Percent':      humidity,
        'Dew_Point_C':           round(dew_point, 2),
        'Humidity_Category':     hum_cat,
        'Wind_Speed_10m_kmh':    wind,
        'Wind_Speed_80m_kmh':    wind * 1.2,
        'Wind_Speed_120m_kmh':   wind * 1.3,
        'Wind_Dir_10m':          float(data.get('wind_dir', 180.0)),
        'Wind_Gusts_kmh':        wind * 1.5,
        'Wind_Category':         wind_cat,
        'Wind_Stagnation':       wind_stagnation,
        'Precipitation_mm':      float(data.get('precipitation', 0.0)),
        'Rain_mm':               float(data.get('precipitation', 0.0)),
        'Is_Raining':            int(float(data.get('precipitation', 0.0)) > 0),
        'Heavy_Rain':            int(float(data.get('precipitation', 0.0)) > 10),
        'Pressure_MSL_hPa':      float(data.get('pressure', 1013.0)),
        'Surface_Pressure_hPa':  float(data.get('pressure', 1013.0)) - 2.0,
        'Solar_Radiation_Wm2':   solar,
        'Direct_Radiation_Wm2':  solar * 0.75,
        'Diffuse_Radiation_Wm2': solar * 0.25,
        'UV_Index':              uv,
        'Cloud_Cover_Percent':   float(data.get('cloud_cover', 20.0)),
        'Cloud_Low_Percent':     float(data.get('cloud_cover', 20.0)) * 0.3,
        'Cloud_Mid_Percent':     float(data.get('cloud_cover', 20.0)) * 0.4,
        'Cloud_High_Percent':    float(data.get('cloud_cover', 20.0)) * 0.3,
        'Is_Daytime':            is_daytime,
        'Sunshine_Seconds':      sunshine,
        'PM2_5_ugm3':            pm25,
        'PM10_ugm3':             pm10,
        'PM_Ratio':              pm_ratio,
        'CO_ugm3':               float(data['co']),
        'NO2_ugm3':              float(data['no2']),
        'SO2_ugm3':              float(data['so2']),
        'O3_ugm3':               float(data['o3']),
        'Dust_ugm3':             float(data.get('dust', 5.0)),
        'NH3_ugm3':              float(data.get('nh3', 2.0)),
        'AOD':                   float(data.get('aod', 0.3)),
        'Temp_Inversion':        int(float(data.get('temp_inversion', 0))),
        'Inversion_Strength_C':  float(data.get('inversion_strength', 0.0)),
        'Festival_Period':       int(data.get('festival_period', 0)),
        'Crop_Burning_Season':   int(data.get('crop_burning', 0)),
        'Hour_sin':              math.sin(2 * math.pi * hour / 24),
        'Hour_cos':              math.cos(2 * math.pi * hour / 24),
        'Month_sin':             math.sin(2 * math.pi * month / 12),
        'Month_cos':             math.cos(2 * math.pi * month / 12),
    }
    return row
    

# ─────────────────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────────────────
def predict_aqi(request):
    try:
        data = request.json

        city       = data.get('city', '').strip()
        state      = data.get('state', '').strip()
        input_date = data.get('date', date.today().isoformat())
        hour       = data.get('hour', datetime.now().hour)

        if not city or not state:
            return jsonify({'success': False, 'error': 'City and State are required.'}), 400

        # Step 1 — coordinates
        lat, lon = get_coordinates(city, state)
        if lat is None:
            return jsonify({'success': False, 'error': f'Location not found: {city}, {state}'}), 400

        print(f"[AQI] {city}, {state} → lat={lat}, lon={lon}")

        # Step 2 — weather
        weather = fetch_weather(lat, lon)
        print(f"[AQI] Weather: {weather}")

        # Step 3 — air quality
        air = fetch_air_quality(lat, lon, input_date)
        print(f"[AQI] Air quality: {air}")

        print("\n===== DEBUG START =====")
        print("CITY:", city, "| STATE:", state)
        print("WEATHER DATA:", weather)
        print("AIR QUALITY DATA:", air)
        print("===== DEBUG END =====\n")

        # Step 4 — merge into single data dict
        combined = {
            'city':          city,
            'state':         state,
            'date':          input_date,
            'hour':          hour,
            'latitude':      lat,
            'longitude':     lon,
            # weather
            'temperature':   weather['temperature'],
            'humidity':      weather['humidity'],
            'pressure':      weather['pressure'],
            'wind_speed':    weather['wind_speed'],
            'wind_dir':      weather['wind_dir'],
            'cloud_cover':   weather['cloud_cover'],
            'precipitation': weather['precipitation'],
            # air quality
            'pm25': air['pm25'],
            'pm10': air['pm10'],
            'no2':  air['no2'],
            'o3':   air['o3'],
            'so2':  air['so2'],
            'co':   air['co'],
            'nh3':  air['nh3'],
        }

        # Step 5 — derive features
        row = derive_features(combined)
        df  = pd.DataFrame([row])

        print("DF BEFORE MODEL:\n", df.head())
        print("NaN values:\n", df.isnull().sum())

        # Step 6 — encode categorical columns
        for col, enc in feat_enc.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except:
                    df[col] = 0

        # Step 7 — align to training feature columns
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feat_cols]

        # Step 8 — predict
        pred_enc   = model.predict(df)[0]
        pred_proba = model.predict_proba(df)[0]
        pred_label = target_enc.inverse_transform([pred_enc])[0]
        classes    = target_enc.classes_

        probabilities = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(classes, pred_proba)
        }

        meta = AQI_META.get(pred_label, AQI_META['Moderate'])

        return jsonify({
            'success':       True,
            'city':          city,
            'state':         state,
            'date':          input_date,
            'category':      pred_label.replace("_"," "),
            'color':         meta['color'],
            'icon':          meta['icon'],
            'health':        meta['health'],
            'advice':        meta['advice'],
            'probabilities': probabilities,
            'confidence':    round(float(max(pred_proba)) * 100, 1),
            # return fetched values so UI can display what was used
            'fetched': {
                'temperature': round(weather['temperature'], 1),
                'humidity':    round(weather['humidity'], 1),
                'wind_speed':  round(weather['wind_speed'], 1),
                'pressure':    round(weather['pressure'], 1),
                'pm25':        round(air['pm25'], 1),
                'pm10':        round(air['pm10'], 1),
                'no2':         round(air['no2'],  1),
                'o3':          round(air['o3'],   1),
                'so2':         round(air['so2'],  1),
                'co':          round(air['co'],   1),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500