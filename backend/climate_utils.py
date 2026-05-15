import requests
import os

# =========================================================
# OFFLINE CLIMATE INFERENCE
# =========================================================

def get_offline_climate(lat, lon):
    """
    Infers climate zone based on latitude and longitude without internet.
    Optimized for Indian and Tropical regions.
    """
    if lat is None or lon is None:
        return "Unknown"

    # Absolute latitude for hemisphere-agnostic calculation
    abs_lat = abs(float(lat))
    lon = float(lon)

    # 1. Tropical Zone (0 to 23.5 degrees)
    if abs_lat <= 23.5:
        # Refine for India (approx 68 to 97 degrees East)
        if 68 <= lon <= 97:
            if abs_lat < 20:
                return "Tropical (Humid)"  # South India
            else:
                return "Tropical (Monsoon-prone)" # Central/Coastal India
        return "Tropical"

    # 2. Subtropical / Dry Zone (23.5 to 35 degrees)
    elif 23.5 < abs_lat <= 35:
        # Often dry or humid subtropical (e.g. North India, Middle East)
        if 60 <= lon <= 80:
            return "Dry / Arid"
        return "Humid Subtropical"

    # 3. Temperate Zone (35 to 66.5 degrees)
    elif 35 < abs_lat <= 66.5:
        return "Temperate"

    # 4. Polar Zone (66.5+ degrees)
    else:
        return "Polar / Cold"

# =========================================================
# OPTIONAL ONLINE WEATHER ENHANCEMENT
# =========================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") # Placeholder

def get_online_weather(lat, lon):
    """
    Fetches real-time weather from OpenWeatherMap if internet is available.
    """
    if not lat or not lon or not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        return None



    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rain": data.get("rain", {}).get("1h", 0),
                "clouds": data["clouds"]["all"],
                "description": data["weather"][0]["description"]
            }
        else:
            print(f"Weather API Status Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Weather API Error: {e}")
        # Silent fail for offline-first resilience
        return None

    
    return None

def get_climate_context(lat, lon):
    """
    Hybrid function: Always returns offline inference, 
    optionally enhances with online data.
    """
    climate_zone = get_offline_climate(lat, lon)
    weather = get_online_weather(lat, lon)
    
    return {
        "climate_zone": climate_zone,
        "weather": weather,
        "is_online": weather is not None
    }
