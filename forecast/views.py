from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

API_KEY = '2c36d0d3a48ba76dce735f75d12de29a'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Traveler-centric constants
TRAVEL_ACTIVITIES = {
    'rain': ['Museums', 'Art galleries', 'Indoor markets', 'Theaters', 'Cafés'],
    'clear': ['Hiking', 'Beaches', 'Outdoor tours', 'Parks', 'Photography spots'],
    'clouds': ['City walks', 'Zoos', 'Botanical gardens', 'Outdoor cafes'],
    'snow': ['Skiing', 'Hot springs', 'Winter festivals', 'Ice skating'],
    'extreme': ['Stay indoors', 'Emergency shelters', 'Shopping malls']
}

PACKING_ITEMS = {
    'rain': ['Umbrella', 'Waterproof jacket', 'Waterproof shoes', 'Quick-dry clothes'],
    'sun': ['Sunscreen', 'Sunglasses', 'Hat', 'Light clothing'],
    'cold': ['Thermal wear', 'Gloves', 'Scarf', 'Warm jacket'],
    'wind': ['Windbreaker', 'Hair ties', 'Sturdy shoes'],
    'extreme': ['Emergency kit', 'Portable charger', 'First aid supplies']
}

def get_current_weather(city):
    """Get current weather data for a city"""
    url = f'{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        return {
            'current_temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'pressure_data': data['main']['pressure'],
            'weather': data.get('weather', [{'main': 'Clear', 'description': 'clear sky'}]),
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_gust_speed': data.get('wind', {}).get('gust', 0),
            'wind_gust_dir': data.get('wind', {}).get('deg', 0),
            'country': data['sys']['country'],
            'visibility': data.get('visibility', 10000) / 1000  # Convert to km
        }
    except Exception as e:
        print(f"Error fetching weather data for {city}: {str(e)}")
        return None

def get_weather_forecast(city):
    """Get 5-hour forecast for a city"""
    url = f'{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric&cnt=5'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['list']
    except Exception as e:
        print(f"Error fetching forecast for {city}: {str(e)}")
        return None

def read_historical_data(filename):
    """Read and clean historical data"""
    try:
        df = pd.read_csv(filename)
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"Error reading historical data: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe if error occurs

def prepare_data(data):
    """Prepare data for rain prediction"""
    le = LabelEncoder()
    if 'WindGustDir' in data.columns:
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    if 'RainTomorrow' in data.columns:
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    
    features = ['MinTemp', 'MaxTemp', 'Humidity', 'Pressure', 'Temp']
    if 'WindGustDir' in data.columns and 'WindGustSpeed' in data.columns:
        features.extend(['WindGustDir', 'WindGustSpeed'])
    
    X = data[features]
    y = data['RainTomorrow'] if 'RainTomorrow' in data.columns else np.zeros(len(data))
    return X, y, le

def train_rain_model(X, y):
    """Train rain prediction model"""
    if len(X) == 0:
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Rain prediction accuracy: {model.score(X_test, y_test):.2f}")
    return model

def get_wind_direction(degrees):
    """Convert wind degrees to compass direction"""
    if degrees is None:
        return "N/A"
    
    degrees = degrees % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.75), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    return next((point for point, start, end in compass_points if start <= degrees < end), "N")

def get_travel_recommendations(weather_condition):
    """Return activity recommendations based on weather"""
    condition = weather_condition.lower()
    if 'rain' in condition:
        return TRAVEL_ACTIVITIES['rain']
    elif 'clear' in condition:
        return TRAVEL_ACTIVITIES['clear']
    elif 'cloud' in condition:
        return TRAVEL_ACTIVITIES['clouds']
    elif 'snow' in condition:
        return TRAVEL_ACTIVITIES['snow']
    else:
        return TRAVEL_ACTIVITIES['clear']

def get_packing_list(weather_data, days):
    """Generate packing list based on forecast"""
    packing_list = set()
    
    # Check for extreme conditions first
    if weather_data['wind_speed'] > 30:  # High wind
        packing_list.update(PACKING_ITEMS['wind'])
    if weather_data['current_temp'] > 35:  # Extreme heat
        packing_list.update(PACKING_ITEMS['sun'] + PACKING_ITEMS['extreme'])
    elif weather_data['current_temp'] < 5:  # Extreme cold
        packing_list.update(PACKING_ITEMS['cold'] + PACKING_ITEMS['extreme'])
    
    # Regular conditions
    weather_desc = weather_data['weather'][0]['description'].lower()
    if 'rain' in weather_desc:
        packing_list.update(PACKING_ITEMS['rain'])
    elif 'clear' in weather_desc:
        packing_list.update(PACKING_ITEMS['sun'])
    
    # Add universal items
    packing_list.update(['Travel documents', 'Medications', 'Toiletries'])
    
    return sorted(packing_list)

def get_emergency_contacts(location):
    """Fetch emergency contacts for location"""
    contacts = {
        'general': {
            'Police': '100',
            'Ambulance': '102',
            'Fire': '101'
        },
        'local': {
            'Tourist Police': '1363',
            'Local Hospital': '104'
        }
    }
    return contacts

def get_severity_level(weather_data):
    """Determine severity level for alerts"""
    if (weather_data['wind_speed'] > 30 or 
        weather_data['current_temp'] > 35 or 
        weather_data['current_temp'] < 5):
        return 'high'
    elif ('rain' in weather_data['weather'][0]['description'].lower() or
          'storm' in weather_data['weather'][0]['description'].lower()):
        return 'medium'
    return 'low'

@csrf_exempt
def update_packing(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            city = data.get('city')
            days = int(data.get('days', 5))
            
            current_weather = get_current_weather(city)
            if current_weather is None:
                return JsonResponse({'error': 'Unable to fetch weather data'}, status=400)
            
            packing_list = get_packing_list(current_weather, days)
            
            return JsonResponse({
                'packing_list': packing_list
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)
        
        if current_weather is None:
            return render(request, 'weather.html', {'error': 'Unable to fetch weather data for the specified city.'})

        # Get forecast data
        forecast_data = get_weather_forecast(city)
        
        # Prepare forecast information
        forecast_times = []
        forecast_temps = []
        forecast_humidity = []
        
        if forecast_data:
            timezone = pytz.timezone('Asia/Kolkata')
            for forecast in forecast_data:
                forecast_time = datetime.fromtimestamp(forecast['dt'], timezone)
                forecast_times.append(forecast_time.strftime("%H:%M"))
                forecast_temps.append(round(forecast['main']['temp'], 1))
                forecast_humidity.append(round(forecast['main']['humidity'], 1))
        else:
            # Fallback to current weather if forecast fails
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            forecast_times = [(now + timedelta(hours=i)).strftime("%H:%M") for i in range(1, 6)]
            forecast_temps = [round(current_weather['current_temp'], 1)] * 5
            forecast_humidity = [round(current_weather['humidity'], 1)] * 5

        # Rain prediction (if historical data available)
        csv_path = os.path.join('C:\\Users\\HP\\OneDrive\\Desktop\\Weather\\weather.csv')
        historical_data = read_historical_data(csv_path)
        rain_model = None
        
        if not historical_data.empty:
            X, y, le = prepare_data(historical_data)
            rain_model = train_rain_model(X, y)

        # Wind direction
        wind_direction = get_wind_direction(current_weather['wind_gust_dir'])

        # Traveler features
        travel_recommendations = get_travel_recommendations(current_weather['weather'][0]['main'])
        packing_list = get_packing_list(current_weather, 5)
        emergency_contacts = get_emergency_contacts(city)
        severity_level = get_severity_level(current_weather)

        # Prepare context
        context = {
            'location': city,
            'current_temp': round(current_weather['current_temp'], 1),
            'MinTemp': round(current_weather['temp_min'], 1),
            'MaxTemp': round(current_weather['temp_max'], 1),
            'feels_like': round(current_weather['feels_like'], 1),
            'humidity': current_weather['humidity'],
            'weather': current_weather['weather'][0]['description'],
            'country': current_weather.get('country', ''),
            'wind': round(current_weather['wind_speed'], 1),
            'wind_direction': wind_direction,
            'pressure': current_weather['pressure_data'],
            'visibility': round(current_weather['visibility'], 1),
            'description': current_weather['weather'][0]['main'].lower(),
            
            # Forecast data
            'time1': forecast_times[0] if len(forecast_times) > 0 else 'N/A',
            'time2': forecast_times[1] if len(forecast_times) > 1 else 'N/A',
            'time3': forecast_times[2] if len(forecast_times) > 2 else 'N/A',
            'time4': forecast_times[3] if len(forecast_times) > 3 else 'N/A',
            'time5': forecast_times[4] if len(forecast_times) > 4 else 'N/A',
            
            'temp1': forecast_temps[0] if len(forecast_temps) > 0 else 'N/A',
            'temp2': forecast_temps[1] if len(forecast_temps) > 1 else 'N/A',
            'temp3': forecast_temps[2] if len(forecast_temps) > 2 else 'N/A',
            'temp4': forecast_temps[3] if len(forecast_temps) > 3 else 'N/A',
            'temp5': forecast_temps[4] if len(forecast_temps) > 4 else 'N/A',
            
            'hum1': forecast_humidity[0] if len(forecast_humidity) > 0 else 'N/A',
            'hum2': forecast_humidity[1] if len(forecast_humidity) > 1 else 'N/A',
            'hum3': forecast_humidity[2] if len(forecast_humidity) > 2 else 'N/A',
            'hum4': forecast_humidity[3] if len(forecast_humidity) > 3 else 'N/A',
            'hum5': forecast_humidity[4] if len(forecast_humidity) > 4 else 'N/A',
            
            # Traveler features
            'travel_recommendations': travel_recommendations,
            'packing_list': packing_list,
            'emergency_contacts': emergency_contacts,
            'severity_level': severity_level,
            'is_extreme_weather': severity_level == 'high'
        }

        return render(request, 'weather.html', context)
    
    return render(request, 'weather.html')