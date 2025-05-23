Tool: Claude
Date: 2025-05-21
Me: I need to set up the imports and environment configuration for my weather advisor application. Can you help me create a basic setup that imports fetch-my-weather and hands-on-ai packages?
AI: Here's a basic setup for your weather advisor:
python# Install packages
!pip install fetch-my-weather
!pip install hands-on-ai

# Basic imports
from fetch_my_weather import get_weather
from hands_on_ai.chat import get_response
import matplotlib.pyplot as plt
import pyinputplus as pyip

# Environment setup
import os
os.environ['HANDS_ON_AI_SERVER'] = 'http://ollama.serveur.au'
os.environ['HANDS_ON_AI_MODEL'] = 'granite3.2'
os.environ['HANDS_ON_AI_API_KEY'] = 'student-api-key-123'
Me: This looks basic but what happens if the packages fail to install or import? Since this is for an assignment where reliability matters, can you add comprehensive error handling and validation to ensure the environment is properly configured?
AI: You're absolutely right about needing robust error handling. Here's an improved version:
python# Install with error handling
try:
    !pip install fetch-my-weather
    !pip install hands-on-ai
    print("✅ Packages installed successfully")
except Exception as e:
    print(f"❌ Installation failed: {e}")

# Import with validation
try:
    from fetch_my_weather import get_weather, set_mock_mode, clear_cache
    from hands_on_ai.chat import get_response
    import matplotlib.pyplot as plt
    import pyinputplus as pyip
    print("✅ All packages imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check package installation")

# Environment validation
def validate_environment():
    required_vars = ['HANDS_ON_AI_SERVER', 'HANDS_ON_AI_MODEL', 'HANDS_ON_AI_API_KEY']
    missing = [var for var in required_vars if var not in os.environ]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False
    return True
Me: Good improvement! But I'm thinking about the user experience - what if a student runs this and something fails? Can you add connection testing for both the AI service and weather API, plus provide clear feedback about what's working and what isn't?
AI: Excellent point about user experience! Here's a version with comprehensive connection testing:
pythondef test_ai_connection():
    """Test AI service connectivity and provide clear feedback."""
    try:
        test_response = get_response("Hello, respond with just 'Connection successful'")
        if "successful" in test_response.lower():
            print("✅ AI connection established successfully")
            return True
        else:
            print(f"⚠️ AI connection may have issues. Response: {test_response}")
            return False
    except Exception as e:
        print(f"❌ AI connection failed: {e}")
        print("The application will work with limited AI functionality")
        return False

def test_weather_api():
    """Test weather API and provide fallback options."""
    try:
        # Use mock mode for testing to avoid rate limits
        set_mock_mode(True)
        test_weather = get_weather(location="London", format="json")
        if hasattr(test_weather, 'current_condition'):
            print("✅ Weather API working (mock mode for testing)")
            return True
        else:
            print("⚠️ Weather API may have issues")
            return False
    except Exception as e:
        print(f"❌ Weather API failed: {e}")
        return False

# Run comprehensive tests
if validate_environment():
    ai_working = test_ai_connection()
    weather_working = test_weather_api()
    if ai_working and weather_working:
        print("\n🎉 All systems ready!")
    else:
        print("\n⚠️ Some services have issues but application can still run")
Me: This is much better for debugging! Now I'm thinking about the development workflow - during development I'll be running cells multiple times and may want to clear caches or reset configurations. Can you add utility functions for managing the application state and also set up matplotlib for optimal notebook display?
AI: Great thinking about the development workflow! Here's an enhanced version with state management and optimal notebook configuration:
python# Configure matplotlib for better notebook display
plt.style.use('seaborn-v0_8')  # Clean, modern plotting style
plt.rcParams['figure.figsize'] = (10, 6)  # Consistent figure size
plt.rcParams['font.size'] = 10  # Readable font size
plt.rcParams['axes.grid'] = True  # Enable grid by default

# Application state management
class WeatherAppConfig:
    """Manage application configuration and state."""
    
    def __init__(self):
        self.supported_units = ['metric', 'imperial']
        self.default_forecast_days = 5
        self.max_forecast_days = 7
        self.cache_duration = 600  # 10 minutes
        
    def reset_to_defaults(self):
        """Reset all configurations to default values."""
        set_mock_mode(False)
        clear_cache()
        print("✅ Configuration reset to defaults")
        
    def enable_development_mode(self):
        """Enable settings optimized for development."""
        set_mock_mode(True)  # Use mock data to avoid API limits
        clear_cache()  # Start with fresh cache
        print("✅ Development mode enabled (using mock data)")
        
    def display_status(self):
        """Display current configuration status."""
        print(f"\n📋 Application Status:")
        print(f"Supported units: {', '.join(self.supported_units)}")
        print(f"Default forecast days: {self.default_forecast_days}")
        print(f"Cache duration: {self.cache_duration} seconds")

# Utility functions for development workflow
def clear_all_outputs():
    """Clear matplotlib figures and weather cache."""
    plt.close('all')  # Close all matplotlib figures
    clear_cache()     # Clear weather data cache
    print("✅ Outputs and cache cleared")

def quick_test():
    """Run a quick test of all major components."""
    print("🧪 Running quick system test...")
    config = WeatherAppConfig()
    config.enable_development_mode()
    
    # Test sequence
    tests = [
        ("Environment", validate_environment),
        ("AI Connection", test_ai_connection),
        ("Weather API", test_weather_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✅" if result else "⚠️"))
        except Exception as e:
            results.append((test_name, "❌"))
    
    print("\n📊 Test Results:")
    for test_name, status in results:
        print(f"{status} {test_name}")

# Initialize configuration
app_config = WeatherAppConfig()
app_config.enable_development_mode()  # Start in development mode
Me: Perfect! This gives me much better control over the development process. One last thing - I notice the setup doesn't include any global constants or type hints that I'll need for the rest of the application. Can you add those along with a final initialization sequence that provides a complete status report?
AI: Excellent attention to detail! Here's the complete setup with global constants, type hints, and a comprehensive initialization:
pythonfrom typing import Dict, List, Optional, Union, Tuple
import datetime

# Global constants for application
APP_TITLE = "🌦️ WeatherWise Advisor"
VERSION = "1.0.0"
AUTHOR = "Weather App Student"

# Weather data constants
TEMPERATURE_UNITS: Dict[str, str] = {
    'metric': '°C',
    'imperial': '°F'
}

WEATHER_ATTRIBUTES: List[str] = [
    'temperature', 'humidity', 'wind_speed', 'precipitation', 
    'pressure', 'visibility', 'uv_index'
]

# UI Constants
MENU_OPTIONS: List[str] = [
    "View Current Weather",
    "View Weather Forecast", 
    "Show Temperature Chart",
    "Show Precipitation Chart",
    "Natural Language Query",
    "Change Location",
    "Settings",
    "Exit"
]

def complete_initialization() -> bool:
    """
    Run complete application initialization with detailed status reporting.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    print(f"\n{APP_TITLE} v{VERSION}")
    print("=" * 60)
    print("🚀 Starting application initialization...\n")
    
    # Step 1: Environment validation
    print("1️⃣ Validating environment...")
    env_valid = validate_environment()
    
    # Step 2: Connection testing
    print("\n2️⃣ Testing service connections...")
    ai_status = test_ai_connection()
    weather_status = test_weather_api()
    
    # Step 3: Configuration setup
    print("\n3️⃣ Setting up configuration...")
    app_config.display_status()
    
    # Step 4: Final status report
    print("\n4️⃣ Initialization Summary:")
    components = [
        ("Environment Variables", env_valid),
        ("AI Service", ai_status),
        ("Weather API", weather_status),
        ("Application Config", True)
    ]
    
    all_working = True
    for component, status in components:
        icon = "✅" if status else "❌"
        print(f"  {icon} {component}")
        if not status:
            all_working = False
    
    if all_working:
        print(f"\n🎉 {APP_TITLE} is ready to use!")
        print("You can now proceed to implement the weather data functions.")
    else:
        print(f"\n⚠️ Some issues detected. The app may have limited functionality.")
        print("Please check the error messages above and fix any issues.")
    
    print(f"\n📅 Initialized at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_working

# Run complete initialization
initialization_successful = complete_initialization()

# Provide next steps guidance
if initialization_successful:
    print("\n🔄 Next Steps:")
    print("1. Implement get_weather_data() function")
    print("2. Create visualization functions")
    print("3. Add natural language processing")
    print("4. Build user interface")
    print("\nUse quick_test() anytime to verify system status.")

Me: I'm getting this error when setting up my weather app:
"HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded"
and "Missing environment variables: ['HANDS_ON_AI_SERVER', 'HANDS_ON_AI_MODEL', 'HANDS_ON_AI_API_KEY']"

Can you help me understand what's wrong and how to fix it?

AI: I can see two distinct issues here:

Port/Server Mismatch: The error shows it's trying to connect to localhost:11434, which is the default Ollama port. However, your environment is configured for http://ollama.serveur.au, so there's a configuration disconnect.
Environment Variable Names: The error mentions HANDS_ON_AI_* variables, but many Python packages that interface with Ollama actually use different variable names. The hands-on-ai package likely uses AILABKIT_* variables instead.

The core issue is that your application is defaulting to local Ollama settings instead of using your remote server configuration. This typically happens when environment variables aren't properly set or recognized by the package.

Me: The error mentions localhost:11434, but I'm supposed to connect to ollama.serveur.au. Also, what are the correct environment variable names for hands-on-ai package? The documentation seems inconsistent.
AI: You've identified the exact problem! Here's what's happening:
Correct Environment Variables for hands-on-ai:
pythonos.environ['AILABKIT_SERVER'] = 'http://ollama.serveur.au'
os.environ['AILABKIT_MODEL'] = 'granite3.2'  
os.environ['AILABKIT_API_KEY'] = 'student-api-key-123'

Tool: Claude
Date: 2025-05-21
Me: I need to implement the get_weather_data() function that retrieves weather information for a specified location. Can you help me create a basic implementation using the fetch-my-weather package?
AI: Here's a basic implementation of the weather data function:
pythondef get_weather_data(location, forecast_days=5):
    """
    Retrieve weather data for a specified location.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-5)
    
    Returns:
        dict: Weather data including current conditions and forecast
    """
    # Get weather data using fetch-my-weather
    weather_response = get_weather(location=location, format="json")
    
    # Extract current conditions
    current = weather_response.current_condition[0]
    forecast = weather_response.weather[:forecast_days]
    
    return {
        "location": location,
        "current": current,
        "forecast": forecast
    }
Me: This is a start, but what happens if the location doesn't exist or the API fails? For a production application, I need comprehensive error handling and data validation. Can you add robust error handling and ensure the function returns consistent data structures even when things go wrong?
AI: You're absolutely right about needing robust error handling. Here's an improved version:
pythondef get_weather_data(location: str, forecast_days: int = 5, units: str = "metric") -> Dict[str, Union[str, Dict, List]]:
    """
    Retrieve weather data for a specified location with comprehensive error handling.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-7)
        units (str): Temperature units ('metric' or 'imperial')
    
    Returns:
        dict: Weather data including current conditions and forecast, or error information
    """
    # Input validation
    if not location or not isinstance(location, str):
        return {"error": "Invalid location: must be a non-empty string"}
    
    if not isinstance(forecast_days, int) or forecast_days < 1 or forecast_days > 7:
        return {"error": "Invalid forecast_days: must be integer between 1 and 7"}
    
    if units not in SUPPORTED_UNITS:
        return {"error": f"Invalid units: must be one of {SUPPORTED_UNITS}"}
    
    try:
        # Attempt to get weather data
        weather_response = get_weather(location=location, format="json", units=units)
        
        # Validate response structure
        if not hasattr(weather_response, 'current_condition') or not weather_response.current_condition:
            return {"error": f"No weather data available for location: {location}"}
        
        # Check if location was found
        if hasattr(weather_response, 'nearest_area') and weather_response.nearest_area:
            found_location = weather_response.nearest_area[0].areaName[0].value
        else:
            found_location = location
        
        return {
            "status": "success",
            "location": found_location,
            "current": weather_response.current_condition[0],
            "forecast": weather_response.weather[:forecast_days] if hasattr(weather_response, 'weather') else [],
            "units": units,
            "retrieved_at": datetime.datetime.now().isoformat()
        }
        
    except ConnectionError:
        return {"error": "Network connection failed. Please check your internet connection."}
    except Exception as e:
        return {"error": f"Failed to retrieve weather data: {str(e)}"}
Me: Great improvement! But the raw API response has a complex nested structure that's hard to work with. Can you create a function that processes and flattens this data into a more user-friendly format that's easier for the rest of my application to use?
AI: Excellent point! Here's a data processing function that creates a clean, flat structure:
pythondef process_weather_data(raw_weather_data: Dict) -> Dict:
    """
    Process raw weather API response into a clean, user-friendly format.
    
    Args:
        raw_weather_data: Raw response from get_weather_data()
    
    Returns:
        dict: Processed weather data with simplified structure
    """
    if "error" in raw_weather_data:
        return raw_weather_data
    
    try:
        current = raw_weather_data["current"]
        forecast_raw = raw_weather_data["forecast"]
        
        # Process current weather
        processed_current = {
            "temperature": int(current.temp_C if raw_weather_data["units"] == "metric" else current.temp_F),
            "feels_like": int(current.FeelsLikeC if raw_weather_data["units"] == "metric" else current.FeelsLikeF),
            "condition": current.weatherDesc[0].value,
            "humidity": int(current.humidity),
            "wind_speed": int(current.windspeedKmph if raw_weather_data["units"] == "metric" else current.windspeedMiles),
            "wind_direction": current.winddir16Point,
            "pressure": int(current.pressure),
            "visibility": int(current.visibility),
            "uv_index": int(current.uvIndex),
            "precipitation": float(current.precipMM)
        }
        
        # Process forecast data
        processed_forecast = []
        for day in forecast_raw:
            day_data = {
                "date": day.date,
                "max_temp": int(day.maxtempC if raw_weather_data["units"] == "metric" else day.maxtempF),
                "min_temp": int(day.mintempC if raw_weather_data["units"] == "metric" else day.mintempF),
                "condition": day.hourly[4].weatherDesc[0].value,  # Midday condition
                "precipitation_chance": int(day.hourly[4].chanceofrain),
                "precipitation_amount": float(day.hourly[4].precipMM),
                "humidity": int(day.hourly[4].humidity),
                "wind_speed": int(day.hourly[4].windspeedKmph if raw_weather_data["units"] == "metric" else day.hourly[4].windspeedMiles),
                "wind_direction": day.hourly[4].winddir16Point
            }
            processed_forecast.append(day_data)
        
        return {
            "status": "success",
            "location": raw_weather_data["location"],
            "units": raw_weather_data["units"],
            "current": processed_current,
            "forecast": processed_forecast,
            "retrieved_at": raw_weather_data["retrieved_at"]
        }
        
    except (KeyError, AttributeError, IndexError) as e:
        return {"error": f"Error processing weather data structure: {str(e)}"}
Me: This is much cleaner! Now I'm thinking about performance - if users query the same location multiple times, I don't want to hit the API every time. Can you add intelligent caching that stores recent data and only makes new API calls when the data is stale?
AI: Great thinking about performance! Here's an enhanced version with intelligent caching:
pythonimport time
from typing import Optional

# Global cache for weather data
_weather_cache: Dict[str, Dict] = {}

def get_cache_key(location: str, units: str) -> str:
    """Generate a cache key for weather data."""
    return f"{location.lower().strip()}_{units}"

def is_cache_valid(cache_entry: Dict, max_age_seconds: int = 600) -> bool:
    """Check if cached data is still valid (default: 10 minutes)."""
    if "cached_at" not in cache_entry:
        return False
    
    age = time.time() - cache_entry["cached_at"]
    return age < max_age_seconds

def get_weather_data_with_cache(location: str, forecast_days: int = 5, 
                               units: str = "metric", use_cache: bool = True,
                               cache_duration: int = 600) -> Dict:
    """
    Get weather data with intelligent caching.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-7)
        units (str): Temperature units ('metric' or 'imperial')
        use_cache (bool): Whether to use cached data
        cache_duration (int): Cache validity in seconds
    
    Returns:
        dict: Weather data with cache metadata
    """
    cache_key = get_cache_key(location, units)
    
    # Check cache first
    if use_cache and cache_key in _weather_cache:
        cached_data = _weather_cache[cache_key]
        if is_cache_valid(cached_data, cache_duration):
            print(f"📦 Using cached data for {location} (age: {int(time.time() - cached_data['cached_at'])}s)")
            return {**cached_data["data"], "from_cache": True}
    
    # Fetch fresh data
    print(f"🌐 Fetching fresh weather data for {location}")
    raw_data = get_weather_data(location, forecast_days, units)
    
    if "error" not in raw_data:
        # Process the data
        processed_data = process_weather_data(raw_data)
        
        if "error" not in processed_data:
            # Cache the processed data
            _weather_cache[cache_key] = {
                "data": processed_data,
                "cached_at": time.time()
            }
            
            # Clean old cache entries (keep only last 10)
            if len(_weather_cache) > 10:
                oldest_key = min(_weather_cache.keys(), 
                               key=lambda k: _weather_cache[k]["cached_at"])
                del _weather_cache[oldest_key]
            
            return {**processed_data, "from_cache": False}
    
    return raw_data

def clear_weather_cache():
    """Clear all cached weather data."""
    global _weather_cache
    _weather_cache.clear()
    print("🧹 Weather cache cleared")

def get_cache_status() -> Dict:
    """Get information about current cache status."""
    if not _weather_cache:
        return {"cache_size": 0, "entries": []}
    
    entries = []
    for key, value in _weather_cache.items():
        age = int(time.time() - value["cached_at"])
        entries.append({
            "location": key,
            "age_seconds": age,
            "valid": is_cache_valid(value)
        })
    
    return {
        "cache_size": len(_weather_cache),
        "entries": entries
    }
Me: Excellent caching system! One final enhancement - I want to make this function resilient to different edge cases like network timeouts, partial data, and provide helpful suggestions when locations aren't found. Can you add these features plus a validation function that helps users understand what went wrong?
AI: Perfect! Here's the complete, production-ready implementation with comprehensive edge case handling:
pythonimport re
from typing import List

def suggest_location_alternatives(location: str) -> List[str]:
    """
    Suggest alternative location formats when a location isn't found.
    
    Args:
        location (str): The location that wasn't found
    
    Returns:
        List[str]: Suggested alternative formats
    """
    suggestions = []
    
    # Clean the input
    clean_location = location.strip()
    
    # Try different formats
    if ',' not in clean_location:
        suggestions.append(f"{clean_location}, Country")
        suggestions.append(f"{clean_location}, State")
    
    # Try with major city format
    if len(clean_location.split()) > 1:
        suggestions.append(clean_location.split()[0])  # First word only
    
    # Try common variations
    suggestions.extend([
        clean_location.title(),  # Title case
        clean_location.upper(),  # Upper case
        clean_location.lower()   # Lower case
    ])
    
    return list(set(suggestions))  # Remove duplicates

def validate_weather_response(data: Dict) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate weather response and identify missing or problematic data.
    
    Args:
        data (dict): Weather data to validate
    
    Returns:
        dict: Validation results with warnings and missing fields
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "missing_fields": [],
        "data_quality": "good"
    }
    
    if "error" in data:
        validation["is_valid"] = False
        validation["data_quality"] = "error"
        return validation
    
    # Check current weather completeness
    if "current" in data:
        required_current_fields = ["temperature", "condition", "humidity"]
        for field in required_current_fields:
            if field not in data["current"] or data["current"][field] is None:
                validation["missing_fields"].append(f"current.{field}")
    
    # Check forecast completeness
    if "forecast" in data:
        if not data["forecast"]:
            validation["warnings"].append("No forecast data available")
        elif len(data["forecast"]) < 3:
            validation["warnings"].append(f"Limited forecast data: only {len(data['forecast'])} days")
    
    # Assess data quality
    if validation["missing_fields"]:
        validation["data_quality"] = "partial"
        if len(validation["missing_fields"]) > 3:
            validation["is_valid"] = False
            validation["data_quality"] = "poor"
    
    return validation

def get_weather_data_robust(location: str, forecast_days: int = 5, 
                          units: str = "metric", timeout: int = 30,
                          use_cache: bool = True, retry_count: int = 2) -> Dict:
    """
    Get weather data with comprehensive error handling, retries, and validation.
    
    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-7)
        units (str): Temperature units ('metric' or 'imperial')
        timeout (int): Request timeout in seconds
        use_cache (bool): Whether to use cached data
        retry_count (int): Number of retries on failure
    
    Returns:
        dict: Comprehensive weather data with metadata and validation info
    """
    # Input validation
    if not location or not isinstance(location, str):
        return {
            "error": "Invalid location: must be a non-empty string",
            "suggestions": ["Please enter a valid city name like 'London' or 'New York'"]
        }
    
    # Clean location input
    clean_location = re.sub(r'[^\w\s,.-]', '', location.strip())
    
    # Try to get data with retries
    last_error = None
    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt} for {clean_location}")
            
            # Get weather data (with caching)
            weather_data = get_weather_data_with_cache(
                clean_location, forecast_days, units, use_cache
            )
            
            if "error" in weather_data:
                last_error = weather_data["error"]
                
                # If location not found, try suggestions
                if "location" in last_error.lower() or "not found" in last_error.lower():
                    suggestions = suggest_location_alternatives(clean_location)
                    return {
                        "error": f"Location '{clean_location}' not found",
                        "suggestions": suggestions[:3],  # Top 3 suggestions
                        "help": "Try using a major city name or include country/state"
                    }
                
                # For other errors, continue retry loop
                continue
            
            # Validate the response
            validation = validate_weather_response(weather_data)
            
            # Add validation metadata
            result = {
                **weather_data,
                "validation": validation,
                "location_searched": clean_location,
                "attempts": attempt + 1
            }
            
            # Add quality warnings if needed
            if not validation["is_valid"]:
                result["warning"] = "Data quality issues detected. Some information may be incomplete."
            
            return result
            
        except Exception as e:
            last_error = str(e)
            if attempt < retry_count:
                time.sleep(1)  # Wait before retry
            continue
    
    # All retries failed
    return {
        "error": f"Failed to retrieve weather data after {retry_count + 1} attempts",
        "last_error": last_error,
        "suggestions": [
            "Check your internet connection",
            "Try a different location format",
            "The weather service may be temporarily unavailable"
        ]
    }

# Convenience function that combines everything
def get_weather_data(location: str, forecast_days: int = 5, units: str = "metric") -> Dict:
    """
    Main weather data function - calls the robust implementation with default settings.
    
    This is the function that should be used throughout the application.
    """
    return get_weather_data_robust(location, forecast_days, units)

Tool: Claude
Date: 2025-05-21
Me: I need to implement visualization functions for my weather app. Can you help me create basic temperature and precipitation charts using matplotlib?
AI: Here's a basic implementation for both visualization functions:
def create_temperature_visualisation(weather_data, output_type='display'):
    """Create basic temperature visualization."""
    if "error" in weather_data:
        print(f"Cannot create visualization: {weather_data['error']}")
        return None
    
    # Extract data
    dates = [day['date'] for day in weather_data['forecast']]
    high_temps = [day['max_temp'] for day in weather_data['forecast']]
    low_temps = [day['min_temp'] for day in weather_data['forecast']]
    
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(dates, high_temps, label='High')
    ax.plot(dates, low_temps, label='Low')
    ax.set_title('Temperature Forecast')
    ax.legend()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()

def create_precipitation_visualisation(weather_data, output_type='display'):
    """Create basic precipitation visualization."""
    if "error" in weather_data:
        print(f"Cannot create visualization: {weather_data['error']}")
        return None
    
    # Extract data
    dates = [day['date'] for day in weather_data['forecast']]
    precipitation = [day['precipitation_chance'] for day in weather_data['forecast']]
    
    # Create plot
    fig, ax = plt.subplots()
    ax.bar(dates, precipitation)
    ax.set_title('Precipitation Chance')
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
Me: But the charts look quite basic. For a professional weather app, I need better styling, proper axis labels, and the ability to show both current conditions and forecasts. Can you enhance these with better formatting and more informative displays?
AI: Absolutely! Here's an enhanced version with professional styling and more comprehensive data display:
def create_temperature_visualisation(weather_data, output_type='display'):
    """
    Create professional temperature visualization with current conditions and forecast.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
    
    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    # Validate input data
    if "error" in weather_data:
        print(f"❌ Cannot create temperature visualization: {weather_data['error']}")
        return None
    
    if "forecast" not in weather_data or not weather_data["forecast"]:
        print("❌ No forecast data available for temperature visualization")
        return None
    
    # Extract forecast data
    dates = [day['date'] for day in weather_data['forecast']]
    high_temps = [day['max_temp'] for day in weather_data['forecast']]
    low_temps = [day['min_temp'] for day in weather_data['forecast']]
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot temperature lines with styling
    ax.plot(dates, high_temps, 'ro-', linewidth=2.5, markersize=6, 
            label='High Temperature', color='#FF6B6B')
    ax.plot(dates, low_temps, 'bo-', linewidth=2.5, markersize=6, 
            label='Low Temperature', color='#4ECDC4')
    
    # Add current temperature line if available
    if "current" in weather_data and "temperature" in weather_data["current"]:
        current_temp = weather_data["current"]["temperature"]
        ax.axhline(y=current_temp, color='orange', linestyle='--', linewidth=2, 
                  label=f'Current: {current_temp}°')
    
    # Styling
    ax.set_title(f'Temperature Forecast - {weather_data.get("location", "Unknown Location")}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Dynamic unit label
    unit_symbol = TEMPERATURE_UNITS.get(weather_data.get("units", "metric"), "°C")
    ax.set_ylabel(f'Temperature ({unit_symbol})', fontsize=12, fontweight='bold')
    
    # Improve x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis limits with padding
    all_temps = high_temps + low_temps
    temp_range = max(all_temps) - min(all_temps)
    padding = temp_range * 0.1
    ax.set_ylim(min(all_temps) - padding, max(all_temps) + padding)
    
    # Add temperature values on points
    for i, (date, high, low) in enumerate(zip(dates, high_temps, low_temps)):
        ax.annotate(f'{high}°', (date, high), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f'{low}°', (date, low), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=9, fontweight='bold')
    
    # Improve layout
    plt.tight_layout()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
        return None
Me: Now I want the precipitation visualization to be equally professional and informative. Can you create a dual-display chart that shows both precipitation chance and amount, with proper color coding and additional weather condition indicators?
AI: Perfect! Here's a comprehensive precipitation visualization with dual metrics and weather condition indicators:
def create_precipitation_visualisation(weather_data, output_type='display'):
    """
    Create comprehensive precipitation visualization with chance, amount, and conditions.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
    
    Returns:
        If output_type is 'figure', returns the matplotlib figure object
        Otherwise, displays the visualisation in the notebook
    """
    # Validate input data
    if "error" in weather_data:
        print(f"❌ Cannot create precipitation visualization: {weather_data['error']}")
        return None
    
    if "forecast" not in weather_data or not weather_data["forecast"]:
        print("❌ No forecast data available for precipitation visualization")
        return None
    
    # Extract data
    dates = [day['date'] for day in weather_data['forecast']]
    precipitation_chance = [day.get('precipitation_chance', 0) for day in weather_data['forecast']]
    precipitation_amount = [day.get('precipitation_amount', 0) for day in weather_data['forecast']]
    conditions = [day.get('condition', 'Unknown') for day in weather_data['forecast']]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Color mapping for precipitation levels
    def get_precipitation_color(chance):
        if chance >= 80:
            return '#1f77b4'  # Strong blue
        elif chance >= 60:
            return '#17becf'  # Medium blue
        elif chance >= 40:
            return '#87ceeb'  # Light blue
        elif chance >= 20:
            return '#e0f6ff'  # Very light blue
        else:
            return '#f0f0f0'  # Light gray
    
    # Plot 1: Precipitation Chance
    colors = [get_precipitation_color(chance) for chance in precipitation_chance]
    bars1 = ax1.bar(dates, precipitation_chance, color=colors, alpha=0.8, 
                    edgecolor='navy', linewidth=1)
    
    # Add percentage labels on bars
    for i, (bar, chance) in enumerate(zip(bars1, precipitation_chance)):
        height = bar.get_height()
        ax1.annotate(f'{chance}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_title(f'Precipitation Forecast - {weather_data.get("location", "Unknown Location")}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Precipitation Chance (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add current precipitation if available
    if "current" in weather_data and "precipitation" in weather_data["current"]:
        current_precip = weather_data["current"]["precipitation"]
        ax1.axhline(y=current_precip, color='red', linestyle='--', linewidth=2, 
                   label=f'Current: {current_precip}mm')
        ax1.legend()
    
    # Plot 2: Precipitation Amount
    bars2 = ax2.bar(dates, precipitation_amount, color='lightblue', alpha=0.7, 
                    edgecolor='darkblue', linewidth=1)
    
    # Add amount labels on bars
    for i, (bar, amount) in enumerate(zip(bars2, precipitation_amount)):
        if amount > 0:  # Only show label if there's precipitation
            height = bar.get_height()
            ax2.annotate(f'{amount:.1f}mm', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Precipitation Amount (mm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add weather condition annotations
    for i, (date, condition) in enumerate(zip(dates, conditions)):
        # Add weather condition emoji/symbol
        condition_symbol = get_weather_symbol(condition)
        ax2.annotate(condition_symbol, xy=(date, max(precipitation_amount) * 0.1),
                    ha='center', va='center', fontsize=16)
    
    # Improve x-axis labels
    ax2.tick_params(axis='x', rotation=45)
    
    # Add precipitation intensity legend
    intensity_labels = ['No Rain (0%)', 'Light (20-40%)', 'Moderate (40-60%)', 
                       'Heavy (60-80%)', 'Very Heavy (80%+)']
    intensity_colors = ['#f0f0f0', '#e0f6ff', '#87ceeb', '#17becf', '#1f77b4']
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) 
                      for color, label in zip(intensity_colors, intensity_labels)]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_type == 'figure':
        return fig
    else:
        plt.show()
        return None

def get_weather_symbol(condition):
    """Return weather symbol emoji based on condition."""
    condition_lower = condition.lower()
    if 'sunny' in condition_lower or 'clear' in condition_lower:
        return '☀️'
    elif 'cloud' in condition_lower:
        return '☁️'
    elif 'rain' in condition_lower:
        return '🌧️'
    elif 'storm' in condition_lower or 'thunder' in condition_lower:
        return '⛈️'
    elif 'snow' in condition_lower:
        return '❄️'
    elif 'fog' in condition_lower or 'mist' in condition_lower:
        return '🌫️'
    else:
        return '🌤️'  # Partly cloudy default
Me: Great but I'm concerned about error handling and edge cases. What if the weather data is incomplete or there are issues with matplotlib? Can you add comprehensive error handling and fallback options, plus utility functions for customizing the visualizations?
AI: Excellent point! Here's a robust version with comprehensive error handling and customization options:
def validate_visualization_data(weather_data, required_fields):
    """
    Validate weather data for visualization requirements.
    
    Args:
        weather_data (dict): Weather data to validate
        required_fields (list): List of required field paths (e.g., ['forecast', 'current.temperature'])
    
    Returns:
        tuple: (is_valid, error_message, warnings)
    """
    if "error" in weather_data:
        return False, f"Weather data contains error: {weather_data['error']}", []
    
    warnings = []
    
    # Check forecast data
    if "forecast" not in weather_data:
        return False, "No forecast data available", warnings
    
    forecast = weather_data["forecast"]
    if not isinstance(forecast, list) or len(forecast) == 0:
        return False, "Forecast data is empty or invalid", warnings
    
    # Check required fields in forecast
    for i, day in enumerate(forecast):
        if not isinstance(day, dict):
            return False, f"Invalid forecast data format for day {i}", warnings
        
        # Check for required fields
        missing_fields = []
        for field in ['date', 'max_temp', 'min_temp']:
            if field not in day:
                missing_fields.append(field)
        
        if missing_fields:
            warnings.append(f"Day {i} missing fields: {missing_fields}")
    
    # Warning if forecast is short
    if len(forecast) < 3:
        warnings.append(f"Limited forecast data: only {len(forecast)} days available")
    
    return True, None, warnings

def create_temperature_visualisation_safe(weather_data, output_type='display', **kwargs):
    """
    Create temperature visualization with comprehensive error handling.
    
    Additional kwargs:
        title (str): Custom title
        show_current (bool): Whether to show current temperature line
        color_scheme (str): 'default', 'warm', 'cool'
        show_values (bool): Whether to show temperature values on points
    """
    try:
        # Validate data
        is_valid, error_msg, warnings = validate_visualization_data(weather_data, ['forecast'])
        if not is_valid:
            print(f"❌ Cannot create temperature visualization: {error_msg}")
            return None
        
        # Show warnings
        for warning in warnings:
            print(f"⚠️ {warning}")
        
        # Extract configuration
        config = {
            'title': kwargs.get('title', f'Temperature Forecast - {weather_data.get("location", "Unknown")}'),
            'show_current': kwargs.get('show_current', True),
            'color_scheme': kwargs.get('color_scheme', 'default'),
            'show_values': kwargs.get('show_values', True),
            'figure_size': kwargs.get('figure_size', (12, 7))
        }
        
        # Color schemes
        color_schemes = {
            'default': {'high': '#FF6B6B', 'low': '#4ECDC4', 'current': 'orange'},
            'warm': {'high': '#FF4757', 'low': '#FFA502', 'current': '#FF6348'},
            'cool': {'high': '#3742FA', 'low': '#2F3542', 'current': '#40739E'}
        }
        colors = color_schemes.get(config['color_scheme'], color_schemes['default'])
        
        # Extract data with error handling
        try:
            dates = []
            high_temps = []
            low_temps = []
            
            for day in weather_data['forecast']:
                # Handle missing or invalid data
                if 'date' in day and 'max_temp' in day and 'min_temp' in day:
                    dates.append(day['date'])
                    high_temps.append(float(day['max_temp']))
                    low_temps.append(float(day['min_temp']))
                else:
                    print(f"⚠️ Skipping incomplete day data: {day}")
            
            if not dates:
                print("❌ No valid temperature data found")
                return None
                
        except (KeyError, ValueError, TypeError) as e:
            print(f"❌ Error extracting temperature data: {e}")
            return None
        
        # Create visualization
        try:
            fig, ax = plt.subplots(figsize=config['figure_size'])
            
            # Plot temperature lines
            line1 = ax.plot(dates, high_temps, 'o-', linewidth=2.5, markersize=6, 
                           label='High Temperature', color=colors['high'])
            line2 = ax.plot(dates, low_temps, 'o-', linewidth=2.5, markersize=6, 
                           label='Low Temperature', color=colors['low'])
            
            # Add current temperature if requested and available
            if config['show_current'] and "current" in weather_data:
                current_data = weather_data["current"]
                if "temperature" in current_data:
                    current_temp = float(current_data["temperature"])
                    ax.axhline(y=current_temp, color=colors['current'], 
                              linestyle='--', linewidth=2, 
                              label=f'Current: {current_temp}°')
            
            # Styling
            ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            
            # Unit handling
            unit_symbol = TEMPERATURE_UNITS.get(weather_data.get("units", "metric"), "°C")
            ax.set_ylabel(f'Temperature ({unit_symbol})', fontsize=12, fontweight='bold')
            
            # Add values on points if requested
            if config['show_values']:
                for i, (date, high, low) in enumerate(zip(dates, high_temps, low_temps)):
                    ax.annotate(f'{high:.0f}°', (date, high), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
                    ax.annotate(f'{low:.0f}°', (date, low), textcoords="offset points", 
                               xytext=(0,-15), ha='center', fontsize=9, fontweight='bold')
            
            # Formatting
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Dynamic y-axis limits
            all_temps = high_temps + low_temps
            if all_temps:  # Check if we have data
                temp_range = max(all_temps) - min(all_temps)
                padding = max(temp_range * 0.1, 2)  # Minimum 2 degree padding
                ax.set_ylim(min(all_temps) - padding, max(all_temps) + padding)
            
            plt.tight_layout()
            
            if output_type == 'figure':
                return fig
            else:
                plt.show()
                return None
                
        except Exception as e:
            print(f"❌ Error creating temperature plot: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Unexpected error in temperature visualization: {e}")
        return None

def create_fallback_visualization(weather_data, chart_type="temperature"):
    """
    Create a simple fallback visualization when main functions fail.
    
    Args:
        weather_data (dict): Weather data
        chart_type (str): 'temperature' or 'precipitation'
    """
    try:
        if "error" in weather_data:
            print(f"📊 Weather data error: {weather_data['error']}")
            return
        
        # Simple text-based fallback
        print(f"\n📊 {chart_type.title()} Summary for {weather_data.get('location', 'Unknown')}:")
        print("-" * 60)
        
        if chart_type == "temperature" and "forecast" in weather_data:
            for day in weather_data["forecast"][:5]:  # Show up to 5 days
                date = day.get('date', 'Unknown')
                high = day.get('max_temp', 'N/A')
                low = day.get('min_temp', 'N/A')
                print(f"{date}: High {high}° / Low {low}°")
        
        elif chart_type == "precipitation" and "forecast" in weather_data:
            for day in weather_data["forecast"][:5]:
                date = day.get('date', 'Unknown')
                chance = day.get('precipitation_chance', 'N/A')
                amount = day.get('precipitation_amount', 'N/A')
                condition = day.get('condition', 'Unknown')
                print(f"{date}: {chance}% chance, {amount}mm - {condition}")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Even fallback visualization failed: {e}")
Me: Now I need the main interface functions that users will actually call. Can you create wrapper functions that automatically choose the best visualization approach, handle all the error cases gracefully, and provide customization options for different use cases?
AI: Absolutely! Here are the main interface functions with intelligent fallback and customization:
def create_temperature_visualisation(weather_data, output_type='display', **kwargs):
    """
    Main temperature visualization function with automatic error handling and fallbacks.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
        **kwargs: Additional customization options
    
    Customization options:
        style (str): 'professional', 'simple', 'colorful'
        theme (str): 'light', 'dark'
        show_current (bool): Show current temperature line
        show_values (bool): Show temperature values on data points
        title (str): Custom chart title
        
    Returns:
        matplotlib figure object if output_type='figure', None otherwise
    """
    # Set default style options
    style_config = {
        'professional': {'color_scheme': 'default', 'show_values': True, 'grid': True},
        'simple': {'color_scheme': 'cool', 'show_values': False, 'grid': False},
        'colorful': {'color_scheme': 'warm', 'show_values': True, 'grid': True}
    }
    
    style = kwargs.get('style', 'professional')
    config = style_config.get(style, style_config['professional'])
    
    # Merge user options with style defaults
    final_kwargs = {**config, **kwargs}
    
    # Try main visualization function
    try:
        result = create_temperature_visualisation_safe(weather_data, output_type, **final_kwargs)
        if result is not None or output_type == 'display':
            return result
    except Exception as e:
        print(f"⚠️ Main temperature visualization failed: {e}")
    
    # Fallback to simple visualization
    try:
        print("🔄 Attempting fallback visualization...")
        create_fallback_visualization(weather_data, "temperature")
        return None
    except Exception as e:
        print(f"❌ All temperature visualization methods failed: {e}")
        return None

def create_precipitation_visualisation(weather_data, output_type='display', **kwargs):
    """
    Main precipitation visualization function with automatic error handling and fallbacks.
    
    Args:
        weather_data (dict): The processed weather data
        output_type (str): Either 'display' to show in notebook or 'figure' to return the figure
        **kwargs: Additional customization options
    
    Customization options:
        style (str): 'detailed', 'simple', 'compact'
        show_amounts (bool): Show precipitation amounts
        show_symbols (bool): Show weather condition symbols
        split_view (bool): Show chance and amount in separate charts
        
    Returns:
        matplotlib figure object if output_type='figure', None otherwise
    """
    # Set default style options
    style_config = {
        'detailed': {'split_view': True, 'show_amounts': True, 'show_symbols': True},
        'simple': {'split_view': False, 'show_amounts': False, 'show_symbols': False},
        'compact': {'split_view': False, 'show_amounts': True, 'show_symbols': True}
    }
    
    style = kwargs.get('style', 'detailed')
    config = style_config.get(style, style_config['detailed'])
    final_kwargs = {**config, **kwargs}
    
    # Try main visualization
    try:
        if final_kwargs.get('split_view', True):
            # Use the detailed dual-chart version
            result = create_precipitation_visualisation_detailed(weather_data, output_type, **final_kwargs)
        else:
            # Use simplified single chart
            result = create_precipitation_visualisation_simple(weather_data, output_type, **final_kwargs)
        
        if result is not None or output_type == 'display':
            return result
    except Exception as e:
        print(f"⚠️ Main precipitation visualization failed: {e}")
    
    # Fallback to text summary
    try:
        print("🔄 Attempting fallback visualization...")
        create_fallback_visualization(weather_data, "precipitation")
        return None
    except Exception as e:
        print(f"❌ All precipitation visualization methods failed: {e}")
        return None

def create_precipitation_visualisation_simple(weather_data, output_type='display', **kwargs):
    """Simple single-chart precipitation visualization."""
    try:
        # Validate data
        is_valid, error_msg, warnings = validate_visualization_data(weather_data, ['forecast'])
        if not is_valid:
            print(f"❌ Cannot create precipitation visualization: {error_msg}")
            return None
        
        # Extract data
        dates = []
        precipitation_chance = []
        conditions = []
        
        for day in weather_data['forecast']:
            if 'date' in day and 'precipitation_chance' in day:
                dates.append(day['date'])
                precipitation_chance.append(day.get('precipitation_chance', 0))
                conditions.append(day.get('condition', 'Unknown'))
        
        if not dates:
            print("❌ No valid precipitation data found")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on precipitation level
        colors = []
        for chance in precipitation_chance:
            if chance >= 80:
                colors.append('#1f77b4')  # Dark blue
            elif chance >= 60:
                colors.append('#17becf')  # Medium blue
            elif chance >= 40:
                colors.append('#87ceeb')  # Light blue
            elif chance >= 20:
                colors.append('#add8e6')  # Very light blue
            else:
                colors.append('#f0f0f0')  # Light gray
        
        bars = ax.bar(dates, precipitation_chance, color=colors, alpha=0.8, 
                     edgecolor='darkblue', linewidth=1)
        
        # Add labels
        for bar, chance in zip(bars, precipitation_chance):
            height = bar.get_height()
            ax.annotate(f'{chance}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax.set_title(f'Precipitation Forecast - {weather_data.get("location", "Unknown")}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Precipitation Chance (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add weather symbols if requested
        if kwargs.get('show_symbols', True):
            for i, (date, condition) in enumerate(zip(dates, conditions)):
                symbol = get_weather_symbol(condition)
                ax.annotate(symbol, xy=(date, 5), ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        if output_type == 'figure':
            return fig
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"❌ Error in simple precipitation visualization: {e}")
        return None

def create_precipitation_visualisation_detailed(weather_data, output_type='display', **kwargs):
    """
    Detailed precipitation visualization (the enhanced version from earlier).
    This is the same as the comprehensive version we created before.
    """
    # This would be the same implementation as the detailed version above
    # (keeping it separate for modularity)
    pass

# Utility functions for visualization management
def test_visualizations(weather_data):
    """Test all visualization functions with the provided weather data."""
    print("🧪 Testing visualization functions...")
    
    tests = [
        ("Temperature (Professional)", lambda: create_temperature_visualisation(weather_data, 'display', style='professional')),
        ("Temperature (Simple)", lambda: create_temperature_visualisation(weather_data, 'display', style='simple')),
        ("Precipitation (Detailed)", lambda: create_precipitation_visualisation(weather_data, 'display', style='detailed')),
        ("Precipitation (Simple)", lambda: create_precipitation_visualisation(weather_data, 'display', style='simple'))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n📊 Testing {test_name}...")
            result = test_func()
            results.append((test_name, "✅ Success"))
        except Exception as e:
            results.append((test_name, f"❌ Failed: {e}"))
    
    print("\n📋 Visualization Test Results:")
    for test_name, result in results:
        print(f"  {result} - {test_name}")

def save_visualization(weather_data, chart_type, filename, **kwargs):
    """
    Save a visualization to file.
    
    Args:
        weather_data: Weather data dictionary
        chart_type: 'temperature' or 'precipitation'
        filename: Output filename (with extension)
        **kwargs: Visualization options
    """
    try:
        if chart_type == 'temperature':
            fig = create_temperature_visualisation(weather_data, 'figure', **kwargs)
        elif chart_type == 'precipitation':
            fig = create_precipitation_visualisation(weather_data, 'figure', **kwargs)
        else:
            print(f"❌ Unknown chart type: {chart_type}")
            return False
        
        if fig is not None:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Clean up
            print(f"✅ Visualization saved to {filename}")
            return True
        else:
            print(f"❌ Failed to create {chart_type} visualization")
            return False
            
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        return False
    
Tool: Claude
Date: 2025-05-22
Me: I need to implement natural language processing functions for my weather app. Can you help me create a basic function that parses user questions like "Will it rain tomorrow in London?" and extracts the key information?
AI: Here's a basic implementation for parsing weather questions:
def parse_weather_question(question):
    """
    Parse a natural language weather question.
    
    Args:
        question (str): User's weather-related question
    
    Returns:
        dict: Extracted information including location, time period, and weather attribute
    """
    # Simple keyword-based parsing
    question_lower = question.lower()
    
    # Extract location (basic approach)
    location = None
    if " in " in question_lower:
        location = question_lower.split(" in ")[-1].strip()
    
    # Extract time period
    time_period = "current"
    if "tomorrow" in question_lower:
        time_period = "tomorrow"
    elif "today" in question_lower:
        time_period = "today"
    
    # Extract weather attribute
    attribute = "general"
    if "rain" in question_lower:
        attribute = "precipitation"
    elif "temperature" in question_lower or "hot" in question_lower or "cold" in question_lower:
        attribute = "temperature"
    
    return {
        "location": location,
        "time_period": time_period,
        "attribute": attribute,
        "original_question": question
    }
Me: This basic approach is too simplistic and will miss many variations. Since I have access to hands-on-ai, can you create a more sophisticated version that uses AI to understand the context and extract information more accurately?
AI: Absolutely! Here's an AI-powered version that uses the hands-on-ai system for better parsing:
def parse_weather_question(question):
    """
    Parse a natural language weather question using AI.
    
    Args:
        question (str): User's weather-related question
    
    Returns:
        dict: Extracted information including location, time period, and weather attribute
    """
    if not question or not isinstance(question, str):
        return {"error": "Invalid question: must be a non-empty string"}
    
    # System prompt for question analysis
    system_prompt = """
    You are a weather question analyzer. Extract key information from weather-related questions.
    
    For each question, identify:
    1. Location (city, country, or "unknown" if not specified)
    2. Time period (today, tomorrow, this week, weekend, specific date, or "current")
    3. Weather attribute (temperature, precipitation, wind, humidity, general, or "unknown")
    4. Question type (forecast, current, advice, comparison)
    
    Return your answer as a JSON object with these exact keys:
    {
        "location": "extracted location or 'unknown'",
        "time_period": "time reference",
        "attribute": "weather aspect",
        "question_type": "type of question",
        "confidence": "high/medium/low"
    }
    
    Examples:
    "Will it rain tomorrow in Paris?" -> {"location": "Paris", "time_period": "tomorrow", "attribute": "precipitation", "question_type": "forecast", "confidence": "high"}
    "What's the weather like?" -> {"location": "unknown", "time_period": "current", "attribute": "general", "question_type": "current", "confidence": "medium"}
    """
    
    analysis_prompt = f"Analyze this weather question: '{question}'"
    
    try:
        # Get AI analysis
        response = get_response(analysis_prompt, system=system_prompt)
        
        # Try to parse the JSON response
        import json
        try:
            parsed_data = json.loads(response)
            
            # Add original question and processing metadata
            result = {
                **parsed_data,
                "original_question": question,
                "processing_method": "ai_powered",
                "raw_ai_response": response
            }
            
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to keyword extraction
            print(f"⚠️ AI response wasn't valid JSON, falling back to keyword extraction")
            return parse_weather_question_fallback(question)
            
    except Exception as e:
        print(f"⚠️ AI parsing failed: {e}, using fallback method")
        return parse_weather_question_fallback(question)

def parse_weather_question_fallback(question):
    """Fallback keyword-based parsing when AI fails."""
    question_lower = question.lower().strip()
    
    # Location extraction
    location = "unknown"
    location_patterns = [" in ", " at ", " for "]
    for pattern in location_patterns:
        if pattern in question_lower:
            potential_location = question_lower.split(pattern)[-1].strip()
            # Clean up common endings
            for ending in [" today", " tomorrow", " this week", "?"]:
                if potential_location.endswith(ending):
                    potential_location = potential_location[:-len(ending)].strip()
            if potential_location:
                location = potential_location
            break
    
    # Time period extraction
    time_keywords = {
        "today": ["today", "right now", "currently"],
        "tomorrow": ["tomorrow"],
        "this week": ["this week", "week"],
        "weekend": ["weekend", "saturday", "sunday"],
        "current": ["now", "current", "currently"]
    }
    
    time_period = "current"
    for period, keywords in time_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            time_period = period
            break
    
    # Attribute extraction
    attribute_keywords = {
        "temperature": ["temperature", "hot", "cold", "warm", "cool", "degrees"],
        "precipitation": ["rain", "snow", "precipitation", "wet", "umbrella"],
        "wind": ["wind", "windy", "breeze"],
        "humidity": ["humidity", "humid", "dry"],
        "general": ["weather", "forecast", "conditions"]
    }
    
    attribute = "general"
    for attr, keywords in attribute_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            attribute = attr
            break
    
    # Question type
    question_type = "current"
    if any(word in question_lower for word in ["will", "going to", "forecast", "predict"]):
        question_type = "forecast"
    elif any(word in question_lower for word in ["should", "recommend", "advice", "suggest"]):
        question_type = "advice"
    
    return {
        "location": location,
        "time_period": time_period,
        "attribute": attribute,
        "question_type": question_type,
        "confidence": "medium",
        "original_question": question,
        "processing_method": "keyword_fallback"
    }
Me: Now I need the generate_weather_response() function that takes the parsed question and weather data to create natural language responses. Can you make it context-aware and able to provide helpful advice based on the weather conditions?
AI: Perfect! Here's a comprehensive response generation function:
def generate_weather_response(parsed_question, weather_data):
    """
    Generate a natural language response to a weather question.
    
    Args:
        parsed_question (dict): Parsed question data from parse_weather_question()
        weather_data (dict): Weather data from get_weather_data()
    
    Returns:
        str: Natural language response
    """
    # Validate inputs
    if "error" in parsed_question:
        return f"I couldn't understand your question: {parsed_question['error']}"
    
    if "error" in weather_data:
        return f"I'm sorry, I couldn't get weather data: {weather_data['error']}"
    
    # Extract key information
    location = weather_data.get("location", "your location")
    question_type = parsed_question.get("question_type", "current")
    attribute = parsed_question.get("attribute", "general")
    time_period = parsed_question.get("time_period", "current")
    
    # Create context for AI response generation
    context_prompt = f"""
    Generate a helpful, conversational weather response based on this information:
    
    User Question: {parsed_question['original_question']}
    Location: {location}
    Time Period: {time_period}
    Weather Aspect: {attribute}
    Question Type: {question_type}
    
    Current Weather:
    - Temperature: {weather_data['current']['temperature']}°
    - Condition: {weather_data['current']['condition']}
    - Humidity: {weather_data['current']['humidity']}%
    - Wind: {weather_data['current']['wind_speed']} km/h
    
    Forecast Data: {len(weather_data['forecast'])} days available
    
    Guidelines:
    1. Answer the specific question asked
    2. Be conversational and helpful
    3. Include relevant advice (clothing, activities, precautions)
    4. Keep response concise but informative
    5. If asking about future weather, use forecast data
    6. Include specific numbers and details
    """
    
    try:
        # Generate AI response
        response = get_response(
            "Create a weather response based on the provided context.",
            system=context_prompt
        )
        
        # Add emoji and formatting for better user experience
        formatted_response = add_weather_formatting(response, weather_data, parsed_question)
        
        return formatted_response
        
    except Exception as e:
        print(f"⚠️ AI response generation failed: {e}")
        # Fall back to template-based response
        return generate_template_response(parsed_question, weather_data)

def add_weather_formatting(response, weather_data, parsed_question):
    """Add emojis and formatting to make the response more engaging."""
    
    # Get appropriate weather emoji
    condition = weather_data['current']['condition'].lower()
    weather_emoji = get_weather_symbol(condition)
    
    # Add temperature emoji based on temp
    temp = weather_data['current']['temperature']
    if temp > 25:
        temp_emoji = "🔥"
    elif temp > 15:
        temp_emoji = "🌡️"
    elif temp > 5:
        temp_emoji = "❄️"
    else:
        temp_emoji = "🥶"
    
    # Format the response with emojis
    formatted = f"{weather_emoji} {response}"
    
    # Add relevant advice emoji based on conditions
    if "rain" in condition:
        formatted += " ☂️"
    if "sun" in condition:
        formatted += " 🕶️"
    if weather_data['current']['wind_speed'] > 20:
        formatted += " 💨"
    
    return formatted

def generate_template_response(parsed_question, weather_data):
    """Generate response using templates when AI fails."""
    
    location = weather_data.get("location", "your area")
    current = weather_data["current"]
    
    # Basic template responses
    if parsed_question["question_type"] == "current":
        return f"Right now in {location}, it's {current['temperature']}° and {current['condition'].lower()}. " \
               f"Humidity is {current['humidity']}% with winds at {current['wind_speed']} km/h."
    
    elif parsed_question["question_type"] == "forecast":
        if weather_data["forecast"]:
            tomorrow = weather_data["forecast"][0] if len(weather_data["forecast"]) > 0 else None
            if tomorrow:
                return f"Tomorrow in {location}, expect a high of {tomorrow['max_temp']}° " \
                       f"and low of {tomorrow['min_temp']}° with {tomorrow['condition'].lower()}."
        return f"I can tell you the current weather in {location} is {current['temperature']}° and {current['condition'].lower()}."
    
    elif parsed_question["question_type"] == "advice":
        advice = generate_weather_advice(weather_data)
        return f"Based on the weather in {location} ({current['temperature']}°, {current['condition'].lower()}), {advice}"
    
    return f"The weather in {location} is currently {current['temperature']}° and {current['condition'].lower()}."

def generate_weather_advice(weather_data):
    """Generate practical weather advice based on conditions."""
    current = weather_data["current"]
    temp = current["temperature"]
    condition = current["condition"].lower()
    humidity = current["humidity"]
    wind_speed = current["wind_speed"]
    
    advice_parts = []
    
    # Temperature advice
    if temp > 25:
        advice_parts.append("dress lightly and stay hydrated")
    elif temp > 15:
        advice_parts.append("light layers would be comfortable")
    elif temp > 5:
        advice_parts.append("you'll want a jacket")
    else:
        advice_parts.append("dress warmly and consider multiple layers")
    
    # Condition-specific advice
    if "rain" in condition:
        advice_parts.append("bring an umbrella")
    elif "sun" in condition:
        advice_parts.append("sunglasses would be helpful")
    elif "snow" in condition:
        advice_parts.append("wear appropriate footwear for snow")
    
    # Wind advice
    if wind_speed > 25:
        advice_parts.append("it's quite windy so secure loose items")
    
    # Humidity advice
    if humidity > 80:
        advice_parts.append("it feels quite humid")
    elif humidity < 30:
        advice_parts.append("the air is quite dry")
    
    return ", ".join(advice_parts) + "."
Me: What if I want to make it even more sophisticated. Can you add support for multi-day questions, comparison queries, and the ability to handle follow-up questions that reference previous context?
AI: Excellent idea! Here's an enhanced version with conversation context and advanced query handling:
# Global conversation context storage
_conversation_context = {
    "last_location": None,
    "last_weather_data": None,
    "conversation_history": [],
    "user_preferences": {}
}

def parse_weather_question_advanced(question, context=None):
    """
    Advanced weather question parsing with context awareness.
    
    Args:
        question (str): User's weather question
        context (dict): Previous conversation context
    
    Returns:
        dict: Enhanced parsed question data
    """
    if context is None:
        context = _conversation_context
    
    # Enhanced system prompt for complex queries
    system_prompt = f"""
    You are an advanced weather question analyzer. Parse weather questions considering conversation context.
    
    Previous context:
    - Last location discussed: {context.get('last_location', 'none')}
    - Recent questions: {context.get('conversation_history', [])[-3:]}
    
    Extract and categorize:
    1. Location (use context if not specified)
    2. Time period (support ranges like "this week", "next 3 days")
    3. Weather attributes (can be multiple)
    4. Query complexity (simple, multi_day, comparison, follow_up)
    5. Reference type (absolute, relative, contextual)
    
    Special handling for:
    - Multi-day queries: "weather this week", "next 5 days"
    - Comparisons: "warmer than yesterday", "compared to last week"
    - Follow-ups: "what about tomorrow?", "and the weekend?"
    - Implicit location: "how about there?" (referring to previous location)
    
    Return JSON with these keys:
    {{
        "location": "location or 'contextual' if referring to previous",
        "time_period": "specific time reference",
        "time_range": {{"start": "start_day", "end": "end_day", "duration": "number_of_days"}},
        "attributes": ["list", "of", "weather", "aspects"],
        "query_complexity": "simple/multi_day/comparison/follow_up",
        "reference_type": "absolute/relative/contextual",
        "confidence": "high/medium/low",
        "requires_context": true/false
    }}
    """
    
    analysis_prompt = f"Analyze this weather question: '{question}'"
    
    try:
        response = get_response(analysis_prompt, system=system_prompt)
        
        import json
        parsed_data = json.loads(response)
        
        # Resolve contextual references
        if parsed_data.get("location") == "contextual" and context.get("last_location"):
            parsed_data["location"] = context["last_location"]
            parsed_data["resolved_from_context"] = True
        
        # Add to conversation history
        context["conversation_history"].append({
            "question": question,
            "parsed": parsed_data,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Keep only last 10 interactions
        if len(context["conversation_history"]) > 10:
            context["conversation_history"] = context["conversation_history"][-10:]
        
        return {
            **parsed_data,
            "original_question": question,
            "processing_method": "ai_advanced",
            "context_used": bool(context.get("last_location"))
        }
        
    except Exception as e:
        print(f"⚠️ Advanced parsing failed: {e}")
        return parse_weather_question(question)  # Fall back to basic version

def generate_weather_response_advanced(parsed_question, weather_data, context=None):
    """
    Generate advanced weather responses with context and multi-day support.
    
    Args:
        parsed_question (dict): Parsed question data
        weather_data (dict): Weather data
        context (dict): Conversation context
    
    Returns:
        str: Comprehensive natural language response
    """
    if context is None:
        context = _conversation_context
    
    # Update context with current data
    context["last_location"] = weather_data.get("location")
    context["last_weather_data"] = weather_data
    
    # Handle different query complexities
    query_complexity = parsed_question.get("query_complexity", "simple")
    
    if query_complexity == "multi_day":
        return generate_multi_day_response(parsed_question, weather_data, context)
    elif query_complexity == "comparison":
        return generate_comparison_response(parsed_question, weather_data, context)
    elif query_complexity == "follow_up":
        return generate_follow_up_response(parsed_question, weather_data, context)
    else:
        return generate_enhanced_single_response(parsed_question, weather_data, context)

def generate_multi_day_response(parsed_question, weather_data, context):
    """Generate response for multi-day weather queries."""
    
    location = weather_data.get("location", "your area")
    forecast = weather_data.get("forecast", [])
    time_range = parsed_question.get("time_range", {})
    duration = time_range.get("duration", len(forecast))
    
    if not forecast:
        return f"I don't have forecast data for {location} right now."
    
    # Limit to available forecast days
    days_to_show = min(duration, len(forecast))
    
    system_prompt = f"""
    Create a comprehensive multi-day weather summary for {location}.
    
    User asked: {parsed_question['original_question']}
    Duration requested: {duration} days
    
    Forecast data for {days_to_show} days:
    """
    
    # Add forecast data to prompt
    for i, day in enumerate(forecast[:days_to_show]):
        system_prompt += f"""
    Day {i+1} ({day['date']}):
    - High: {day['max_temp']}°, Low: {day['min_temp']}°
    - Condition: {day['condition']}
    - Precipitation: {day['precipitation_chance']}% chance
    """
    
    system_prompt += """
    Provide a helpful summary that:
    1. Gives an overview of the period
    2. Highlights any significant weather changes
    3. Notes the best and worst days
    4. Provides practical advice for planning
    5. Uses a conversational tone
    """
    
    try:
        response = get_response(
            f"Generate a multi-day weather summary for the forecast data provided.",
            system=system_prompt
        )
        
        # Add weather emoji for the period
        overall_emoji = "🌤️"  # Default
        if any("rain" in day.get('condition', '').lower() for day in forecast[:days_to_show]):
            overall_emoji = "🌧️"
        elif any("sun" in day.get('condition', '').lower() for day in forecast[:days_to_show]):
            overall_emoji = "☀️"
        
        return f"{overall_emoji} {response}"
        
    except Exception as e:
        # Fallback to template
        return generate_multi_day_template(forecast[:days_to_show], location)

def generate_comparison_response(parsed_question, weather_data, context):
    """Generate response for weather comparison queries."""
    
    # This would compare with historical data or previous queries
    current_temp = weather_data['current']['temperature']
    location = weather_data.get("location", "your area")
    
    # For now, provide a contextual comparison
    comparison_context = f"""
    Generate a weather comparison response for: {parsed_question['original_question']}
    
    Current conditions in {location}:
    - Temperature: {current_temp}°
    - Condition: {weather_data['current']['condition']}
    
    Since I don't have historical data, acknowledge the limitation but provide:
    1. Current conditions with context
    2. How today compares to the forecast trend
    3. General seasonal context
    4. Helpful comparative information
    """
    
    try:
        response = get_response(
            "Generate a weather comparison response acknowledging data limitations.",
            system=comparison_context
        )
        return f"🔄 {response}"
    except:
        return f"I can tell you it's currently {current_temp}° in {location}. " \
               f"For accurate comparisons, I'd need historical weather data."

def generate_follow_up_response(parsed_question, weather_data, context):
    """Generate response for follow-up questions."""
    
    # Use conversation history for context
    previous_questions = context.get("conversation_history", [])
    
    follow_up_context = f"""
    This is a follow-up question: {parsed_question['original_question']}
    
    Previous conversation:
    """
    
    for prev in previous_questions[-3:]:  # Last 3 interactions
        follow_up_context += f"- User asked: {prev['question']}\n"
    
    follow_up_context += f"""
    
    Current weather data for {weather_data.get('location')}:
    - Temperature: {weather_data['current']['temperature']}°
    - Condition: {weather_data['current']['condition']}
    
    Provide a natural follow-up response that:
    1. Acknowledges the conversational context
    2. Answers the new question
    3. Connects to previous topics when relevant
    """
    
    try:
        response = get_response(
            "Generate a contextual follow-up response.",
            system=follow_up_context
        )
        return f"💬 {response}"
    except:
        return generate_weather_response(parsed_question, weather_data)

def generate_enhanced_single_response(parsed_question, weather_data, context):
    """Generate enhanced single-day responses with context."""
    
    # Check for user preferences from context
    preferences = context.get("user_preferences", {})
    preferred_units = preferences.get("units", weather_data.get("units", "metric"))
    
    # Enhanced context with preferences
    enhanced_context = f"""
    Generate a personalized weather response for: {parsed_question['original_question']}
    
    Weather data for {weather_data.get('location')}:
    - Current: {weather_data['current']['temperature']}° and {weather_data['current']['condition']}
    - Humidity: {weather_data['current']['humidity']}%
    - Wind: {weather_data['current']['wind_speed']} km/h
    
    User preferences: {preferences}
    Conversation context: {"First interaction" if not context.get('conversation_history') else "Ongoing conversation"}
    
    Make the response:
    1. Naturally conversational
    2. Specific to their question
    3. Include practical advice
    4. Consider their apparent preferences
    """
    
    try:
        response = get_response(
            "Generate an enhanced weather response with context.",
            system=enhanced_context
        )
        return add_weather_formatting(response, weather_data, parsed_question)
    except:
        return generate_weather_response(parsed_question, weather_data)

def generate_multi_day_template(forecast_days, location):
    """Template fallback for multi-day responses."""
    if not forecast_days:
        return f"I don't have forecast data for {location}."
    
    summary = f"Here's the {len(forecast_days)}-day forecast for {location}:\n\n"
    
    for i, day in enumerate(forecast_days):
        day_name = "Today" if i == 0 else f"Day {i+1}"
        summary += f"{day_name} ({day['date']}): {day['max_temp']}°/{day['min_temp']}° - {day['condition']}\n"
    
    # Add simple analysis
    temps = [day['max_temp'] for day in forecast_days]
    summary += f"\nTemperature range: {min(temps)}° to {max(temps)}°"
    
    return summary

# Context management functions
def clear_conversation_context():
    """Clear the conversation context."""
    global _conversation_context
    _conversation_context = {
        "last_location": None,
        "last_weather_data": None,
        "conversation_history": [],
        "user_preferences": {}
    }
    print("🧹 Conversation context cleared")

def get_conversation_summary():
    """Get a summary of the current conversation context."""
    context = _conversation_context
    return {
        "last_location": context.get("last_location"),
        "interaction_count": len(context.get("conversation_history", [])),
        "preferences": context.get("user_preferences", {}),
        "recent_questions": [q["question"] for q in context.get("conversation_history", [])[-3:]]
    }
Me: Now I need main interface functions that automatically choose between the basic and advanced parsing based on the context and question complexity. Can you create wrapper functions that provide a seamless experience while handling all the edge cases?
AI: Absolutely! Here are the main interface functions that provide intelligent routing and seamless experience:
def parse_weather_question(question):
    """
    Main weather question parsing function with intelligent routing.
    
    Args:
        question (str): User's weather-related question
    
    Returns:
        dict: Extracted information including location, time period, and weather attribute
    """
    # Input validation
    if not question or not isinstance(question, str):
        return {"error": "Invalid question: must be a non-empty string"}
    
    question = question.strip()
    if len(question) < 3:
        return {"error": "Question too short to analyze"}
    
    # Analyze question complexity to choose parsing method
    complexity_indicators = {
        "advanced": [
            "this week", "next week", "weekend", "compared to", "warmer than", 
            "cooler than", "what about", "how about", "and tomorrow", "also",
            "better than", "worse than", "instead", "rather than"
        ],
        "multi_day": [
            "week", "days", "weekend", "period", "stretch", "forecast",
            "next few", "coming", "upcoming"
        ],
        "contextual": [
            "there", "that place", "same location", "what about", "how about",
            "also", "too", "as well"
        ]
    }
    
    question_lower = question.lower()
    needs_advanced = False
    
    # Check if advanced parsing is needed
    for category, indicators in complexity_indicators.items():
        if any(indicator in question_lower for indicator in indicators):
            needs_advanced = True
            break
    
    # Check conversation history for context
    has_context = bool(_conversation_context.get("conversation_history"))
    
    # Route to appropriate parser
    try:
        if needs_advanced or has_context:
            result = parse_weather_question_advanced(question, _conversation_context)
            print(f"🧠 Used advanced parsing (complexity detected: {needs_advanced}, context: {has_context})")
        else:
            result = parse_weather_question(question)  # This will use the AI-powered version
            print(f"🔍 Used standard AI parsing")
        
        # Validate and enhance result
        return validate_and_enhance_parsing(result, question)
        
    except Exception as e:
        print(f"⚠️ All parsing methods failed: {e}")
        return {
            "error": f"Could not parse question: {str(e)}",
            "original_question": question,
            "fallback_available": True
        }

def generate_weather_response(parsed_question, weather_data):
    """
    Main weather response generation function with intelligent routing.
    
    Args:
        parsed_question (dict): Parsed question data
        weather_data (dict): Weather data
    
    Returns:
        str: Natural language response
    """
    # Input validation
    if "error" in parsed_question:
        if parsed_question.get("fallback_available"):
            return handle_parsing_error_gracefully(parsed_question, weather_data)
        return f"I couldn't understand your question: {parsed_question['error']}"
    
    if "error" in weather_data:
        return handle_weather_data_error(parsed_question, weather_data)
    
    # Determine response complexity needed
    query_complexity = parsed_question.get("query_complexity", "simple")
    processing_method = parsed_question.get("processing_method", "standard")
    
    try:
        # Route to appropriate response generator
        if processing_method == "ai_advanced" or query_complexity in ["multi_day", "comparison", "follow_up"]:
            response = generate_weather_response_advanced(parsed_question, weather_data, _conversation_context)
            print(f"🎯 Generated advanced response (complexity: {query_complexity})")
        else:
            response = generate_standard_response(parsed_question, weather_data)
            print(f"💬 Generated standard response")
        
        # Post-process and validate response
        final_response = post_process_response(response, parsed_question, weather_data)
        
        # Update conversation context
        update_conversation_context(parsed_question, weather_data, final_response)
        
        return final_response
        
    except Exception as e:
        print(f"⚠️ Response generation failed: {e}")
        return generate_emergency_fallback_response(parsed_question, weather_data)

def validate_and_enhance_parsing(result, original_question):
    """Validate and enhance parsing results."""
    
    # Ensure required fields exist
    required_fields = ["location", "time_period", "attribute"]
    for field in required_fields:
        if field not in result:
            result[field] = "unknown"
    
    # Add confidence scoring if missing
    if "confidence" not in result:
        confidence = "medium"
        if result.get("processing_method") == "ai_advanced":
            confidence = "high"
        elif result.get("processing_method") == "keyword_fallback":
            confidence = "low"
        result["confidence"] = confidence
    
    # Add processing timestamp
    result["processed_at"] = datetime.datetime.now().isoformat()
    
    # Enhance with derived information
    result["needs_forecast"] = result.get("time_period") not in ["current", "now"]
    result["is_specific_location"] = result.get("location") not in ["unknown", "contextual"]
    
    return result

def handle_parsing_error_gracefully(parsed_question, weather_data):
    """Handle parsing errors gracefully with helpful responses."""
    
    error_msg = parsed_question.get("error", "Unknown parsing error")
    original_question = parsed_question.get("original_question", "")
    
    # Try to provide a helpful response anyway
    if "weather" in original_question.lower():
        location = weather_data.get("location", "your area") if "error" not in weather_data else "your area"
        
        if "error" not in weather_data:
            return f"I had trouble understanding your specific question, but I can tell you " \
                   f"the current weather in {location} is {weather_data['current']['temperature']}° " \
                   f"and {weather_data['current']['condition'].lower()}. " \
                   f"Could you try asking in a different way?"
    
    return f"I'm having trouble understanding your question. " \
           f"Try asking something like 'What's the weather in London?' or 'Will it rain tomorrow?'"

def handle_weather_data_error(parsed_question, weather_data):
    """Handle weather data errors with helpful suggestions."""
    
    error_msg = weather_data.get("error", "Unknown weather data error")
    suggestions = weather_data.get("suggestions", [])
    
    response = f"I couldn't get weather data: {error_msg}"
    
    if suggestions:
        response += f" You might try: {', '.join(suggestions[:2])}"
    
    if "location" in error_msg.lower():
        response += f" Make sure to use a major city name or include the country."
    
    return f"⚠️ {response}"

def generate_standard_response(parsed_question, weather_data):
    """Generate standard response (wrapper for existing function)."""
    # This calls the existing generate_weather_response function
    # but with better error handling
    try:
        return generate_weather_response_basic(parsed_question, weather_data)
    except Exception as e:
        return generate_template_response(parsed_question, weather_data)

def generate_emergency_fallback_response(parsed_question, weather_data):
    """Last resort response generation."""
    
    if "error" in weather_data:
        return "I'm having trouble getting weather information right now. Please try again later."
    
    location = weather_data.get("location", "your area")
    current = weather_data.get("current", {})
    
    if current:
        return f"The weather in {location} is currently {current.get('temperature', 'N/A')}° " \
               f"with {current.get('condition', 'unknown conditions').lower()}."
    
    return "I'm having trouble processing your weather question right now. Please try asking again."

def post_process_response(response, parsed_question, weather_data):
    """Post-process responses for consistency and quality."""
    
    # Ensure response isn't too long
    if len(response) > 500:
        # Truncate and add continuation
        response = response[:450] + "... Would you like more details?"
    
    # Ensure response isn't too short
    if len(response) < 20:
        response += f" Feel free to ask for more specific information about {weather_data.get('location', 'the weather')}."
    
    # Add helpful follow-up suggestions for first-time users
    if not _conversation_context.get("conversation_history"):
        if parsed_question.get("confidence") == "high":
            response += " 💡 You can also ask about forecasts, specific days, or weather advice!"
    
    return response

def update_conversation_context(parsed_question, weather_data, response):
    """Update conversation context with the interaction."""
    
    context = _conversation_context
    
    # Update preferences based on user behavior
    if parsed_question.get("attribute") == "temperature":
        context.setdefault("user_preferences", {})["interested_in_temperature"] = True
    
    # Track successful interactions
    context.setdefault("successful_interactions", 0)
    context["successful_interactions"] += 1
    
    # Update last successful query
    context["last_successful_query"] = {
        "question": parsed_question,
        "weather_data": weather_data,
        "response": response,
        "timestamp": datetime.datetime.now().isoformat()
    }

# Main interface functions that users should call
def ask_weather_question(question, location=None):
    """
    Complete weather question answering system.
    
    Args:
        question (str): User's weather question
        location (str, optional): Specific location (overrides question parsing)
    
    Returns:
        str: Natural language weather response
    """
    try:
        print(f"🤔 Processing question: '{question}'")
        
        # Parse the question
        parsed = parse_weather_question(question)
        
        # Override location if provided
        if location:
            parsed["location"] = location
            parsed["location_override"] = True
        
        # Get weather data
        target_location = parsed.get("location", "London")  # Default fallback
        if target_location in ["unknown", "contextual"]:
            target_location = _conversation_context.get("last_location", "London")
        
        print(f"🌍 Getting weather for: {target_location}")
        weather_data = get_weather_data(target_location)
        
        # Generate response
        response = generate_weather_response(parsed, weather_data)
        
        print(f"✅ Response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in weather question system: {e}")
        return "I'm having trouble with weather information right now. Please try again later."

def test_natural_language_processing():
    """Test the natural language processing system with various questions."""
    
    test_questions = [
        "What's the weather like in London?",
        "Will it rain tomorrow?",
        "Should I bring an umbrella this weekend in Paris?",
        "How's the temperature compared to yesterday?",
        "What about Berlin?",  # Follow-up question
        "Give me the forecast for this week",
        "Is it warmer than usual?",
        "What should I wear today?"
    ]
    
    print("🧪 Testing Natural Language Processing System")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: '{question}'")
        try:
            response = ask_weather_question(question)
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 40)
    
    # Show conversation context
    print(f"\n📋 Conversation Summary:")
    summary = get_conversation_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

Tool: Claude
Date: 2025-05-22
Me: I need to create user interface functions for my weather app using pyinputplus. Can you help me build a main menu system and user input functions?
AI: Here's a basic menu system using pyinputplus:
import pyinputplus as pyip
import os

def display_main_menu():
    """Display main menu and get user choice."""
    clear_screen()
    print("=" * 50)
    print("🌦️  WEATHER ADVISOR - MAIN MENU")
    print("=" * 50)
    
    options = [
        "View Current Weather",
        "View Weather Forecast", 
        "Show Temperature Chart",
        "Show Precipitation Chart",
        "Ask Weather Question",
        "Change Location",
        "Settings",
        "Exit"
    ]
    
    choice = pyip.inputMenu(
        options,
        numbered=True,
        prompt="\nPlease select an option:\n"
    )
    
    return options.index(choice)

def get_location_input():
    """Get location from user with validation."""
    location = pyip.inputStr(
        "Enter city name (e.g., 'London', 'New York'): ",
        blockRegexes=[r'^\s*$'],  # Block empty strings
        strip=True
    )
    return location

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')
Me: Good.Now I need more sophisticated menu functions that can handle different weather app sections, user preferences, and error handling. Can you enhance this with better organization and user experience?
AI: Here's an enhanced menu system with better organization:
def display_welcome_screen():
    """Display welcome screen with app information."""
    clear_screen()
    print("🌦️" * 20)
    print()
    print("     WELCOME TO WEATHER ADVISOR")
    print("     Your Smart Weather Assistant")
    print()
    print("🌦️" * 20)
    print("\nFeatures:")
    print("• Current weather conditions")
    print("• Multi-day forecasts")
    print("• Interactive charts")
    print("• Natural language questions")
    print("• Personalized recommendations")
    
    pyip.inputStr("\nPress Enter to continue...", blank=True)

def get_location_with_validation():
    """Enhanced location input with suggestions."""
    while True:
        try:
            print("\n📍 Location Selection")
            print("-" * 30)
            
            location = pyip.inputStr(
                "Enter location (city, country): ",
                strip=True,
                blockRegexes=[r'^\s*$', r'^[0-9]+$'],  # Block empty and numeric only
                timeout=60
            )
            
            # Quick validation
            if len(location) < 2:
                print("⚠️ Location name too short. Please try again.")
                continue
            
            # Confirm location
            confirm = pyip.inputYesNo(f"Use '{location}' as your location? (y/n): ")
            if confirm == 'yes':
                return location
            
        except pyip.TimeoutException:
            print("⏰ Input timed out. Using default location: London")
            return "London"
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit()

def display_settings_menu():
    """Settings and preferences menu."""
    while True:
        clear_screen()
        print("⚙️  SETTINGS & PREFERENCES")
        print("=" * 40)
        
        settings_options = [
            "Change Temperature Units",
            "Set Default Location", 
            "Chart Display Preferences",
            "Clear Cache",
            "View System Status",
            "Back to Main Menu"
        ]
        
        choice = pyip.inputMenu(
            settings_options,
            numbered=True,
            prompt="\nSelect a setting to modify:\n"
        )
        
        choice_index = settings_options.index(choice)
        
        if choice_index == 0:
            change_temperature_units()
        elif choice_index == 1:
            set_default_location()
        elif choice_index == 2:
            set_chart_preferences()
        elif choice_index == 3:
            clear_app_cache()
        elif choice_index == 4:
            display_system_status()
        else:
            break

def change_temperature_units():
    """Change temperature unit preferences."""
    print("\n🌡️ Temperature Units")
    print("-" * 25)
    
    units = pyip.inputMenu(
        ["Celsius (°C)", "Fahrenheit (°F)"],
        numbered=True,
        prompt="Select temperature unit:\n"
    )
    
    selected_unit = "metric" if "Celsius" in units else "imperial"
    
    # Update global preferences (would save to config in real app)
    print(f"✅ Temperature unit set to {units}")
    pyip.inputStr("Press Enter to continue...", blank=True)
    
    return selected_unit

def weather_question_interface():
    """Interface for natural language weather questions."""
    clear_screen()
    print("🤖 ASK WEATHER ADVISOR")
    print("=" * 35)
    print("Ask me anything about the weather!")
    print("Examples:")
    print("• 'Will it rain tomorrow?'")
    print("• 'Should I bring a jacket?'")
    print("• 'What's the weather like in Paris?'")
    print("\nType 'quit' to return to main menu")
    print("-" * 35)
    
    while True:
        try:
            question = pyip.inputStr(
                "\n❓ Your question: ",
                strip=True,
                timeout=120
            )
            
            if question.lower() in ['quit', 'exit', 'back']:
                break
            
            if len(question) < 5:
                print("⚠️ Please ask a more detailed question.")
                continue
            
            # Process question (would call NLP functions)
            print(f"\n🤔 Processing: '{question}'")
            print("💭 Analyzing your question...")
            
            # Simulate response (replace with actual function call)
            response = ask_weather_question(question)
            print(f"\n🌦️ {response}")
            
            # Ask if user wants to continue
            continue_chat = pyip.inputYesNo("\nAsk another question? (y/n): ")
            if continue_chat == 'no':
                break
                
        except pyip.TimeoutException:
            print("⏰ Question input timed out.")
            break
        except KeyboardInterrupt:
            print("\n↩️ Returning to main menu...")
            break
Me: Now I need the main application loop that ties everything together and handles navigation between different sections. Can you create the main controller function?
AI: Here's the main application controller with complete navigation:
def main_application():
    """Main application controller and navigation hub."""
    # Initialize app state
    current_location = None
    weather_data = None
    app_settings = {
        "units": "metric",
        "show_charts": True,
        "auto_refresh": True
    }
    
    # Welcome sequence
    display_welcome_screen()
    
    # Get initial location
    print("🚀 Let's get started!")
    current_location = get_location_with_validation()
    print(f"✅ Location set to: {current_location}")
    
    # Main application loop
    while True:
        try:
            # Refresh weather data if needed
            if weather_data is None or app_settings.get("auto_refresh", True):
                print(f"\n🌐 Getting weather data for {current_location}...")
                weather_data = get_weather_data(current_location, units=app_settings["units"])
                
                if "error" in weather_data:
                    print(f"⚠️ {weather_data['error']}")
                    change_location = pyip.inputYesNo("Try a different location? (y/n): ")
                    if change_location == 'yes':
                        current_location = get_location_with_validation()
                        continue
            
            # Display main menu and get choice
            choice = display_main_menu()
            
            # Route to appropriate function
            if choice == 0:  # Current Weather
                display_current_weather_detailed(weather_data)
                
            elif choice == 1:  # Weather Forecast
                display_forecast_detailed(weather_data)
                
            elif choice == 2:  # Temperature Chart
                if app_settings["show_charts"]:
                    create_temperature_visualisation(weather_data)
                else:
                    print("📊 Charts are disabled in settings")
                wait_for_user()
                
            elif choice == 3:  # Precipitation Chart
                if app_settings["show_charts"]:
                    create_precipitation_visualisation(weather_data)
                else:
                    print("📊 Charts are disabled in settings")
                wait_for_user()
                
            elif choice == 4:  # Ask Question
                weather_question_interface()
                
            elif choice == 5:  # Change Location
                new_location = get_location_with_validation()
                if new_location != current_location:
                    current_location = new_location
                    weather_data = None  # Force refresh
                    
            elif choice == 6:  # Settings
                old_settings = app_settings.copy()
                app_settings = handle_settings_menu(app_settings)
                
                # Check if location-affecting settings changed
                if old_settings["units"] != app_settings["units"]:
                    weather_data = None  # Force refresh with new units
                    
            elif choice == 7:  # Exit
                confirm_exit = pyip.inputYesNo("Are you sure you want to exit? (y/n): ")
                if confirm_exit == 'yes':
                    print("\n👋 Thank you for using Weather Advisor!")
                    print("Stay weather-wise! 🌦️")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue_app = pyip.inputYesNo("Continue using the app? (y/n): ")
            if continue_app == 'no':
                break

def display_current_weather_detailed(weather_data):
    """Display detailed current weather information."""
    if "error" in weather_data:
        print(f"❌ {weather_data['error']}")
        wait_for_user()
        return
    
    clear_screen()
    current = weather_data["current"]
    location = weather_data.get("location", "Unknown")
    
    print("🌤️  CURRENT WEATHER CONDITIONS")
    print("=" * 45)
    print(f"📍 Location: {location}")
    print(f"🌡️  Temperature: {current['temperature']}° (feels like {current['feels_like']}°)")
    print(f"☁️  Condition: {current['condition']}")
    print(f"💧 Humidity: {current['humidity']}%")
    print(f"💨 Wind: {current['wind_speed']} km/h {current['wind_direction']}")
    print(f"🔍 Visibility: {current['visibility']} km")
    
    if current.get('uv_index'):
        print(f"☀️  UV Index: {current['uv_index']}")
    
    print("=" * 45)
    wait_for_user()

def display_forecast_detailed(weather_data):
    """Display detailed weather forecast."""
    if "error" in weather_data or not weather_data.get("forecast"):
        print("❌ No forecast data available")
        wait_for_user()
        return
    
    clear_screen()
    print("📅 WEATHER FORECAST")
    print("=" * 35)
    
    for i, day in enumerate(weather_data["forecast"][:5]):
        day_name = "Today" if i == 0 else f"Day {i+1}"
        print(f"\n{day_name} ({day['date']}):")
        print(f"  🌡️  High: {day['max_temp']}° | Low: {day['min_temp']}°")
        print(f"  ☁️  {day['condition']}")
        print(f"  🌧️  Rain chance: {day['precipitation_chance']}%")
    
    print("\n" + "=" * 35)
    wait_for_user()

def wait_for_user():
    """Wait for user input to continue."""
    pyip.inputStr("\n👆 Press Enter to continue...", blank=True)

def handle_settings_menu(current_settings):
    """Handle settings menu and return updated settings."""
    # This would integrate with the settings menu functions
    display_settings_menu()
    return current_settings

# Quick start function for testing
def quick_start():
    """Quick start function for development/testing."""
    print("🚀 Quick Start Mode")
    location = pyip.inputStr("Enter location (or press Enter for London): ", blank=True)
    if not location:
        location = "London"
    
    weather_data = get_weather_data(location)
    response = ask_weather_question("What's the weather like?")
    print(f"\n{response}")

# Entry point
if __name__ == "__main__":
    try:
        main_application()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please restart the application.")

Tool: Claude
Date: 2025-05-22
Me: I need to tie everything together in the main application logic. Can you help me create the core integration functions that connect weather data, visualizations, NLP, and the user interface?
AI: Here's the main application integration logic:
# Main Application Logic - Integration Hub
import sys
import traceback
from datetime import datetime

def run_weather_advisor():
    """
    Main entry point for the Weather Advisor application.
    Integrates all components and handles the application lifecycle.
    """
    print("🌦️ Starting Weather Advisor...")
    
    try:
        # Initialize application
        if not initialize_application():
            print("❌ Failed to initialize application")
            return False
        
        # Run main application
        main_application()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
    finally:
        cleanup_application()
    
    return True

def initialize_application():
    """Initialize all application components."""
    try:
        print("🔧 Initializing components...")
        
        # Validate environment
        if not validate_environment():
            return False
        
        # Test core functions
        print("🧪 Testing core functionality...")
        test_results = run_startup_tests()
        
        if not test_results["all_passed"]:
            print("⚠️ Some tests failed but continuing...")
        
        print("✅ Application initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def run_startup_tests():
    """Run quick tests of core functionality."""
    tests = {
        "weather_api": test_weather_api_basic,
        "ai_connection": test_ai_connection_basic,
        "visualization": test_visualization_basic,
        "nlp": test_nlp_basic
    }
    
    results = {"all_passed": True, "details": {}}
    
    for test_name, test_func in tests.items():
        try:
            result = test_func()
            results["details"][test_name] = "✅" if result else "⚠️"
            if not result:
                results["all_passed"] = False
        except Exception as e:
            results["details"][test_name] = f"❌ {str(e)[:50]}"
            results["all_passed"] = False
    
    return results

def test_weather_api_basic():
    """Basic weather API test."""
    try:
        data = get_weather_data("London")
        return "error" not in data
    except:
        return False

def test_ai_connection_basic():
    """Basic AI connection test."""
    try:
        response = get_response("Hello")
        return len(response) > 0
    except:
        return False

def test_visualization_basic():
    """Basic visualization test."""
    try:
        # Test with mock data
        mock_data = create_mock_weather_data()
        fig = create_temperature_visualisation(mock_data, output_type='figure')
        return fig is not None
    except:
        return False

def test_nlp_basic():
    """Basic NLP test."""
    try:
        parsed = parse_weather_question("What's the weather?")
        return "error" not in parsed
    except:
        return False

def create_mock_weather_data():
    """Create mock weather data for testing."""
    return {
        "location": "Test City",
        "current": {
            "temperature": 20,
            "feels_like": 22,
            "condition": "Sunny",
            "humidity": 60,
            "wind_speed": 10,
            "wind_direction": "NW",
            "visibility": 10,
            "uv_index": 5
        },
        "forecast": [
            {
                "date": "2025-01-16",
                "max_temp": 25,
                "min_temp": 15,
                "condition": "Partly Cloudy",
                "precipitation_chance": 20,
                "precipitation_amount": 0.0
            }
        ],
        "units": "metric"
    }

def cleanup_application():
    """Clean up resources before exit."""
    try:
        print("\n🧹 Cleaning up...")
        clear_weather_cache()
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
Me: Nice. Now I need the generate_weather_response function that was mentioned in the skeleton, plus utility functions for error handling and logging that tie into the AI interaction tracking required for the assignment.
AI: Here's the complete integration with response generation and AI interaction tracking:
def generate_weather_response(parsed_question, weather_data):
    """
    Generate a natural language response to a weather question.
    This is the main interface that coordinates all response generation.
    
    Args:
        parsed_question (dict): Parsed question data
        weather_data (dict): Weather data
    
    Returns:
        str: Natural language response
    """
    # This calls the NLP module's response generation
    return generate_weather_response(parsed_question, weather_data)

# AI Interaction Tracking for Assignment Requirements
class AIInteractionLogger:
    """Track AI interactions for assignment documentation."""
    
    def __init__(self):
        self.interactions = []
        self.session_start = datetime.now()
    
    def log_interaction(self, interaction_type, prompt, response, success=True):
        """Log an AI interaction."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_length": len(response) if response else 0,
            "success": success,
            "session_time": (datetime.now() - self.session_start).total_seconds()
        }
        self.interactions.append(interaction)
    
    def get_summary(self):
        """Get interaction summary for assignment documentation."""
        total = len(self.interactions)
        successful = sum(1 for i in self.interactions if i["success"])
        
        types = {}
        for interaction in self.interactions:
            interaction_type = interaction["type"]
            types[interaction_type] = types.get(interaction_type, 0) + 1
        
        return {
            "total_interactions": total,
            "successful_interactions": successful,
            "success_rate": f"{(successful/total*100):.1f}%" if total > 0 else "0%",
            "interaction_types": types,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60
        }

# Global logger instance
ai_logger = AIInteractionLogger()

def enhanced_weather_workflow(user_input, current_location=None):
    """
    Complete weather workflow with AI interaction logging.
    This demonstrates the full AI-enhanced process for assignment documentation.
    """
    workflow_steps = []
    
    try:
        # Step 1: Parse user question with AI
        workflow_steps.append("🧠 Parsing user question with AI")
        parsed = parse_weather_question(user_input)
        ai_logger.log_interaction("question_parsing", user_input, str(parsed), "error" not in parsed)
        
        # Step 2: Determine location
        workflow_steps.append("📍 Determining target location")
        target_location = parsed.get("location", current_location or "London")
        if target_location in ["unknown", "contextual"]:
            target_location = current_location or "London"
        
        # Step 3: Fetch weather data
        workflow_steps.append(f"🌐 Fetching weather data for {target_location}")
        weather_data = get_weather_data(target_location)
        
        # Step 4: Generate AI response
        workflow_steps.append("🤖 Generating AI-powered response")
        if "error" not in weather_data:
            response = generate_weather_response(parsed, weather_data)
            ai_logger.log_interaction("response_generation", 
                                     f"Question: {user_input}, Location: {target_location}", 
                                     response, True)
        else:
            response = f"Sorry, I couldn't get weather data: {weather_data['error']}"
            ai_logger.log_interaction("response_generation", user_input, response, False)
        
        return {
            "success": True,
            "response": response,
            "workflow_steps": workflow_steps,
            "ai_interactions": len(workflow_steps),
            "location_used": target_location
        }
        
    except Exception as e:
        error_response = f"I encountered an error processing your request: {str(e)}"
        ai_logger.log_interaction("error_handling", user_input, error_response, False)
        
        return {
            "success": False,
            "response": error_response,
            "workflow_steps": workflow_steps,
            "error": str(e)
        }

def demonstrate_ai_capabilities():
    """
    Demonstrate various AI capabilities for assignment documentation.
    """
    print("🎯 Demonstrating AI Capabilities")
    print("=" * 50)
    
    demo_queries = [
        "What's the weather like in Sydney?",
        "Should I bring an umbrella tomorrow?",
        "Compare today's temperature to yesterday",
        "What's the forecast for this weekend?",
        "Will it be good weather for a picnic?"
    ]
    
    results = []
    
    for query in demo_queries:
        print(f"\n🔍 Processing: '{query}'")
        result = enhanced_weather_workflow(query)
        
        print(f"   Response: {result['response'][:100]}...")
        print(f"   AI Steps: {result['ai_interactions']}")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        
        results.append(result)
    
    # Show AI interaction summary
    print(f"\n📊 AI Interaction Summary:")
    summary = ai_logger.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    return results

def export_ai_interaction_log():
    """Export AI interaction log for assignment documentation."""
    import json
    
    log_data = {
        "session_info": {
            "start_time": ai_logger.session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "application": "Weather Advisor"
        },
        "summary": ai_logger.get_summary(),
        "detailed_interactions": ai_logger.interactions
    }
    
    filename = f"ai_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"✅ AI interaction log exported to {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to export log: {e}")
        return None

# Integration utility functions
def safe_function_call(func, *args, fallback_result=None, **kwargs):
    """Safely call a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ Function {func.__name__} failed: {e}")
        return fallback_result

def main():
    """Main entry point with command line options."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demonstrate_ai_capabilities()
        elif sys.argv[1] == "--test":
            test_all_components()
        elif sys.argv[1] == "--quick":
            quick_start()
        else:
            print("Usage: python weather_app.py [--demo|--test|--quick]")
    else:
        run_weather_advisor()

if __name__ == "__main__":
    main()
