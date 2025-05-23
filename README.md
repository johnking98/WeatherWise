Weather Advisor
A smart weather application that combines real-time weather data with AI-powered natural language processing for intuitive weather queries.
Features

Current Weather & Forecasts: Real-time weather data for any location
Interactive Charts: Temperature and precipitation visualizations
Natural Language Queries: Ask questions like "Will it rain tomorrow?"
Smart Location Search: Intelligent location validation with suggestions
Multiple Data Sources: Fallback strategies for reliable data access
User-Friendly Interface: Clean console-based menu system

Requirements

Python 3.8 or higher
Internet connection for weather data
Terminal/Command prompt

Installation

Clone or download the project files
Install required packages:

pip install fetch-my-weather hands-on-ai matplotlib pyinputplus
Quick Start

Run the main application:

python main_application.py

Follow the setup prompts to configure your location
Explore features through the main menu

Project Structure
weather-advisor/
├── setup_configuration.py    # Environment setup
├── weather_data_functions.py  # Data retrieval & processing
├── visualization_functions.py # Charts & graphs
├── nlp_functions.py          # AI question processing
├── user_interface.py         # Menu & user interaction
├── main_application.py       # Main app controller
└── testing_examples.py       # Tests & demonstrations
Main Components
🔧 Setup & Configuration

Environment variable management
Package validation
Connection testing

🌡️ Weather Data Functions

Location-based weather retrieval
Multi-day forecasts
Data caching with fallback strategies

📊 Visualization Functions

Temperature trend charts
Precipitation probability graphs
Professional styling options

🤖 Natural Language Processing

AI-powered question parsing
Context-aware responses
Conversation memory

🧭 User Interface

Interactive menu system
Input validation
Settings management

🔄 Main Application Logic

Component integration
Error handling
AI interaction logging

Usage Examples
Basic Weather Query
Enter location: Sydney, Australia
Temperature: 22°C (feels like 24°C)
Condition: Partly Cloudy
Natural Language Questions

"What's the weather like in London?"
"Should I bring an umbrella tomorrow?"
"Will it be warm this weekend?"

Chart Generation

Temperature trends over multiple days
Precipitation probability charts
Customizable styling options

Configuration
The application uses these environment variables:

AILABKIT_SERVER: AI service endpoint
AILABKIT_MODEL: AI model name
AILABKIT_API_KEY: Authentication key

Error Handling
The application includes robust error handling for:

Invalid locations
Network connectivity issues
API service unavailability
Missing data scenarios

Testing
Run comprehensive tests:
python testing_examples.py
Troubleshooting
Location Not Found

Try including country name (e.g., "Perth, Australia")
Use major city names
Check spelling

API Connection Issues

Verify internet connection
Check environment variables
Try again later if service is busy

Missing Features

Ensure all packages are installed
Some features require AI service availability
Fallback modes activate automatically

Development Notes
This project demonstrates effective AI collaboration techniques:

Intentional prompting strategies
Iterative problem solving
Comprehensive error handling
Professional code organization

License
Educational project - free to use and modify.
Support
For issues or questions, refer to the troubleshooting section or check component-specific error messages for detailed guidance.

Built with AI assistance - This project showcases human-AI collaboration in software development.