import React, { useState, useEffect } from 'react';

export default function WeatherComponent({ from, to }) {
  const [weather, setWeather] = useState(null);
  const [cityName, setCityName] = useState(null); // Store the city name
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setCityName(null); // Reset city name on new fetch
    console.log('Fetching prediction for weather:', { from, to });
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from, to }),
    })
      .then(response => {
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        return response.json();
      })
      .then(data => {
        console.log('Prediction data for weather:', data);
        if (data.error) {
          setError(data.error);
          setLoading(false);
        } else {
          setWeather(data.weather_conditions); // Use weather_conditions from response

          // Get coordinates of the 'from' detector (first in the path)
          const coords = data.alternative_paths[0].coordinates[0]; // Coordinates for 'from' detector
          const lat = coords.latitude;
          const lon = coords.longitude;
          console.log('Coordinates for reverse geocoding:', { lat, lon });

          // Fetch city name using Nominatim
          fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`, {
            headers: {
              'User-Agent': 'TrafficPredictionApp/1.0 (your-email@example.com)', // Replace with your app name and email
            },
          })
            .then(res => res.json())
            .then(geoData => {
              console.log('Reverse geocoding result:', geoData);
              if (geoData && geoData.address) {
                // Extract city name (can vary: city, town, village, etc.)
                const city = geoData.address.city || geoData.address.town || geoData.address.village || 'Unknown City';
                setCityName(city);
              } else {
                setCityName('Unknown City');
              }
              setLoading(false);
            })
            .catch(err => {
              console.error('Reverse geocoding error:', err);
              setCityName('Unknown City');
              setLoading(false);
            });
        }
      })
      .catch(err => {
        console.error('Fetch error:', err);
        setError(err.message);
        setLoading(false);
      });
  }, [from, to]);

  const getWeatherEmoji = (weather) => {
    if (!weather || !weather.description) return '🌡️';

    const condition = weather.description.toLowerCase();
    switch (condition) {
      case 'clear': return '☀️';
      case 'clouds': return '☁️';
      case 'rain': return '🌧️';
      case 'drizzle': return '🌦️';
      case 'thunderstorm': return '⛈️';
      case 'snow': return '❄️';
      case 'mist':
      case 'fog':
      case 'haze': return '🌫️';
      case 'fair': return '☀️';
      default: return '🌡️';
    }
  };

  const getWeatherInfo = (weather) => {
    if (!weather) return null;

    const emoji = getWeatherEmoji(weather);
    const temp = Math.round(weather.temperature);
    const feelsLike = Math.round(weather.temperature); // No feels_like in data, use temp
    const description = weather.description;
    const humidity = weather.humidity;
    const windSpeed = weather.wind_speed;
    const pressure = weather.pressure || 1013; // Default if not provided
    const visibility = weather.visibility || 10; // Default if not provided

    return (
      <div className="weather-info">
        <div className="weather-main">
          <span className="weather-emoji">{emoji}</span>
          <span className="weather-temp">{temp}°C</span>
        </div>
        <div className="weather-description">
          {description.charAt(0).toUpperCase() + description.slice(1)}
        </div>
        <div className="weather-details">
          <div>💧 Humidity: {humidity}%</div>
          <div>💨 Wind: {windSpeed.toFixed(1)} km/h</div>
          <div>🌡️ Feels like: {feelsLike}°C</div>
          <div>🌀 Pressure: {pressure} hPa</div>
          <div>🔭 Visibility: {visibility} km</div>
        </div>
      </div>
    );
  };

  if (loading) return <div className="weather-container">Loading weather...</div>;
  if (error) return <div className="weather-container">Error: {error}</div>;

  return (
    <div className="weather-container">
      <div className="weather-card">
        <h3>Weather at {cityName || from} {getWeatherEmoji(weather)}</h3>
        {getWeatherInfo(weather)}
      </div>
    </div>
  );
}