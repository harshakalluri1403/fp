import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for Leaflet marker icons
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Component to handle auto-zoom
function MapBounds({ pathCoords }) {
  const map = useMap();

  useEffect(() => {
    if (pathCoords.length >= 2) {
      const bounds = L.latLngBounds(pathCoords);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 }); // Adjust padding and maxZoom as needed
      console.log('Map bounds set:', bounds);
    }
  }, [map, pathCoords]);

  return null;
}

export default function MapComponent({ from, to }) {
  const [pathCoords, setPathCoords] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('Fetching map data for:', { from, to });
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
        console.log('Map path data:', data);
        if (data.error) {
          setError(data.error);
        } else {
          const coords = data.alternative_paths[0].coordinates.map(coord => [
            coord.latitude,
            coord.longitude
          ]);
          console.log('Path coordinates:', coords);
          const validCoords = coords.filter(coord => 
            coord[0] !== 0.0 && coord[1] !== 0.0 && !isNaN(coord[0]) && !isNaN(coord[1])
          );
          console.log('Valid coordinates:', validCoords);
          if (validCoords.length < 2) {
            setError('Invalid coordinates for mapping');
          } else {
            setPathCoords(validCoords);
          }
        }
      })
      .catch(err => {
        console.error('Fetch error:', err);
        setError(err.message);
      });
  }, [from, to]);

  if (error) return <div>Error: {error}</div>;
  if (pathCoords.length < 2) {
    console.log('Not enough valid coordinates to render map:', pathCoords);
    return <div>Loading map or invalid coordinates...</div>;
  }

  // Initial center (will be adjusted by MapBounds)
  const initialCenter = [
    pathCoords.reduce((sum, coord) => sum + coord[0], 0) / pathCoords.length,
    pathCoords.reduce((sum, coord) => sum + coord[1], 0) / pathCoords.length,
  ];
  console.log('Initial map center:', initialCenter);

  return (
    <div style={{ height: '70vh', border: '1px solid #ccc' }}>
      <MapContainer
        center={initialCenter}
        zoom={13} // Initial zoom, will be overridden by fitBounds
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {pathCoords.map((coord, idx) => (
          <Marker key={idx} position={coord}>
            <Popup>{idx === 0 ? from : idx === pathCoords.length - 1 ? to : `Node ${idx}`}</Popup>
          </Marker>
        ))}
        <Polyline positions={pathCoords} color="blue" weight={5} opacity={0.8} />
        <MapBounds pathCoords={pathCoords} />
      </MapContainer>
    </div>
  );
}