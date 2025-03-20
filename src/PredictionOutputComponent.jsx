import React, { useState, useEffect } from 'react';

export default function PredictionOutputComponent({ from, to, submitted }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (submitted) {
      setLoading(true);
      setError(null);
      console.log('Fetching prediction for:', { from, to });
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from, to }),
      })
        .then(response => {
          console.log('Response status:', response.status, 'Headers:', response.headers);
          if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
          return response.json();
        })
        .then(data => {
          console.log('Prediction data:', data);
          if (data.error) {
            setError(data.error);
          } else {
            setPrediction(data.alternative_paths[0]);
          }
          setLoading(false);
        })
        .catch(err => {
          console.error('Fetch error details:', err.message, err.stack); // More detailed error
          setError(`Failed to fetch: ${err.message}`);
          setLoading(false);
        });
    }
  }, [submitted, from, to]);

  console.log('Render state:', { loading, error, prediction });

  if (loading) return <div>Loading prediction...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!submitted) return <div>Enter detectors to see prediction</div>;
  if (!prediction) return <div>No prediction data available</div>;

  return (
    <div className="prediction-output">
      <p>ETA: <strong>{prediction.eta_minutes.toFixed(1)} minutes</strong></p>
      <p className="path-info">Path: <strong>{prediction.path.join(' → ')}</strong></p>
      <p>Distance: <strong>{(prediction.distance / 1000).toFixed(2)} km</strong></p>
    </div>
  );
}