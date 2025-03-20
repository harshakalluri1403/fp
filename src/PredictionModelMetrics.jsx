import { useState, useEffect } from 'react';

export default function PredictionModelMetrics({ submitted }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!submitted) return;

    const fetchMetrics = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch('http://localhost:5000/metrics');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log('Metrics data:', data);
        setMetrics(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [submitted]);

  if (loading) return <div className="model-metrics">Loading metrics...</div>;
  if (error) return <div className="model-metrics">Error: {error}</div>;

  return (
    <div className="model-metrics">
      <h3>Prediction Model Metrics</h3>
      {metrics ? (
        <p>
          The model trained on March 19, 2025, for 50 epochs, achieving a final validation loss of{' '}
          {metrics.final_val_loss.toFixed(6)}, MSE of {metrics.mse.toFixed(6)}, RMSE of{' '}
          {metrics.rmse.toFixed(6)}, MAE of {metrics.mae.toFixed(6)}, and an R² score of{' '}
          {metrics.r2_score.toFixed(4)}.
        </p>
      ) : (
        <p>
          The model trained on March 19, 2025, for 50 epochs, achieving a final validation loss of 0.014597, MSE of 0.014596, RMSE of 0.120814, MAE of 0.066132, and an R² score of 0.5915.
        </p>
      )}
    </div>
  );
}