// frontend/src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ID Document Classifier</h1>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-input">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="image-upload"
            />
            <label htmlFor="image-upload" className="upload-button">
              Select Image
            </label>
            {selectedFile && (
              <span className="file-name">{selectedFile.name}</span>
            )}
          </div>

          {previewUrl && (
            <div className="preview-container">
              <img src={previewUrl} alt="Preview" className="image-preview" />
            </div>
          )}

          <button
            type="submit"
            disabled={!selectedFile || isLoading}
            className="submit-button"
          >
            {isLoading ? 'Processing...' : 'Classify Document'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {prediction && (
          <div className="results-container">
            <h2>Classification Results</h2>
            
            {prediction.llm_prediction && (
              <div className="prediction-card">
                <h3>LLM Classification</h3>
                <p className="prediction">
                  <strong>Document Type:</strong> {prediction.llm_prediction.class_name}
                </p>
                <p className="confidence">
                  <strong>Confidence:</strong> {prediction.llm_prediction.confidence.toFixed(1)}%
                </p>
              </div>
            )}
            
            {prediction.mobilenet_prediction && (
              <div className="prediction-card">
                <h3>MobileNet Classification</h3>
                <p className="prediction">
                  <strong>Document Type:</strong> {prediction.mobilenet_prediction.class_name}
                </p>
                <p className="confidence">
                  <strong>Confidence:</strong> {prediction.mobilenet_prediction.confidence.toFixed(1)}%
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
