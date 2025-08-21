import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [features, setFeatures] = useState({});
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict/credit", {
        features,
      });
      setResult(res.data);
    } catch (err) {
      setResult({ error: err.response?.data?.detail || "Request failed" });
    }
  };

  return (
    
    <div className="container">
      <header className="app-header">
       <h1>🛡️ Fraud Detection Dashboard</h1>
       <p>Powered by FastAPI + React</p>
      </header>
      <h2>Credit Fraud Prediction</h2>
      <form onSubmit={handleSubmit}>
        {["Amount", "Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`)].map((key) => (
          <div key={key}>
            <label>{key}</label>
            <input type="number" name={key} step="any" onChange={handleChange} />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>
      {result && (
        <div className="result">
          <h3>Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;