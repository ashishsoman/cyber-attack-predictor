import React, { FormEvent, FormEventHandler, useState } from "react";
import axios from "axios";
import { ProbabilityResult } from "./types/types";

function PredictionForm() {
  const [industry, setIndustry] = useState("Finance");
  const [country, setCountry] = useState("USA");
  const [cybersecurity,setCyberSecurity] = useState("0.5")
  const [result, setResult] = useState<ProbabilityResult>();

  const handleSubmit = async (e:FormEvent) => {
    e.preventDefault();
    const response = await axios.post("http://localhost:8000/predict", {
      industry,
      country, 
      cybersecurity
    });
    setResult(response.data);
  };

  return (
    <div>
      <form style={{
    display: 'flex',
    flexDirection: 'column',
    gap:'10px',
    gridGap: '10px',
    width: '100%'

      }} onSubmit={handleSubmit}>
        <div style={{ display: 'flex', gap:'10px' }}>Industry:
          <select style={{ flexGrow: '1' }} onChange={(e) => setIndustry(e.target.value)}>
            <option>Finance</option>
            <option>Energy</option>
            <option>Retail</option>
            <option>Tech</option>
            <option>Education</option>
          </select>
        </div>
        <div style={{ display: 'flex', gap:'10px' }} >Country:
          <input style={{ flexGrow:'1' }} value={country} onChange={(e) => setCountry(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap:'10px' }}>Cybersecurity posture *in float (0 - 1):
          <input type="number" value={cybersecurity} onChange={(e) => setCyberSecurity((e.target.value))} />
        </div>
        <button style={{backgroundColor:'#007AFF'}} type="submit">Predict</button>
      </form>
      {result && (
        <div style={{display:'flex', alignItems:'center', flexDirection:'column'}}>
          <h3>Prediction Result:</h3>
          <p>Attack Predicted: {result.attack_predicted ? 
            <span style={{color:'red',fontWeight:'bold'}}>Yes</span> :
            <span style={{color:'green',fontWeight:'bold'}}>No</span>
             }</p>
          <p>Probability: {result.probability.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;
