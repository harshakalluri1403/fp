import React from 'react';

export default function FormComponent({ fromValue, setFromValue, toValue, setToValue, onSubmit }) {
  return (
    <form onSubmit={onSubmit}>
      <label htmlFor="from">From:</label>
      <input
        type="text"
        placeholder="Enter start detector (e.g., B1-3)"
        className="from"
        id="from"
        value={fromValue}
        onChange={(e) => setFromValue(e.target.value)}
        required
      />
      
      <label htmlFor="to">To:</label>
      <input
        type="text"
        placeholder="Enter destination detector (e.g., B1-5)"
        className="to"
        id="to"
        value={toValue}
        onChange={(e) => setToValue(e.target.value)}
        required
      />
      
      <label htmlFor="model">Model:</label>
      <select id="model" className="model-select">
        <option value="gat">GAT</option>
      </select>
      
      <button type="submit" className="submitBtn">Submit</button>
    </form>
  );
}