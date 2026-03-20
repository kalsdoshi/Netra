import React from 'react';
import '../styles/Graph.css';

export default function GraphControls({ filters, setFilters, onRebuild, loading }) {
  return (
    <div className="graph-controls app-panel">
      <h3>Graph Explorer</h3>
      
      <div className="control-group">
        <label>Minimum Edge Weight (Similarity)</label>
        <div className="range-wrap">
          <input 
            type="range" 
            min="0" max="1" step="0.05" 
            value={filters.minWeight} 
            onChange={(e) => setFilters({...filters, minWeight: parseFloat(e.target.value)})} 
          />
          <span className="range-badge">{filters.minWeight.toFixed(2)}</span>
        </div>
      </div>

      <div className="control-group">
        <label>Search Nodes (Person/Object)</label>
        <input 
          type="text" 
          placeholder="e.g. person_1 or car" 
          value={filters.search} 
          onChange={(e) => setFilters({...filters, search: e.target.value})} 
        />
      </div>

      <div className="control-group">
        <label>Show Edge Modality</label>
        <select 
          value={filters.modality} 
          onChange={(e) => setFilters({...filters, modality: e.target.value})}
        >
          <option value="all">All Connections</option>
          <option value="face">Face Similarity Only</option>
          <option value="object">Object Similarity Only</option>
        </select>
      </div>

      <button className="cta-btn rebuild-btn" disabled={loading} onClick={onRebuild}>
        {loading ? "Rebuilding..." : "⚡ Rebuild Graph Database"}
      </button>
    </div>
  );
}
