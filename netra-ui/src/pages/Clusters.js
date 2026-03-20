import React, { useEffect, useState } from "react";
import { getClusters, mergeClusters } from "../api/api";
import "../styles/Clusters.css";

export default function Clusters() {
  const [clusters, setClusters] = useState({});
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [stats, setStats] = useState({ total: 0, faces: 0 });

  useEffect(() => {
    fetchClusters();
  }, []);

  const fetchClusters = async () => {
    setLoading(true);
    try {
      const res = await getClusters();
      if (res.data.error) {
        console.error("Error:", res.data.error);
      } else {
        setClusters(res.data.clusters || {});
        const total = Object.keys(res.data.clusters || {}).length;
        const faces = Object.values(res.data.clusters || {}).reduce((sum, c) => sum + c.size, 0);
        setStats({ total, faces });
      }
    } catch (err) {
      console.error("Failed to fetch clusters:", err);
    } finally {
      setLoading(false);
    }
  };

  const toggleSelect = (id) => {
    if (selected.includes(id)) {
      setSelected(selected.filter((x) => x !== id));
    } else {
      if (selected.length < 2) {
        setSelected([...selected, id]);
      }
    }
  };

  const handleMerge = async () => {
    if (selected.length !== 2) return;
    try {
      await mergeClusters(selected[0], selected[1]);
      setSelected([]);
      await fetchClusters();
    } catch (err) {
      console.error("Merge failed:", err);
    }
  };

  const filteredClusters = Object.entries(clusters).filter(([id]) =>
    id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 🔥 DETAIL VIEW
  if (selectedCluster) {
    return (
      <div className="clusters-page">
        <button className="back-btn" onClick={() => setSelectedCluster(null)}>
          ← Back to Clusters
        </button>
        <div className="detail-header">
          <div className="detail-title">
            <h1>{selectedCluster.id}</h1>
            <span className="badge">{selectedCluster.faces.length} faces</span>
          </div>
        </div>
        <div className="gallery-grid">
          {selectedCluster.faces.map((face, i) => (
            <div key={i} className="gallery-item">
              <img
                src={`http://127.0.0.1:8000/images/${face.image}`}
                alt={`Face ${i}`}
                className="gallery-img"
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="clusters-page">
      {/* HEADER */}
      <header className="header-section">
        <div className="header-content">
          <div className="title-area">
            <h2 className="main-title">Identity Nexus</h2>
            <p className="subtitle">Consolidated gallery of mapped human entities</p>
          </div>
        </div>
      </header>

      {/* STATS */}
      <div className="stats-section">
        <div className="stat-card">
          <div className="stat-number">{stats.total}</div>
          <div className="stat-label">People</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.faces}</div>
          <div className="stat-label">Total Faces</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{selected.length}/2</div>
          <div className="stat-label">Selected</div>
        </div>
      </div>

      {/* SEARCH */}
      <div className="search-section">
        <input
          type="text"
          placeholder="🔍 Search clusters..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      {/* MERGE BUTTON */}
      {selected.length === 2 && (
        <div className="merge-action">
          <button className="merge-btn" onClick={handleMerge}>
            ⚡ MERGE {selected[0]} + {selected[1]}
          </button>
        </div>
      )}

      {/* CLUSTERS GRID */}
      {loading ? (
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading clusters...</p>
        </div>
      ) : filteredClusters.length === 0 ? (
        <div className="empty-state">
          <p>No clusters found. Process images first!</p>
        </div>
      ) : (
        <div className="clusters-grid">
          {filteredClusters.map(([id, cluster]) => (
            <div
              key={id}
              className={`cluster-card ${selected.includes(id) ? "selected" : ""}`}
              onClick={() => toggleSelect(id)}
            >
              <div className="cluster-image-wrapper">
                <img
                  src={`http://127.0.0.1:8000/thumbnail/${cluster.representative}`}
                  alt={id}
                  className="cluster-image"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedCluster({ id, faces: cluster.faces });
                  }}
                />
                <div className="cluster-overlay">
                  <span className="cluster-size">{cluster.size}</span>
                </div>
              </div>
              <div className="cluster-info">
                <h3>{id}</h3>
                <p>{cluster.size} face{cluster.size !== 1 ? "s" : ""}</p>
                <button
                  className={`select-btn ${selected.includes(id) ? "selected" : ""}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleSelect(id);
                  }}
                >
                  {selected.includes(id) ? "✓ Selected" : "Select"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
