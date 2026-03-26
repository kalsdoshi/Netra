import React, { useEffect, useState, useCallback } from "react";
import { LuGitMerge } from "react-icons/lu";
import { getClustersPaginated, mergeClusters } from "../api/api";
import "../styles/Clusters.css";

const PAGE_SIZE = 24;

export default function Clusters() {
  const [clusters, setClusters] = useState({});
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [stats, setStats] = useState({ total: 0, faces: 0 });
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const fetchClusters = useCallback(async (p = 1) => {
    setLoading(true);
    try {
      const res = await getClustersPaginated(p, PAGE_SIZE);
      if (res.data.error) {
        console.error("Error:", res.data.error);
      } else {
        setClusters(res.data.clusters || {});
        setTotalPages(res.data.pages || 1);
        const faces = Object.values(res.data.clusters || {}).reduce((sum, c) => sum + c.size, 0);
        setStats({ total: res.data.total_clusters || 0, faces });
        setPage(res.data.page || 1);
      }
    } catch (err) {
      console.error("Failed to fetch clusters:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchClusters(1);
  }, [fetchClusters]);

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      fetchClusters(newPage);
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
      await fetchClusters(page);
    } catch (err) {
      console.error("Merge failed:", err);
    }
  };

  const filteredClusters = Object.entries(clusters).filter(([id]) =>
    id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Detail View
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
                loading="lazy"
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Grid View
  return (
    <div className="clusters-page">
      <header className="header-section">
        <div className="header-content">
          <div className="title-area">
            <h2 className="main-title">Identity Nexus</h2>
            <p className="subtitle">
              {stats.total} identities · {stats.faces} total faces · Page {page} of {totalPages}
            </p>
          </div>
        </div>
      </header>

      <div className="stats-section">
        <div className="stat-card">
          <div className="stat-number">{stats.total}</div>
          <div className="stat-label">People</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.faces}</div>
          <div className="stat-label">Faces on Page</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{selected.length}/2</div>
          <div className="stat-label">Selected</div>
        </div>
      </div>

      <div className="search-section">
        <input
          type="text"
          placeholder="Search clusters..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      {/* Merge Button */}
      {selected.length === 2 && (
        <div className="merge-action">
          <button className="merge-btn" onClick={handleMerge}>
            <LuGitMerge size={18} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} /> MERGE {selected[0]} + {selected[1]}
          </button>
        </div>
      )}

      {/* Grid */}
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
        <>
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
                    loading="lazy"
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

          {/* Pagination Controls */}
          <div className="pagination-controls">
            <button 
              className="pagination-btn" 
              disabled={page <= 1} 
              onClick={() => handlePageChange(page - 1)}
            >
              ← Previous
            </button>
            <span className="pagination-info">
              Page {page} of {totalPages}
            </span>
            <button 
              className="pagination-btn" 
              disabled={page >= totalPages} 
              onClick={() => handlePageChange(page + 1)}
            >
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
