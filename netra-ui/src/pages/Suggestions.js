import React, { useEffect, useState } from "react";
import { getSuggestions, getClusters, mergeClusters } from "../api/api";
import "../styles/Suggestions.css";

export default function Suggestions() {
  const [suggestions, setSuggestions] = useState([]);
  const [clusters, setClusters] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError("");

    try {
      const [s, c] = await Promise.all([getSuggestions(), getClusters()]);

      setSuggestions(s?.data?.suggestions || []);
      setClusters(c?.data?.clusters || {});
    } catch (err) {
      console.error("Suggestions load failed:", err);
      setError("Could not load suggestions.");
    } finally {
      setLoading(false);
    }
  };

  const handleMerge = async (cluster1, cluster2) => {
    try {
      await mergeClusters(cluster1, cluster2);
      await fetchData();
    } catch (err) {
      console.error("Merge failed:", err);
      setError("Merge failed. Please retry.");
    }
  };

  return (
    <section className="suggestions-page">
      <div className="suggestions-header">
        <div>
          <h2 className="page-title">Merge Suggestions</h2>
          <p className="page-subtitle">Review nearest clusters and merge with one click.</p>
        </div>
        <button type="button" className="suggestion-btn" onClick={fetchData}>
          Refresh
        </button>
      </div>

      {loading && <p className="status-text">Loading suggestions...</p>}
      {error && <p className="status-text">{error}</p>}

      {!loading && suggestions.length === 0 && (
        <div className="empty-note">No merge suggestions are available right now.</div>
      )}

      <div className="suggestions-grid">
        {suggestions.map((s, i) => {
          const c1 = clusters[s.cluster1];
          const c2 = clusters[s.cluster2];

          if (!c1 || !c2) return null;

          return (
            <article key={`${s.cluster1}-${s.cluster2}-${i}`} className="suggestion-card">
              <div className="suggestion-pair">
                <div className="face-chip">
                  <img
                    src={`http://127.0.0.1:8000/thumbnail/${c1.representative}`}
                    alt={s.cluster1}
                  />
                  <div>
                    <strong>{s.cluster1}</strong>
                    <p className="suggestion-score">{c1.size} faces</p>
                  </div>
                </div>

                <span className="suggestion-score">Close Match</span>

                <div className="face-chip">
                  <img
                    src={`http://127.0.0.1:8000/thumbnail/${c2.representative}`}
                    alt={s.cluster2}
                  />
                  <div>
                    <strong>{s.cluster2}</strong>
                    <p className="suggestion-score">{c2.size} faces</p>
                  </div>
                </div>
              </div>

              <div className="suggestion-actions">
                <span className="metric-pill">similarity: {Number(s.score || 0).toFixed(2)}</span>
                <button
                  type="button"
                  className="suggestion-btn"
                  onClick={() => handleMerge(s.cluster1, s.cluster2)}
                >
                  Merge Clusters
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}