import React, { useEffect, useState } from 'react';
import { getGraph, getClusters } from '../api/api';
import '../styles/Clusters.css'; // Inherits base polished CSS 

export default function Home({ navigateTo }) {
  const [stats, setStats] = useState({ nodes: 0, edges: 0, communities: 0, identities: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStats() {
      try {
        const [graphRes, clusterRes] = await Promise.all([getGraph(), getClusters()]);
        
        let nodes = 0, edges = 0, identities = 0;
        let communities = new Set();

        if (graphRes.data && !graphRes.data.error) {
          nodes = graphRes.data.nodes?.length || 0;
          edges = graphRes.data.edges?.length || 0;
          graphRes.data.nodes?.forEach(n => {
            if (n.community) communities.add(n.community);
          });
        }
        
        if (clusterRes.data && !clusterRes.data.error) {
          identities = clusterRes.data.total_clusters || 0;
        }

        setStats({ nodes, edges, communities: communities.size, identities });
      } catch (e) {
        console.error("Dashboard fetch failed", e);
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  return (
    <div className="clusters-page" style={{ animation: 'fadeIn 0.5s ease' }}>
      <header className="header-section">
        <div className="title-area">
          <h2 className="main-title" style={{ fontSize: '2.8rem' }}>Netra Intelligence Base</h2>
          <p className="subtitle">Welcome to your local AI multimodel ecosystem.</p>
        </div>
      </header>

      <div className="stats-section" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        <div className="stat-card">
           <div className="stat-number">{stats.identities}</div>
           <div className="stat-label">Unique Identities</div>
        </div>
        <div className="stat-card">
           <div className="stat-number">{stats.nodes}</div>
           <div className="stat-label">Analyzed Photos</div>
        </div>
        <div className="stat-card">
           <div className="stat-number">{stats.edges}</div>
           <div className="stat-label">Semantic Links</div>
        </div>
        <div className="stat-card">
           <div className="stat-number">{stats.communities}</div>
           <div className="stat-label">Detected Events/Albums</div>
        </div>
      </div>

      <div className="dashboard-actions" style={{ display: 'flex', gap: '1.5rem', marginTop: '2rem' }}>
        <button className="cta-btn" onClick={() => navigateTo('graph')} style={{ padding: '1.5rem 2rem', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '1.8rem' }}>🌌</span>
          <div>
             <div style={{ fontWeight: 800 }}>Enter 3D Matrix</div>
             <div style={{ fontSize: '0.85rem', opacity: 0.8, fontWeight: 500 }}>Visualize entire relational index</div>
          </div>
        </button>

        <button className="cta-btn" onClick={() => navigateTo('clusters')} style={{ padding: '1.5rem 2rem', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.05)', boxShadow: 'none', border: '1px solid rgba(255,255,255,0.1)' }}>
          <span style={{ fontSize: '1.8rem' }}>👥</span>
          <div>
             <div style={{ fontWeight: 800 }}>Manage Identities</div>
             <div style={{ fontSize: '0.85rem', opacity: 0.8, fontWeight: 500 }}>Review extracted faces</div>
          </div>
        </button>
      </div>

    </div>
  );
}
