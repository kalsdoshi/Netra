import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { LuUsers, LuBox, LuLink, LuGlobe } from 'react-icons/lu';
import { getGraph, rebuildGraph, getObjectsByImage } from '../api/api';
import GraphControls from '../components/GraphControls';
import '../styles/Graph.css';

export default function Graph() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverEdge, setHoverEdge] = useState(null);
  const [activeEdgeMeta, setActiveEdgeMeta] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeObjects, setNodeObjects] = useState([]);
  const [lightbox, setLightbox] = useState(null);
  const [controlsCollapsed, setControlsCollapsed] = useState(false);
  
  const [filters, setFilters] = useState({ minWeight: 0.15, search: '', modality: 'all' });
  const graphRef = useRef();
  const textureCache = useRef(new Map());

  useEffect(() => {
    fetchGraph();
  }, []);

  const fetchGraph = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getGraph();
      if (res.data?.error) {
        setError(res.data.error);
        setGraphData({ nodes: [], edges: [] });
      } else {
        setGraphData(res.data);
      }
    } catch (err) {
      console.error('Failed to load graph:', err);
      setError('Could not connect to graph database.');
    } finally {
      setLoading(false);
    }
  };

  const handleRebuild = async () => {
    setLoading(true);
    setError('');
    try {
      await rebuildGraph();
      await fetchGraph();
    } catch (err) {
      console.error('Rebuild failed', err);
      setError('Failed to rebuild graph.');
      setLoading(false);
    }
  };

  // Pre-process and filter data
  const processedData = useMemo(() => {
    if (!graphData.nodes || !graphData.edges) return { nodes: [], links: [] };

    let filteredNodes = graphData.nodes.filter(node => {
      if (!filters.search) return true;
      const term = filters.search.toLowerCase();
      const inId = node.id.toLowerCase().includes(term);
      const inClusters = node.clusters.some(c => c.toLowerCase().includes(term));
      const inObjects = node.objects.some(o => o.label.toLowerCase().includes(term));
      return inId || inClusters || inObjects;
    });

    const activeNodeIds = new Set(filteredNodes.map(n => n.id));

    let filteredLinks = graphData.edges.filter(edge => {
      if (edge.weight < filters.minWeight) return false;
      if (filters.modality !== 'all' && edge.modality !== filters.modality) return false;
      return activeNodeIds.has(edge.source) && activeNodeIds.has(edge.target);
    });

    return {
      nodes: filteredNodes.map(n => ({...n})),
      links: filteredLinks.map(e => ({...e}))
    };
  }, [graphData, filters]);

  // Load objects for selected node
  useEffect(() => {
    if (!selectedNode) {
      setNodeObjects([]);
      return;
    }
    const loadObjects = async () => {
      try {
        const res = await getObjectsByImage(selectedNode.id);
        setNodeObjects(res?.data?.objects || []);
      } catch { setNodeObjects([]); }
    };
    loadObjects();
  }, [selectedNode]);

  // Get connected edges for a node
  const getNodeConnections = useCallback((nodeId) => {
    if (!graphData.edges) return [];
    return graphData.edges.filter(
      e => (typeof e.source === 'object' ? e.source.id : e.source) === nodeId ||
           (typeof e.target === 'object' ? e.target.id : e.target) === nodeId
    ).sort((a, b) => b.weight - a.weight).slice(0, 8);
  }, [graphData.edges]);

  // Handle Edge Hover Logic
  const handleEdgeHover = useCallback((edge) => {
    setHoverEdge(edge);
    if (!edge) { setActiveEdgeMeta(null); return; }

    const sourceNode = typeof edge.source === 'object' ? edge.source : processedData.nodes.find(n => n.id === edge.source);
    const targetNode = typeof edge.target === 'object' ? edge.target : processedData.nodes.find(n => n.id === edge.target);

    if (sourceNode && targetNode) {
      const sharedClusters = sourceNode.clusters.filter(c => targetNode.clusters.includes(c));
      const sourceObjLabels = sourceNode.objects.map(o => o.label);
      const targetObjLabels = targetNode.objects.map(o => o.label);
      const sharedObjects = [...new Set(sourceObjLabels.filter(label => targetObjLabels.includes(label)))];

      setActiveEdgeMeta({ source: sourceNode, target: targetNode, edge, sharedClusters, sharedObjects });
    }
  }, [processedData.nodes]);

  // Compute node Three.js 3D representations
  const renderNode = useCallback((node) => {
    const isSearched = filters.search && (
      node.id.includes(filters.search) || 
      node.clusters.some(c => c.includes(filters.search))
    );
    const isSelected = selectedNode && selectedNode.id === node.id;

    let imgTexture = textureCache.current.get(node.id);
    if (!imgTexture) {
      imgTexture = new THREE.TextureLoader().load(`http://127.0.0.1:8000/image-thumb/${node.id}`);
      imgTexture.colorSpace = THREE.SRGBColorSpace;
      textureCache.current.set(node.id, imgTexture);
    }

    const material = new THREE.SpriteMaterial({ 
      map: imgTexture,
      color: isSelected ? 0x00ffff : isSearched ? 0xff00ff : 0xffffff,
      transparent: true,
      opacity: 0.95
    });
    
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(isSelected ? 24 : isSearched ? 20 : 14, isSelected ? 24 : isSearched ? 20 : 14, 1);

    return sprite;
  }, [filters.search, selectedNode]);

  // Handle node click — select and fly to
  const handleNodeClick = useCallback((node) => {
    setSelectedNode(prev => prev?.id === node.id ? null : node);
    setActiveEdgeMeta(null);

    const distance = 80;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
    graphRef.current?.cameraPosition(
      { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
      node,
      1500
    );
  }, []);

  // Close detail panel
  const closeDetail = () => {
    setSelectedNode(null);
    setNodeObjects([]);
  };

  const connections = selectedNode ? getNodeConnections(selectedNode.id) : [];

  return (
    <section className={`graph-page-3d ${selectedNode ? 'detail-open' : ''}`}>
      {/* Controls Bar — collapsible */}
      <div className={`graph-controls-wrapper ${controlsCollapsed ? 'collapsed' : ''}`}>
        <button
          className="controls-toggle"
          onClick={() => setControlsCollapsed(!controlsCollapsed)}
          title={controlsCollapsed ? 'Show Controls' : 'Hide Controls'}
        >
          {controlsCollapsed ? '▼ Controls' : '▲ Hide'}
        </button>
        {!controlsCollapsed && (
          <GraphControls 
            filters={filters} 
            setFilters={setFilters} 
            onRebuild={handleRebuild} 
            loading={loading}
          />
        )}
      </div>

      <div className="graph-main-area">
        {/* 3D Graph Canvas */}
        <div className="graph-view-container">
          {loading && <div className="graph-overlay">Initializing Relational Matrix...</div>}
          {error && <div className="graph-overlay error">{error}</div>}
          
          {!loading && !error && processedData.nodes.length === 0 && (
            <div className="graph-overlay">No nexus points match parameters.</div>
          )}

          <ForceGraph3D
            ref={graphRef}
            graphData={processedData}
            nodeLabel="id"
            nodeThreeObject={renderNode}
            linkColor={link => {
              const isHovered = hoverEdge && (link.source === hoverEdge.source && link.target === hoverEdge.target);
              if (isHovered) return 'rgba(255, 255, 255, 1)';
              return link.modality === 'face' ? 'rgba(0, 255, 255, 0.35)' : 'rgba(255, 165, 0, 0.35)';
            }}
            linkWidth={link => {
              const isHovered = hoverEdge && (link.source === hoverEdge.source && link.target === hoverEdge.target);
              return isHovered ? link.weight * 5 : link.weight * 2;
            }}
            linkDirectionalParticles={link => (hoverEdge && (link.source === hoverEdge.source && link.target === hoverEdge.target)) ? 2 : 0}
            linkDirectionalParticleWidth={1.5}
            onNodeHover={node => setHoverNode(node)}
            onLinkHover={handleEdgeHover}
            onNodeClick={handleNodeClick}
            onBackgroundClick={() => { setSelectedNode(null); setActiveEdgeMeta(null); }}
            backgroundColor="#050811"
            warmupTicks={60}
            cooldownTicks={120}
          />

          {/* Floating Node Hover Tooltip (bottom-left) */}
          {hoverNode && !selectedNode && !activeEdgeMeta && (
            <div className="glass-panel node-inspector">
              <img src={`http://127.0.0.1:8000/image-thumb/${hoverNode.id}`} alt={hoverNode.id} />
              <div className="inspector-info">
                <h4>{hoverNode.id}</h4>
                <div className="tag-list">
                  {hoverNode.clusters.map(c => <span key={c} className="tag cluster-tag">{c}</span>)}
                  {hoverNode.objects.slice(0, 3).map(o => <span key={o.label} className="tag object-tag">{o.label}</span>)}
                </div>
              </div>
            </div>
          )}

          {/* Floating Edge Tooltip */}
          {activeEdgeMeta && !selectedNode && (
            <div className="glass-panel edge-inspector">
              <div className="edge-header">
                <h4>Semantic Link</h4>
                <span className="weight-badge">{activeEdgeMeta.edge.weight.toFixed(2)}</span>
              </div>
              <div className="edge-nodes-preview">
                 <img src={`http://127.0.0.1:8000/image-thumb/${activeEdgeMeta.source.id}`} alt="Source" />
                 <div className="connection-icon">{'\u27F7'}</div>
                 <img src={`http://127.0.0.1:8000/image-thumb/${activeEdgeMeta.target.id}`} alt="Target" />
              </div>
              <div className="edge-metadata">
                {activeEdgeMeta.sharedClusters.length > 0 && (
                  <div className="meta-section">
                    <h5><LuUsers size={14} style={{ verticalAlign: 'middle', marginRight: '0.3rem' }} /> Shared Identities ({activeEdgeMeta.edge.face_sim.toFixed(2)})</h5>
                    <div className="tag-list">
                      {activeEdgeMeta.sharedClusters.map(c => <span key={c} className="tag cluster-tag">{c}</span>)}
                    </div>
                  </div>
                )}
                {activeEdgeMeta.sharedObjects.length > 0 && (
                  <div className="meta-section">
                    <h5><LuBox size={14} style={{ verticalAlign: 'middle', marginRight: '0.3rem' }} /> Shared Objects ({activeEdgeMeta.edge.object_sim.toFixed(2)})</h5>
                    <div className="tag-list">
                      {activeEdgeMeta.sharedObjects.map(o => <span key={o} className="tag object-tag">{o}</span>)}
                    </div>
                  </div>
                )}
                {activeEdgeMeta.sharedClusters.length === 0 && activeEdgeMeta.sharedObjects.length === 0 && (
                  <p className="status-text ghost">Weak systemic relation over threshold.</p>
                )}
              </div>
            </div>
          )}

          {/* Stats badge */}
          <div className="graph-stats-badge">
            <span>{processedData.nodes.length} nodes</span>
            <span className="dot">{'\u00B7'}</span>
            <span>{processedData.links.length} links</span>
          </div>
        </div>

        {/* Right-Side Detail Panel */}
        {selectedNode && (
          <aside className="detail-panel">
            <div className="detail-panel-inner">
              {/* Header */}
              <div className="detail-panel-header">
                <h3>Image Details</h3>
                <button className="detail-close-btn" onClick={closeDetail}>{'\u2715'}</button>
              </div>

              {/* Preview with fullscreen button */}
              <div className="detail-preview">
                <img
                  src={`http://127.0.0.1:8000/images/${selectedNode.id}`}
                  alt={selectedNode.id}
                  onClick={() => setLightbox(selectedNode.id)}
                />
                <button
                  className="fullscreen-btn"
                  onClick={() => setLightbox(selectedNode.id)}
                  title="View fullscreen"
                >
                  {'\u26F6'}
                </button>
              </div>

              {/* Filename */}
              <div className="detail-filename">{selectedNode.id}</div>

              {/* People in this image */}
              {selectedNode.clusters.length > 0 && (
                <div className="detail-section">
                  <h4><LuUsers size={16} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} /> People Detected</h4>
                  <div className="tag-list">
                    {selectedNode.clusters.map(c => (
                      <span key={c} className="tag cluster-tag">{c}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Objects in this image */}
              {nodeObjects.length > 0 && (
                <div className="detail-section">
                  <h4><LuBox size={16} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} /> Objects Detected</h4>
                  <div className="detail-objects-list">
                    {nodeObjects.map((obj, i) => (
                      <div key={i} className="detail-object-row">
                        <span className="obj-label">{obj.label}</span>
                        <div className="obj-bar-wrapper">
                          <div
                            className="obj-bar"
                            style={{ width: `${Math.round(obj.confidence * 100)}%` }}
                          />
                        </div>
                        <span className="obj-conf">{(obj.confidence * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Connections */}
              {connections.length > 0 && (
                <div className="detail-section">
                  <h4><LuLink size={16} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} /> Top Connections</h4>
                  <div className="detail-connections">
                    {connections.map((conn, i) => {
                      const otherId = (typeof conn.source === 'object' ? conn.source.id : conn.source) === selectedNode.id
                        ? (typeof conn.target === 'object' ? conn.target.id : conn.target)
                        : (typeof conn.source === 'object' ? conn.source.id : conn.source);
                      return (
                        <div
                          key={i}
                          className="detail-connection-row"
                          onClick={() => {
                            const targetNode = processedData.nodes.find(n => n.id === otherId);
                            if (targetNode) handleNodeClick(targetNode);
                          }}
                        >
                          <img src={`http://127.0.0.1:8000/image-thumb/${otherId}`} alt={otherId} />
                          <div className="conn-info">
                            <span className="conn-name">{otherId}</span>
                            <span className={`conn-type ${conn.modality}`}>{conn.modality}</span>
                          </div>
                          <span className="conn-score">{conn.weight.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Community */}
              {selectedNode.community !== undefined && (
                <div className="detail-section">
                  <h4><LuGlobe size={16} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} /> Community</h4>
                  <span className="community-badge">Group {selectedNode.community}</span>
                </div>
              )}
            </div>
          </aside>
        )}
      </div>

      {/* Fullscreen Lightbox */}
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightbox(null)}>{'\u2715'}</button>
            <img src={`http://127.0.0.1:8000/images/${lightbox}`} alt={lightbox} />
            <p className="lightbox-caption">{lightbox}</p>
          </div>
        </div>
      )}
    </section>
  );
}
