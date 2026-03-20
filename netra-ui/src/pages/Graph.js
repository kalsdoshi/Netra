import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { getGraph, rebuildGraph } from '../api/api';
import GraphControls from '../components/GraphControls';
import '../styles/Graph.css';

export default function Graph() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverEdge, setHoverEdge] = useState(null);
  const [activeEdgeMeta, setActiveEdgeMeta] = useState(null);
  
  const [filters, setFilters] = useState({ minWeight: 0.15, search: '', modality: 'all' });
  const graphRef = useRef();

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

  // Handle Edge Hover Logic
  const handleEdgeHover = useCallback((edge) => {
    setHoverEdge(edge);
    if (!edge) {
      setActiveEdgeMeta(null);
      return;
    }

    // Edge comes with `source` and `target` which might be the node objects if already parsed by ForceGraph3D
    const sourceNode = typeof edge.source === 'object' ? edge.source : processedData.nodes.find(n => n.id === edge.source);
    const targetNode = typeof edge.target === 'object' ? edge.target : processedData.nodes.find(n => n.id === edge.target);

    if (sourceNode && targetNode) {
      // Compute overlapping identities
      const sharedClusters = sourceNode.clusters.filter(c => targetNode.clusters.includes(c));
      
      // Compute overlapping objects
      const sourceObjLabels = sourceNode.objects.map(o => o.label);
      const targetObjLabels = targetNode.objects.map(o => o.label);
      const sharedObjects = sourceObjLabels.filter(label => targetObjLabels.includes(label));
      
      // Deduplicate object list
      const uniqueSharedObjects = [...new Set(sharedObjects)];

      setActiveEdgeMeta({
        source: sourceNode,
        target: targetNode,
        edge: edge,
        sharedClusters,
        sharedObjects: uniqueSharedObjects
      });
    }
  }, [processedData.nodes]);

  const textureCache = useRef(new Map());

  // Compute node Three.js 3D representations
  const renderNode = useCallback((node) => {
    // Determine Highlight
    const isSearched = filters.search && (
      node.id.includes(filters.search) || 
      node.clusters.some(c => c.includes(filters.search))
    );

    let imgTexture = textureCache.current.get(node.id);
    if (!imgTexture) {
      // Use the newly optimized thumbnail endpoint instead of the multi-megabyte raw picture bounds
      imgTexture = new THREE.TextureLoader().load(`http://127.0.0.1:8000/image-thumb/${node.id}`);
      imgTexture.colorSpace = THREE.SRGBColorSpace;
      textureCache.current.set(node.id, imgTexture);
    }

    const material = new THREE.SpriteMaterial({ 
      map: imgTexture,
      color: isSearched ? 0xff00ff : 0xffffff,
      transparent: true,
      opacity: 0.95
    });
    
    const sprite = new THREE.Sprite(material);
    
    // Scale up slightly for better visibility
    sprite.scale.set(16, 16, 1);
    
    // Add glowing border effect if searched
    if (isSearched) {
      sprite.scale.set(22, 22, 1);
    }

    return sprite;
  }, [filters.search]);

  return (
    <section className="graph-page-3d">
      <GraphControls 
        filters={filters} 
        setFilters={setFilters} 
        onRebuild={handleRebuild} 
        loading={loading}
      />

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
            return link.modality === 'face' ? 'rgba(0, 255, 255, 0.4)' : 'rgba(255, 165, 0, 0.4)';
          }}
          linkWidth={link => {
            const isHovered = hoverEdge && (link.source === hoverEdge.source && link.target === hoverEdge.target);
            return isHovered ? link.weight * 6 : link.weight * 3;
          }}
          linkDirectionalParticles={link => (hoverEdge && (link.source === hoverEdge.source && link.target === hoverEdge.target)) ? 2 : 0}
          linkDirectionalParticleWidth={1.5}
          onNodeHover={node => setHoverNode(node)}
          onLinkHover={handleEdgeHover}
          onNodeClick={node => {
            // Smooth Camera tracking to node
            const distance = 80;
            const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
            graphRef.current.cameraPosition(
              { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
              node,
              2000
            );
          }}
          backgroundColor="#050811"
        />

        {/* Floating Node Inspector */}
        {hoverNode && !activeEdgeMeta && (
          <div className="glass-panel node-inspector">
            <img src={`http://127.0.0.1:8000/images/${hoverNode.id}`} alt={hoverNode.id} />
            <div className="inspector-info">
              <h4>{hoverNode.id}</h4>
              <div className="tag-list">
                {hoverNode.clusters.map(c => <span key={c} className="tag cluster-tag">{c}</span>)}
                {hoverNode.objects.slice(0, 4).map(o => <span key={o.label} className="tag object-tag">{o.label}</span>)}
              </div>
            </div>
          </div>
        )}

        {/* Floating Semantic Edge Inspector */}
        {activeEdgeMeta && (
          <div className="glass-panel edge-inspector">
            <div className="edge-header">
              <h4>Semantic Linkage</h4>
              <span className="weight-badge">Link Weight: {activeEdgeMeta.edge.weight.toFixed(2)}</span>
            </div>
            
            <div className="edge-nodes-preview">
               <img src={`http://127.0.0.1:8000/images/${activeEdgeMeta.source.id}`} alt="Source" />
               <div className="connection-icon">⟷</div>
               <img src={`http://127.0.0.1:8000/images/${activeEdgeMeta.target.id}`} alt="Target" />
            </div>

            <div className="edge-metadata">
              {activeEdgeMeta.sharedClusters.length > 0 && (
                <div className="meta-section">
                  <h5>👥 Shared Identities (Sim: {activeEdgeMeta.edge.face_sim.toFixed(2)})</h5>
                  <div className="tag-list">
                    {activeEdgeMeta.sharedClusters.map(c => <span key={c} className="tag cluster-tag">{c}</span>)}
                  </div>
                </div>
              )}

              {activeEdgeMeta.sharedObjects.length > 0 && (
                <div className="meta-section">
                  <h5>📦 Shared Objects (Sim: {activeEdgeMeta.edge.object_sim.toFixed(2)})</h5>
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

      </div>
    </section>
  );
}
