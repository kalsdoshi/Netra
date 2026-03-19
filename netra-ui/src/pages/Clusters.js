import React, { useEffect, useState } from "react";
import { getClusters, mergeClusters } from "../api/api";

export default function Clusters() {
  const [clusters, setClusters] = useState({});
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selected, setSelected] = useState([]);
  /*const [suggestions, setSuggestions] = useState([]);*/

  useEffect(() => {
    fetchClusters();
  }, []);

  const fetchClusters = async () => {
    try {
      const res = await getClusters();
      if (res.data.error) {
        alert("Error: " + res.data.error);
      } else {
        setClusters(res.data.clusters || {});
      }
    } catch (err) {
      console.error("Failed to fetch clusters:", err);
      alert("Failed to fetch clusters. Make sure the API is running and images are processed.");
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

    await mergeClusters(selected[0], selected[1]);

    alert("Merged!");

    setSelected([]);
    fetchClusters();
  };

  // 🔥 DETAIL VIEW
  if (selectedCluster) {
    return (
      <div>
        <button onClick={() => setSelectedCluster(null)}>⬅ Back</button>

        <h2>{selectedCluster.id}</h2>

        <div style={{ display: "flex", flexWrap: "wrap" }}>
          {selectedCluster.faces.map((face, i) => (
            <img
              key={i}
              src={`http://127.0.0.1:8000/images/${face.image}`}
              alt=""
              width={150}
              style={{ margin: 10 }}
            />
          ))}
        </div>
      </div>
    );
  }
  

  // 🔥 GRID VIEW
  return (
    <div>
      <h2>Clusters</h2>

      {/* 🔥 Merge Button */}
      {selected.length === 2 && (
        <button onClick={handleMerge}>
          🔥 Merge {selected[0]} + {selected[1]}
        </button>
      )}

      <div style={{ display: "flex", flexWrap: "wrap" }}>
        {Object.entries(clusters).map(([id, cluster]) => (
          console.log(cluster.faces)) || (
          <div
            key={id}
            style={{
              margin: 10,
              border: selected.includes(id)
                ? "3px solid green"
                : "1px solid #ccc",
              padding: 10,
            }}
          >
            {/* 🔥 CLICK IMAGE → OPEN CLUSTER */}
            <img
              src={`http://127.0.0.1:8000/thumbnail/${cluster.representative}`}
              alt=""
              width={120}
              style={{ cursor: "pointer" }}
              onClick={() =>
                setSelectedCluster({
                  id,
                  faces: cluster.faces,
                })
              }
            />

            <p>{id}</p>
            <p>{cluster.size} faces</p>

            {/* 🔥 SELECT BUTTON */}
            <button onClick={() => toggleSelect(id)}>
              {selected.includes(id) ? "Unselect" : "Select"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
