import React, { useEffect, useMemo, useState } from "react";
import { getClusters, getObjectsByImage } from "../api/api";
import "../styles/Objects.css";

export default function Objects() {
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState("");
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dimensions, setDimensions] = useState({ w: 1, h: 1 });

  useEffect(() => {
    const loadImages = async () => {
      try {
        const res = await getClusters();
        const clusterEntries = Object.values(res?.data?.clusters || {});
        const uniqueImages = Array.from(
          new Set(clusterEntries.flatMap((cluster) => (cluster.faces || []).map((face) => face.image)))
        );
        setImages(uniqueImages);
        if (uniqueImages.length > 0) {
          setSelectedImage(uniqueImages[0]);
        }
      } catch (err) {
        console.error("Failed to load image list:", err);
        setError("Could not load images from clusters.");
      }
    };

    loadImages();
  }, []);

  useEffect(() => {
    if (!selectedImage) return;

    const loadObjects = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await getObjectsByImage(selectedImage);
        setObjects(res?.data?.objects || []);
      } catch (err) {
        console.error("Object fetch failed:", err);
        setError("Could not fetch detected objects for this image.");
        setObjects([]);
      } finally {
        setLoading(false);
      }
    };

    loadObjects();
  }, [selectedImage]);

  const labelSummary = useMemo(() => {
    const counts = objects.reduce((acc, obj) => {
      const key = obj.label || "unknown";
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [objects]);

  const toPct = (value, axis) => {
    if (!value || !dimensions[axis]) return 0;
    return (value / dimensions[axis]) * 100;
  };

  return (
    <section className="objects-page">
      <div>
        <h2 className="page-title">Objects by Source Image</h2>
        <p className="page-subtitle">Select any indexed image to inspect object detections.</p>
      </div>

      <div className="objects-toolbar app-panel">
        <select
          className="objects-select"
          value={selectedImage}
          onChange={(e) => setSelectedImage(e.target.value)}
        >
          {images.map((img) => (
            <option key={img} value={img}>
              {img}
            </option>
          ))}
        </select>
        <span className="page-subtitle">{loading ? "Loading detections..." : `${objects.length} objects`}</span>
      </div>

      {error && <p className="status-text">{error}</p>}

      {!!selectedImage && (
        <div className="objects-canvas">
          <img
            src={`http://127.0.0.1:8000/images/${selectedImage}`}
            alt={selectedImage}
            onLoad={(event) => {
              const target = event.currentTarget;
              setDimensions({ w: target.naturalWidth || 1, h: target.naturalHeight || 1 });
            }}
          />

          {objects.map((obj, idx) => {
            const [x1, y1, x2, y2] = obj.bbox || [0, 0, 0, 0];

            return (
              <div
                key={`${obj.label}-${idx}`}
                className="object-box"
                style={{
                  left: `${toPct(x1, "w")}%`,
                  top: `${toPct(y1, "h")}%`,
                  width: `${toPct(x2 - x1, "w")}%`,
                  height: `${toPct(y2 - y1, "h")}%`,
                }}
              >
                <span className="object-box-label">{obj.label}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="object-list">
        {labelSummary.map(([name, count]) => (
          <span className="object-chip" key={name}>
            {name}: {count}
          </span>
        ))}
      </div>
    </section>
  );
}