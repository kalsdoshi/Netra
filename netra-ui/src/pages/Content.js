import React, { useEffect, useState } from "react";
import { getContentGroups } from "../api/api";
import "../styles/Content.css";

// Emoji map for common object categories
const CATEGORY_ICONS = {
  "People": "\u{1F465}",
  "Car": "\u{1F697}",
  "Dog": "\u{1F415}",
  "Cat": "\u{1F431}",
  "Chair": "\u{1FA91}",
  "Couch": "\u{1F6CB}",
  "Dining Table": "\u{1F37D}",
  "Bottle": "\u{1F37E}",
  "Cup": "\u{2615}",
  "Cell Phone": "\u{1F4F1}",
  "Laptop": "\u{1F4BB}",
  "Tv": "\u{1F4FA}",
  "Book": "\u{1F4DA}",
  "Potted Plant": "\u{1F33F}",
  "Bed": "\u{1F6CF}",
  "Bicycle": "\u{1F6B2}",
  "Motorcycle": "\u{1F3CD}",
  "Bus": "\u{1F68C}",
  "Truck": "\u{1F69B}",
  "Boat": "\u{26F5}",
  "Backpack": "\u{1F392}",
  "Umbrella": "\u{2602}",
  "Handbag": "\u{1F45C}",
  "Tie": "\u{1F454}",
  "Bench": "\u{1FA91}",
  "Bird": "\u{1F426}",
  "Horse": "\u{1F434}",
  "Cow": "\u{1F404}",
  "Sports Ball": "\u{26BD}",
  "Tennis Racket": "\u{1F3BE}",
  "Skateboard": "\u{1F6F9}",
  "Surfboard": "\u{1F3C4}",
  "Clock": "\u{1F550}",
  "Vase": "\u{1F3FA}",
  "Scissors": "\u{2702}",
  "Uncategorized": "\u{1F4C1}",
};

export default function Content() {
  const [groups, setGroups] = useState([]);
  const [totalImages, setTotalImages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [lightbox, setLightbox] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await getContentGroups();
        if (res.data && !res.data.error) {
          setGroups(res.data.groups || []);
          setTotalImages(res.data.total_images || 0);
        }
      } catch (e) {
        console.error("Content groups fetch failed", e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="content-page">
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing image content...</p>
        </div>
      </div>
    );
  }

  // Detail view — show all images in a selected group
  if (selectedGroup) {
    const group = groups.find((g) => g.label === selectedGroup);
    if (!group) return null;
    const icon = CATEGORY_ICONS[group.label] || "📦";

    return (
      <div className="content-page">
        <button className="back-btn" onClick={() => setSelectedGroup(null)}>
          ← Back to Categories
        </button>

        <header className="content-detail-header">
          <span className="content-detail-icon">{icon}</span>
          <div>
            <h2 className="main-title">{group.label}</h2>
            <p className="subtitle">{group.count} photos in this category</p>
          </div>
        </header>

        <div className="content-detail-grid">
          {group.images.map((img, i) => (
            <div
              key={i}
              className="content-detail-item"
              onClick={() => setLightbox(img)}
            >
              <img
                src={`http://127.0.0.1:8000/image-thumb/${img}`}
                alt={img}
                loading="lazy"
              />
              <div className="content-detail-name">{img}</div>
            </div>
          ))}
        </div>

        {/* Lightbox */}
        {lightbox && (
          <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
            <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
              <button className="lightbox-close" onClick={() => setLightbox(null)}>✕</button>
              <img
                src={`http://127.0.0.1:8000/images/${lightbox}`}
                alt={lightbox}
              />
              <p className="lightbox-caption">{lightbox}</p>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Category grid view
  return (
    <div className="content-page">
      <header className="header-section">
        <div className="title-area">
          <h2 className="main-title">🏷️ Content Classification</h2>
          <p className="subtitle">
            {totalImages} photos classified into {groups.length} categories by AI-detected content
          </p>
        </div>
      </header>

      {/* Stats row */}
      <div className="content-stats">
        <div className="stat-card">
          <div className="stat-number">{totalImages}</div>
          <div className="stat-label">Total Photos</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{groups.length}</div>
          <div className="stat-label">Categories</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {groups.find((g) => g.label === "People")?.count || 0}
          </div>
          <div className="stat-label">With People</div>
        </div>
      </div>

      {/* Category cards */}
      <div className="content-grid">
        {groups.map((group) => {
          const icon = CATEGORY_ICONS[group.label] || "📦";
          // Use first image as preview
          const previewImg = group.images?.[0];

          return (
            <div
              key={group.label}
              className="content-card"
              onClick={() => setSelectedGroup(group.label)}
            >
              <div className="content-card-preview">
                {previewImg ? (
                  <img
                    src={`http://127.0.0.1:8000/image-thumb/${previewImg}`}
                    alt={group.label}
                    loading="lazy"
                  />
                ) : (
                  <div className="content-card-empty" />
                )}
                <div className="content-card-overlay">
                  <span className="content-card-icon">{icon}</span>
                </div>
              </div>
              <div className="content-card-info">
                <h3>{group.label}</h3>
                <p>{group.count} photo{group.count !== 1 ? "s" : ""}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
