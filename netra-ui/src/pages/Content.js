import React, { useEffect, useState } from "react";
import {
  LuUsers, LuCar, LuDog, LuCat, LuArmchair, LuSofa, LuUtensils,
  LuWine, LuCoffee, LuSmartphone, LuLaptop, LuTv, LuBookOpen,
  LuSprout, LuBedDouble, LuBike, LuShip, LuBackpack, LuUmbrella,
  LuShoppingBag, LuBird, LuDumbbell, LuClock, LuScissors, LuBox,
  LuTags, LuFolder, LuTruck, LuGlobe
} from "react-icons/lu";
import { getContentGroups } from "../api/api";
import "../styles/Content.css";

// SVG icon map for common object categories
const CATEGORY_ICONS = {
  "People": LuUsers,
  "Car": LuCar,
  "Dog": LuDog,
  "Cat": LuCat,
  "Chair": LuArmchair,
  "Couch": LuSofa,
  "Dining Table": LuUtensils,
  "Bottle": LuWine,
  "Cup": LuCoffee,
  "Cell Phone": LuSmartphone,
  "Laptop": LuLaptop,
  "Tv": LuTv,
  "Book": LuBookOpen,
  "Potted Plant": LuSprout,
  "Bed": LuBedDouble,
  "Bicycle": LuBike,
  "Motorcycle": LuBike,
  "Bus": LuTruck,
  "Truck": LuTruck,
  "Boat": LuShip,
  "Backpack": LuBackpack,
  "Umbrella": LuUmbrella,
  "Handbag": LuShoppingBag,
  "Tie": LuTags,
  "Bench": LuArmchair,
  "Bird": LuBird,
  "Horse": LuGlobe,
  "Cow": LuGlobe,
  "Sports Ball": LuDumbbell,
  "Tennis Racket": LuDumbbell,
  "Skateboard": LuBike,
  "Surfboard": LuShip,
  "Clock": LuClock,
  "Vase": LuGlobe,
  "Scissors": LuScissors,
  "Uncategorized": LuFolder,
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
    const IconComp = CATEGORY_ICONS[group.label] || LuBox;

    return (
      <div className="content-page">
        <button className="back-btn" onClick={() => setSelectedGroup(null)}>
          ← Back to Categories
        </button>

        <header className="content-detail-header">
          <span className="content-detail-icon"><IconComp size={32} /></span>
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
          <h2 className="main-title"><LuTags style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Content Classification</h2>
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
          const IconComp = CATEGORY_ICONS[group.label] || LuBox;
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
                  <span className="content-card-icon"><IconComp size={24} /></span>
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
