import React, { useEffect, useState } from "react";
import { LuCalendarDays } from "react-icons/lu";
import { getTimeline } from "../api/api";
import "../styles/Timeline.css";

export default function Timeline() {
  const [groups, setGroups] = useState([]);
  const [totalImages, setTotalImages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [expandedDate, setExpandedDate] = useState(null);
  const [lightbox, setLightbox] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await getTimeline();
        if (res.data && !res.data.error) {
          setGroups(res.data.groups || []);
          setTotalImages(res.data.total_images || 0);
          // Auto-expand the first group
          if (res.data.groups?.length > 0) {
            setExpandedDate(res.data.groups[0].date);
          }
        }
      } catch (e) {
        console.error("Timeline fetch failed", e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="timeline-page">
        <div className="loading">
          <div className="spinner"></div>
          <p>Reading EXIF data from all images...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="timeline-page">
      <header className="header-section">
        <div className="title-area">
          <h2 className="main-title"><LuCalendarDays style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Photo Timeline</h2>
          <p className="subtitle">
            {totalImages} photos across {groups.length} dates — organized by when they were taken
          </p>
        </div>
      </header>

      {/* Stats */}
      <div className="timeline-stats">
        <div className="stat-card">
          <div className="stat-number">{totalImages}</div>
          <div className="stat-label">Total Photos</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{groups.length}</div>
          <div className="stat-label">Unique Dates</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {groups.length > 0 ? Math.round(totalImages / groups.length) : 0}
          </div>
          <div className="stat-label">Avg Per Day</div>
        </div>
      </div>

      {/* Timeline */}
      <div className="timeline-container">
        <div className="timeline-line" />

        {groups.map((group) => {
          const isExpanded = expandedDate === group.date;

          return (
            <div key={group.date} className="timeline-group">
              <button
                className={`timeline-header ${isExpanded ? "expanded" : ""}`}
                onClick={() => setExpandedDate(isExpanded ? null : group.date)}
              >
                <div className="timeline-dot" />
                <div className="timeline-date-info">
                  <h3 className="timeline-date">{group.label}</h3>
                  <span className="timeline-count">{group.count} photo{group.count !== 1 ? "s" : ""}</span>
                </div>
                <span className="timeline-chevron">{isExpanded ? "▲" : "▼"}</span>
              </button>

              {isExpanded && (
                <div className="timeline-gallery">
                  {group.images.map((img, i) => (
                    <div
                      key={i}
                      className="timeline-thumb"
                      onClick={() => setLightbox(img.image)}
                    >
                      <img
                        src={`http://127.0.0.1:8000/image-thumb/${img.image}`}
                        alt={img.image}
                        loading="lazy"
                      />
                      <div className="timeline-thumb-name">{img.image}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
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
