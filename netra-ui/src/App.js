import React, { useState } from "react";
import {
  LuScanEye,
  LuLayoutDashboard,
  LuUsers,
  LuSparkles,
  LuBox,
  LuCalendarDays,
  LuTags,
  LuSearch,
  LuOrbit,
} from "react-icons/lu";

import Home from "./pages/Home";
import Clusters from "./pages/Clusters";
import Search from "./pages/Search";
import Suggestions from "./pages/Suggestions";
import Objects from "./pages/Objects";
import Graph from "./pages/Graph";
import Timeline from "./pages/Timeline";
import Content from "./pages/Content";
import "./App.css";

function App() {
  const [tab, setTab] = useState("home");

  const tabs = [
    { id: "home", label: "Dashboard", icon: <LuLayoutDashboard /> },
    { id: "clusters", label: "Identity Nexus", icon: <LuUsers /> },
    { id: "suggestions", label: "Merge AI", icon: <LuSparkles /> },
    { id: "objects", label: "Object Index", icon: <LuBox /> },
    { id: "timeline", label: "Timeline", icon: <LuCalendarDays /> },
    { id: "content", label: "Content Tags", icon: <LuTags /> },
    { id: "search", label: "Semantic Search", icon: <LuSearch /> },
    { id: "graph", label: "3D Matrix", icon: <LuOrbit /> },
  ];

  const renderPage = () => {
    switch (tab) {
      case "home":
        return <Home navigateTo={setTab} />;
      case "clusters":
        return <Clusters />;
      case "search":
        return <Search />;
      case "suggestions":
        return <Suggestions />;
      case "objects":
        return <Objects />;
      case "timeline":
        return <Timeline />;
      case "content":
        return <Content />;
      case "graph":
        return <Graph />;
      default:
        return <Home navigateTo={setTab} />;
    }
  };

  return (
    <div className="app-layout-desktop fade-in">
      <div className="bg-orb bg-orb-a" />
      <div className="bg-orb bg-orb-b" />

      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <div className="logo-box"><LuScanEye /></div>
          <div className="brand-text">
            <h2>Netra</h2>
            <span>Vision Engine</span>
          </div>
        </div>
        
        <nav className="sidebar-nav" aria-label="Main sidebar navigation">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebar-link ${tab === item.id ? "active" : ""}`}
              onClick={() => setTab(item.id)}
            >
              <span className="sidebar-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>
        
        <div className="sidebar-footer">
           <div className="status-dot"></div> System Online
        </div>
      </aside>

      <main className="app-content-desktop">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;