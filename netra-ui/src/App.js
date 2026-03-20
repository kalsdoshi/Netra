import React, { useState } from "react";

import Clusters from "./pages/Clusters";
import Search from "./pages/Search";
import Suggestions from "./pages/Suggestions";
import Objects from "./pages/Objects";
import Graph from "./pages/Graph";
import "./App.css";

function App() {
  const [tab, setTab] = useState("clusters");

  const tabs = [
    { id: "clusters", label: "Clusters" },
    { id: "search", label: "Search" },
    { id: "suggestions", label: "Suggestions" },
    { id: "objects", label: "Objects" },
    { id: "graph", label: "Graph" },
  ];

  const renderPage = () => {
    switch (tab) {
      case "clusters":
        return <Clusters />;
      case "search":
        return <Search />;
      case "suggestions":
        return <Suggestions />;
      case "objects":
        return <Objects />;
      case "graph":
        return <Graph />;
      default:
        return <Clusters />;
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-orb bg-orb-a" />
      <div className="bg-orb bg-orb-b" />
      <div className="bg-grid" />

      <header className="app-topbar">
        <div>
          <p className="app-kicker">Netra Vision System</p>
          <h1>Intelligence Workbench</h1>
        </div>
        <nav className="tab-nav" aria-label="Main navigation tabs">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`tab-btn ${tab === item.id ? "active" : ""}`}
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-content">{renderPage()}</main>
    </div>
  );
}

export default App;