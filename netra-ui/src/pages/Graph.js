import React, { useEffect, useMemo, useState } from "react";
import { getClusters, getSuggestions } from "../api/api";
import "../styles/Graph.css";

function Bars({ data, maxValue, formatter }) {
	if (data.length === 0) {
		return <p className="page-subtitle">No chart data available.</p>;
	}

	return (
		<div>
			{data.map((entry) => (
				<div className="bar-row" key={entry.label}>
					<span className="bar-label">{entry.label}</span>
					<div className="bar-track">
						<div
							className="bar-fill"
							style={{ width: `${Math.max((entry.value / (maxValue || 1)) * 100, 4)}%` }}
						/>
					</div>
					<span className="bar-value">{formatter(entry.value)}</span>
				</div>
			))}
		</div>
	);
}

export default function Graph() {
	const [clusters, setClusters] = useState({});
	const [suggestions, setSuggestions] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState("");

	useEffect(() => {
		const fetchAll = async () => {
			setLoading(true);
			setError("");

			try {
				const [clusterRes, suggestionRes] = await Promise.all([getClusters(), getSuggestions()]);
				setClusters(clusterRes?.data?.clusters || {});
				setSuggestions(suggestionRes?.data?.suggestions || []);
			} catch (err) {
				console.error("Graph load failed:", err);
				setError("Could not load analytics data.");
			} finally {
				setLoading(false);
			}
		};

		fetchAll();
	}, []);

	const metrics = useMemo(() => {
		const list = Object.entries(clusters);
		const totalClusters = list.length;
		const totalFaces = list.reduce((sum, [, cluster]) => sum + (cluster.size || 0), 0);
		const avgFaces = totalClusters > 0 ? totalFaces / totalClusters : 0;

		return {
			totalClusters,
			totalFaces,
			avgFaces,
			totalSuggestions: suggestions.length,
		};
	}, [clusters, suggestions]);

	const sizeBars = useMemo(() => {
		return Object.entries(clusters)
			.map(([id, cluster]) => ({ label: id, value: cluster.size || 0 }))
			.sort((a, b) => b.value - a.value)
			.slice(0, 8);
	}, [clusters]);

	const suggestionBars = useMemo(() => {
		return suggestions
			.map((s) => ({
				label: `${s.cluster1} <> ${s.cluster2}`,
				value: Number(s.score || 0),
			}))
			.sort((a, b) => b.value - a.value)
			.slice(0, 8);
	}, [suggestions]);

	const maxSize = Math.max(...sizeBars.map((s) => s.value), 1);
	const maxSuggestion = Math.max(...suggestionBars.map((s) => s.value), 1);

	return (
		<section className="graph-page">
			<div>
				<h2 className="page-title">Cluster Analytics</h2>
				<p className="page-subtitle">Live view of cluster distribution and merge confidence trends.</p>
			</div>

			{loading && <p className="status-text">Loading analytics...</p>}
			{error && <p className="status-text">{error}</p>}

			{!loading && !error && (
				<>
					<div className="metric-grid">
						<article className="metric-card">
							<h3>Total Clusters</h3>
							<strong>{metrics.totalClusters}</strong>
						</article>
						<article className="metric-card">
							<h3>Total Faces</h3>
							<strong>{metrics.totalFaces}</strong>
						</article>
						<article className="metric-card">
							<h3>Avg Faces/Cluster</h3>
							<strong>{metrics.avgFaces.toFixed(1)}</strong>
						</article>
						<article className="metric-card">
							<h3>Merge Suggestions</h3>
							<strong>{metrics.totalSuggestions}</strong>
						</article>
					</div>

					<div className="grid-two">
						<div className="chart-panel">
							<h3>Largest Clusters</h3>
							<Bars data={sizeBars} maxValue={maxSize} formatter={(value) => `${value}`} />
						</div>

						<div className="chart-panel">
							<h3>Top Suggestion Similarities</h3>
							<Bars data={suggestionBars} maxValue={maxSuggestion} formatter={(value) => value.toFixed(2)} />
						</div>
					</div>
				</>
			)}
		</section>
	);
}
