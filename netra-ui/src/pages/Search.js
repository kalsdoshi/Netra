import React, { useEffect, useMemo, useState } from "react";
import { searchFace } from "../api/api";
import "../styles/Search.css";

export default function Search() {
	const [file, setFile] = useState(null);
	const [results, setResults] = useState([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");

	const previewUrl = useMemo(() => {
		if (!file) return "";
		return URL.createObjectURL(file);
	}, [file]);

	useEffect(() => {
		return () => {
			if (previewUrl) {
				URL.revokeObjectURL(previewUrl);
			}
		};
	}, [previewUrl]);

	const onFileChange = (event) => {
		const selected = event.target.files?.[0] || null;
		setFile(selected);
		setResults([]);
		setError("");
	};

	const runSearch = async () => {
		if (!file) return;

		setLoading(true);
		setError("");

		try {
			const response = await searchFace(file);
			setResults(response?.data?.results || []);
		} catch (err) {
			console.error("Search failed:", err);
			setError("Face search failed. Please retry.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<section className="search-page">
			<div>
				<h2 className="page-title">Face Search</h2>
				<p className="page-subtitle">Upload a query image and inspect nearest matches from your index.</p>
			</div>

			<div className="search-upload app-panel">
				<label className="file-input">
					<input type="file" accept="image/*" onChange={onFileChange} />
					<span>{file ? file.name : "Choose query image"}</span>
				</label>
				<button type="button" className="cta-btn" disabled={!file || loading} onClick={runSearch}>
					{loading ? "Searching..." : "Run Search"}
				</button>
			</div>

			<div className="search-layout">
				<aside className="query-card">
					<h3>Query</h3>
					{previewUrl ? (
						<img src={previewUrl} alt="Query preview" className="query-preview" />
					) : (
						<p className="page-subtitle">No image selected yet.</p>
					)}
				</aside>

				<div className="search-results app-panel">
					<h3>Matches</h3>
					{error && <p className="status-text">{error}</p>}
					{!loading && !error && results.length === 0 && (
						<p className="page-subtitle">No matches to show yet.</p>
					)}
					{results.map((item, index) => (
						<article key={`${item.image}-${index}`} className="result-row">
							<img
								className="result-thumb"
								src={`http://127.0.0.1:8000/images/${item.image}`}
								alt={item.image}
							/>
							<div>
								<strong>{item.image}</strong>
								<p className="page-subtitle">result #{index + 1}</p>
							</div>
							<span className="score-tag">score {Number(item.score || 0).toFixed(2)}</span>
						</article>
					))}
				</div>
			</div>
		</section>
	);
}
