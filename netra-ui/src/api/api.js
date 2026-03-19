import axios from "axios";
const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const getClusters = () => API.get("/clusters");

export const mergeClusters = (id1, id2) =>
  API.post("/merge", { id1, id2 });

export const getSuggestions = () => API.get("/suggestions");

export const searchFace = (file) => {
  const formData = new FormData();
  formData.append("file", file);

  return API.post("/search-face", formData);
};