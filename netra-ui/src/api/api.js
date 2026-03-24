import axios from "axios";
const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});


export const getClusters = () => API.get("/clusters");

export const mergeClusters = (id1, id2) =>
  API.post("/merge", { id1, id2 });

export const getSuggestions = (params = { threshold: 0.3, limit: 20 }) =>
  API.get("/suggestions", { params });

export const searchFace = (file) => {
  const formData = new FormData();
  formData.append("file", file);

  return API.post("/search-face", formData);
};

export const getObjectsByImage = (imageName) => API.get(`/objects/${imageName}`);

export const getGraph = () => API.get("/graph");
export const rebuildGraph = () => API.post("/graph/rebuild");
export const getClustersPaginated = (page = 1, limit = 20) => API.get(`/clusters/paginated?page=${page}&limit=${limit}`);
export const getTimeline = () => API.get("/timeline");
export const getContentGroups = () => API.get("/content-groups");
