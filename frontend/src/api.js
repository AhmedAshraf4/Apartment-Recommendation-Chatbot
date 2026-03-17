export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      (typeof data === "string" ? data : "Request failed");
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }

  return data;
}