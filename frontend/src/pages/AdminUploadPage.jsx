import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { apiRequest } from "../api";
import PageHeader from "../components/PageHeader";

export default function AdminUploadPage() {
  const navigate = useNavigate();

  const [checking, setChecking] = useState(true);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    async function checkAuth() {
      try {
        await apiRequest("/admin/me");
        setChecking(false);
      } catch {
        navigate("/admin/login");
      }
    }

    checkAuth();
  }, [navigate]);

  async function handleUpload(e) {
    e.preventDefault();

    if (!file) {
      setError("Please select an Excel file.");
      return;
    }

    setUploading(true);
    setError("");
    setMessage("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const data = await apiRequest("/admin/upload", {
        method: "POST",
        body: formData,
      });

      setResult(data);
      setMessage("Upload completed successfully.");
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleLogout() {
    try {
      await apiRequest("/admin/logout", {
        method: "POST",
      });
    } finally {
      navigate("/admin/login");
    }
  }

  if (checking) {
    return <div className="page centered">Checking session...</div>;
  }

  return (
    <div className="page">
      <PageHeader
        title="Admin Upload"
        subtitle="Upload apartment Excel data for indexing."
        compact={true}
      />

      <div className="topbar">
        <div className="topbar-actions">
          <Link to="/" className="btn btn-secondary">
            Home
          </Link>
          <button className="btn btn-primary" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>

      <div className="upload-card">
        <form onSubmit={handleUpload} className="form">
          <label>Excel File</label>
          <input
            type="file"
            accept=".xlsx,.xls"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            required
          />

          {message && <div className="success-box">{message}</div>}
          {error && <div className="error-box">{error}</div>}

          <button className="btn btn-primary" type="submit" disabled={uploading}>
            {uploading ? "Uploading..." : "Upload File"}
          </button>
        </form>

        {result && (
          <div className="result-card">
            <h2>Upload Result</h2>
            <p><strong>Message:</strong> {result.message}</p>
            {"apartments_count" in result && (
              <p><strong>Apartments Count:</strong> {result.apartments_count}</p>
            )}
            {"indexed_count" in result && (
              <p><strong>Indexed Count:</strong> {result.indexed_count}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}