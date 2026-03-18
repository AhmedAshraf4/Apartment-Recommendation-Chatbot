import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiRequest } from "../api";
import PageHeader from "../components/PageHeader";

export default function AdminUploadPage() {
  const navigate = useNavigate();

  const [isCheckingSession, setIsCheckingSession] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadResult, setUploadResult] = useState(null);

  useEffect(() => {
    const verifyAdminSession = async () => {
      try {
        await apiRequest("/admin/me");
        setIsCheckingSession(false);
      } catch {
        navigate("/admin/login");
      }
    };

    verifyAdminSession();
  }, [navigate]);

  const handleUpload = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setErrorMessage("Please select an Excel file.");
      return;
    }

    setIsUploading(true);
    setErrorMessage("");
    setSuccessMessage("");
    setUploadResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await apiRequest("/admin/upload", {
        method: "POST",
        body: formData,
      });

      setUploadResult(response);
      setSuccessMessage("Upload completed successfully.");
    } catch (error) {
      setErrorMessage(error.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await apiRequest("/admin/logout", {
        method: "POST",
      });
    } finally {
      navigate("/admin/login");
    }
  };

  if (isCheckingSession) {
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
        <form className="form" onSubmit={handleUpload}>
          <label>Excel File</label>
          <input
            type="file"
            accept=".xlsx,.xls"
            onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
            required
          />

          {successMessage && <div className="success-box">{successMessage}</div>}
          {errorMessage && <div className="error-box">{errorMessage}</div>}

          <button className="btn btn-primary" type="submit" disabled={isUploading}>
            {isUploading ? "Uploading..." : "Upload File"}
          </button>
        </form>

        {uploadResult && (
          <div className="result-card">
            <h2>Upload Result</h2>
            <p>
              <strong>Message:</strong> {uploadResult.message}
            </p>
            {"apartments_count" in uploadResult && (
              <p>
                <strong>Apartments Count:</strong> {uploadResult.apartments_count}
              </p>
            )}
            {"indexed_count" in uploadResult && (
              <p>
                <strong>Indexed Count:</strong> {uploadResult.indexed_count}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}