import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { apiRequest } from "../api";
import PageHeader from "../components/PageHeader";

export default function AdminLoginPage() {
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function checkAuth() {
      try {
        await apiRequest("/admin/me");
        navigate("/admin/upload");
      } catch {
        setChecking(false);
      }
    }

    checkAuth();
  }, [navigate]);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      await apiRequest("/admin/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      navigate("/admin/upload");
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  if (checking) {
    return <div className="page centered">Checking session...</div>;
  }

  return (
    <div className="page">
      <PageHeader
        title="Admin Login"
        subtitle="Enter your admin credentials to access the upload portal."
      />

      <div className="centered page-content">
        <div className="auth-card">
          <h2>Sign in</h2>
          <p className="muted">Authorized admin access only</p>

          <form onSubmit={handleSubmit} className="form">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              required
            />

            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              required
            />

            {error && <div className="error-box">{error}</div>}

            <button className="btn btn-primary full" type="submit" disabled={loading}>
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>

          <Link to="/" className="back-link">
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}