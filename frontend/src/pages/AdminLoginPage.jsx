import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiRequest } from "../api";
import PageHeader from "../components/PageHeader";

export default function AdminLoginPage() {
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCheckingSession, setIsCheckingSession] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const verifySession = async () => {
      try {
        await apiRequest("/admin/me");
        navigate("/admin/upload");
      } catch {
        setIsCheckingSession(false);
      }
    };

    verifySession();
  }, [navigate]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setErrorMessage("");

    try {
      await apiRequest("/admin/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      navigate("/admin/upload");
    } catch (error) {
      setErrorMessage(error.message || "Login failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isCheckingSession) {
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

          <form className="form" onSubmit={handleSubmit}>
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Enter username"
              required
            />

            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter password"
              required
            />

            {errorMessage && <div className="error-box">{errorMessage}</div>}

            <button
              className="btn btn-primary full"
              type="submit"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Logging in..." : "Login"}
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