import { Link } from "react-router-dom";
import PageHeader from "../components/PageHeader";

export default function HomePage() {
  return (
    <div className="page">
      <PageHeader
        title="Smart apartment search and admin management"
        subtitle="Choose your portal to continue."
      />

      <section className="section">
        <div className="cards">
          <div className="card">
            <h2>Admin</h2>
            <p>Login and upload apartment Excel data for indexing.</p>
            <Link to="/admin/login" className="btn btn-primary">
              Go to Admin
            </Link>
          </div>

          <div className="card">
            <h2>User</h2>
            <p>Browse and chat with the assistant to find properties.</p>
            <Link to="/user" className="btn btn-secondary">
              Go to User
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}