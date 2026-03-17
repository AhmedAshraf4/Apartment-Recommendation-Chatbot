import logo from "../assets/cover.png";

export default function PageHeader({ title, subtitle }) {
  return (
    <div className="page-header">
      <div className="page-header__banner">
        <img className="page-header__image" src={logo} alt="Dorra cover" />
      </div>

      <div className="page-header__text-block">
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
    </div>
  );
}

