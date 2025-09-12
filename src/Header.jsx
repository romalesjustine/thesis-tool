import { Link } from "react-router-dom";

function Header() {
  return (
    <header>
      <div className="header-container">
        <Link to="/">
          <img src="./src/assets/logo.png" alt="Logo" />
        </Link>
        <div className="nav-links">
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/about">About</Link>
            </li>
            <li>
              <Link to="/howitworks"
      >How it works</Link>
            </li>
          </ul>
        </div>
      </div>
    </header>
  );
}

export default Header;
