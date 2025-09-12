import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";
import Landing from "./Landing";
import About from "./About";
import Howitworks from "./Howitworks";

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/about" element={<About />} />
        <Route path="/howitworks" element={<Howitworks />} />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
