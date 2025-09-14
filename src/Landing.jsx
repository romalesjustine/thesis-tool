import { useState } from "react";

<link rel="stylesheet" href="./src/styles/landing.css" />;

function Landing() {
  const [inputText, setInputText] = useState("");
  const [summary, setSummary] = useState("");
  const [category, setCategory] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError("Please enter text to summarize");
      return;
    }

    setIsLoading(true);
    setError("");
    setSummary("");
    setCategory("");

    try {
      const response = await fetch("/api/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error("Failed to summarize text");
      }

      const data = await response.json();
      setSummary(data.summary);
      setCategory(data.category);
    } catch (err) {
      setError(err.message || "An error occurred while summarizing");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setInputText("");
    setSummary("");
    setCategory("");
    setError("");
  };

  return (
    <>
      <div className="video-background">
        <video autoPlay muted loop>
          <source src="./src/assets/background.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
      <div className="hero-container">
        <h1>
          Transform long news articles into concise, insightful
          summaries—automatically categorized in seconds.
        </h1>
        <p>
          Paste the text or upload your preferred news article, and let LORECAST
          generate a clear summary with its relevant category.
        </p>

        <div className="summarizer-section">
          <div className="text-input-wrapper">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste your article text here..."
              className="text-input"
              rows={8}
              disabled={isLoading}
            />
            <button
              onClick={handleClear}
              className="clear-btn"
              disabled={isLoading}
            >
              <i className="fa fa-trash-alt"></i>
            </button>

            <button
              onClick={handleSummarize}
              disabled={isLoading || !inputText.trim()}
              className="summarize-btn"
            >
              {isLoading ? "Summarizing..." : "Summarize Article "} ✨
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {(summary || category) && (
            <div className="results-section">
              {category && (
                <div className="category-result">
                  <h3>Category</h3>
                  <span className="category-tag">{category}</span>
                </div>
              )}

              {summary && (
                <div className="summary-result">
                  <h3>Summary</h3>
                  <p>{summary}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Landing;
