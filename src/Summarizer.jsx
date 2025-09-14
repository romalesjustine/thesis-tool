import { useState } from "react";

function Summarizer() {
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
    <div className="summarizer-container">
      <div className="summarizer-header">
        <h1>Article Summarizer</h1>
        <p>Paste your article text below and get an instant summary with category classification</p>
      </div>

      <div className="input-section">
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste your article text here..."
          className="text-input"
          rows={10}
          disabled={isLoading}
        />

        <div className="button-group">
          <button
            onClick={handleSummarize}
            disabled={isLoading || !inputText.trim()}
            className="summarize-btn"
          >
            {isLoading ? "Summarizing..." : "Summarize Article"}
          </button>
          <button
            onClick={handleClear}
            className="clear-btn"
            disabled={isLoading}
          >
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

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
  );
}

export default Summarizer;