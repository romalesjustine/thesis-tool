import React from 'react';

const Howitworks = () => {
  const steps = [
    {
      number: "01",
      text: "Paste the text or upload a news file. LORECAST supports common formats like .txt, .pdf, or even direct links."
    },
    {
      number: "02",
      text: "Our system parses the content to extract key parts: title, body, embedded metadata, and tone cues. This step uses NLP techniques to normalize and clean the data for better summarization results."
    },
    {
      number: "03",
      text: "Using transformer-based modeling (Longformer), LORECAST generates a short, context-rich summary. It keeps essential points while eliminating noise."
    },
    {
      number: "04",
      text: "Instead of relying on manual tags, LORECAST uses zero-shot classification to assign your article to the most relevant category—like Politics, Sports, Business, Tech, and more—based on semantic matching."
    },
    {
      number: "05",
      text: "Once LORECAST completes the summarization and categorization, results are shown in a clear layout. Users can download, copy, or submit a new article for fresh analysis—all within a sleek, easy-to-use interface."
    }
  ];

  return (
    <>
      {/* Hero Section */}
      <div className="howitworks-hero">
        <div className="hero-content">
          <h1>
            How <span className="highlight">LoReCast</span> Works
          </h1>
          <p>
            Transform any news article into a concise summary with automatic
            categorization with our advanced, algorithm-driven system.
          </p>
        </div>
      </div>

      {/* Animated Infinite Carousel: 01→02→03→04→05 → loop */}
      <div className="animation-container">
        <div className="layer-background">
          <div className="step-flow">
            <div className="step-container">
              {steps.map((step, index) => (
                <div className="step-card" key={`step-${index}`}>
                  <div className="step-number">{step.number}</div>
                  <div className="step-content">
                    <p>{step.text}</p>
                  </div>
                </div>
              ))}

              {/* Second Set: Duplicated for seamless loop */}
              {steps.map((step, index) => (
                <div className="step-card" key={`second-${index}`}>
                  <div className="step-number">{step.number}</div>
                  <div className="step-content">
                    <p>{step.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Divider */}
      <hr className="section-divider" />

     {/* CTA */}
    <div className="cta-section">
      <div className="cta-content">
        <h2>Ready to Transform Your Reading Experience?</h2>
        <p>
          Start summarizing lengthy news articles in seconds with LoReCast's
          intelligent algorithm system.
        </p>
        <a
          href="/"
          className="cta-button"
          aria-label="Go to LoReCast homepage"
        >
          Try LoReCast Now
        </a>
      </div>
    </div>
    </>
  );
};

export default React.memo(Howitworks);