function Landing() {
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
          Transform long news articles into concise,
          insightfulsummaries—automatically categorized in seconds.
        </h1>
        <p>
          Paste the text or upload your preferred news article, and let LORECAST
          generate a clear summary with its relevant category.
        </p>
      </div>
    </>
  );
}

export default Landing;
