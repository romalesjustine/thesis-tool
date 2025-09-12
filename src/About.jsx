<link rel="stylesheet" href="./src/styles/about.css" />;

function About() {
  return (
    <>
      <div className="video-background">
        <video autoPlay muted loop>
          <source src="./src/assets/background.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
      <div className="about-upper-section">
        <div className="upper-left">
          <h1>
            We turn news into insight{" "}
            <span className="h1-gray">with clarity and context</span>
          </h1>
        </div>
        <div className="upper-right">
          <p>
            LoReCast uses algorithm to help readers quickly understand long news
            articles by generating concise summaries and tagging their topics —
            automatically. Built for students, researchers, and everyday
            readers.
          </p>
        </div>
      </div>

      <img
        src="./src/assets/grouppic.jpg"
        alt="Group Photo"
        className="group-photo"
      />

      <hr className="hr-upper" />

      <div className="second-section">
        <div className="second-upper">
          <h1>Together we are building smarter readers</h1>
        </div>
        <div className="second-bottom">
          <p>
            Our mission is to help people navigate the overwhelming volume of
            digital information by turning long-form news into clear,
            categorized insights.
          </p>
          <p>
            In today’s world, reading through full-length news articles can be
            time-consuming — especially when all you need is the core message.
            LoReCast is a smart tool that helps you instantly understand news
            articles by automatically generating accurate summaries and
            assigning topic labels.
          </p>
          <p>We’ve designed LoReCast using cutting-edge AI:</p>
          <ul>
            <li>
              Longformer for handling long documents and generating concise
              summaries
            </li>
            <li>
              Reinforcement Learning to improve the quality and relevance of
              each output
            </li>
            <li>
              Zero-shot Learning to accurately tag topics without needing
              pre-labeled training data
            </li>
          </ul>
          <p>
            This project was built to support students, researchers, and
            everyday readers in navigating information overload with speed and
            confidence.
          </p>
        </div>
      </div>

      <hr className="hr-bottom" />

      <div className="third-section">
        <div className="third-upper">
          <h1>Meet our amazing Team</h1>
        </div>
      </div>
    </>
  );
}

export default About;
