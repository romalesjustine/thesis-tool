<link rel="stylesheet" href="./src/styles/about.css" />;

function About() {
  return (
    <>
      <div className="about-upper-section">
        <div className="upper-left">
          <h1>
            We turn news into insight{" "}
            <span className="h1-gray">with clarity and context</span>
          </h1>
        </div>
        <div className="upper-right">
          <p>
            Instantly get the gist of long news articles. LoReCast automatically
            generates concise summaries and topical tags for long news articles,
            enabling students, researchers, and professionals to accelerate
            their understanding and research.
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
            Information overload is a real challenge, with valuable details lost
            in lengthy news articles.
          </p>
          <p>
            While existing online algorithms have been the choice of many people
            for quick-read automation. However, this study found multiple issues
            in maintaining the readability and conciseness of the generated
            summaries, which often contain redundant information. While
            Longformer has demonstrated strong performance in terms of ROUGE
            scores, its inference speed (FPS) suggests that it may require more
            computational resources in practical applications.
          </p>
          <p>
            This model focuses on optimizing the structure to enhance
            conciseness and fluency, achieving more efficient summarization of
            long articles.
          </p>
          <p>
            What sets LORECAST apart from existing summarization AIs is its:
          </p>

          <li>
            Reinforcement learning (RL) component, which is designed to reduce
            redundancy and improve fluency in abstractive summarization.
          </li>
          <li>
            Zero-Shot Learning (ZSL) module provides contextual categorization
            using the baseline Longformer model.
          </li>

          <p>
            The integration of Zero-Shot Learning greatly helps in understanding
            the context of long articles, as it enables the system to identify
            the category of a given news item.
          </p>
        </div>
      </div>
    </>
  );
}

export default About;
