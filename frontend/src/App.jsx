import { useState } from "react";
import "./index.css";

function App() {
  const [news, setNews] = useState("");
  const [result, setResult] = useState(null);

  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);

  // ---------------------------------------------------------
  // Temporary prediction
  // Flask API connection will be added in the next step.
  // ---------------------------------------------------------

  const handleCheckNews = () => {
    if (!news.trim()) {
      alert("Please enter a news headline or article.");
      return;
    }

    setResult({
      prediction: "REAL",
      confidence: 99.28,
      signals: [
        "technology",
        "research",
        "system",
        "energy",
        "new"
      ]
    });
  };

  return (
    <div className="app">

      {/* =====================================================
          NAVBAR
      ===================================================== */}

      <nav className="navbar">

        <div className="logo">
          <span className="logo-icon">✓</span>
          Fake News Detector
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#about">About</a>
          <a href="#how">How It Works</a>

          <button
            className="login-nav-btn"
            onClick={() => setShowLogin(true)}
          >
            Login
          </button>
        </div>

      </nav>


      {/* =====================================================
          HERO / NEWS DETECTION SECTION
      ===================================================== */}

      <section className="hero" id="home">

        <div className="hero-content">

          <div className="badge">
            AI + NLP + Machine Learning
          </div>

          <h1>
            AI-Powered
            <span> Fake News Detector</span>
          </h1>

          <p>
            Analyze news content using Natural Language Processing
            and Machine Learning to identify whether it is
            Real or Fake.
          </p>

        </div>


        {/* NEWS INPUT CARD */}

        <div className="detector-card">

          <h2>Check Your News</h2>

          <p className="detector-description">
            Paste a news headline or article below and let our
            machine learning model analyze it.
          </p>

          <textarea
            value={news}
            onChange={(e) => setNews(e.target.value)}
            placeholder="Paste your news headline or article here..."
            rows="8"
          />

          <button
            className="check-btn"
            onClick={handleCheckNews}
          >
            Check News
          </button>


          {/* =================================================
              RESULT CARD
          ================================================= */}

          {result && (
            <div className="result-card">

              <div className="result-header">
                <h3>Prediction Result</h3>
              </div>

              <div
                className={`prediction-badge ${
                  result.prediction === "REAL"
                    ? "real"
                    : "fake"
                }`}
              >
                {result.prediction}
              </div>

              <div className="confidence-section">

                <div className="confidence-title">
                  <span>Confidence</span>
                  <strong>{result.confidence}%</strong>
                </div>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${result.confidence}%`
                    }}
                  ></div>
                </div>

              </div>


              <div className="signals-section">

                <h4>Model Signals</h4>

                <div className="signals">

                  {result.signals.map((signal, index) => (
                    <span key={index}>
                      {signal}
                    </span>
                  ))}

                </div>

              </div>

            </div>
          )}

        </div>

      </section>


      {/* =====================================================
          ABOUT SECTION
      ===================================================== */}

      <section className="about" id="about">

        <div className="section-heading">
          <span>ABOUT THE PROJECT</span>

          <h2>
            Intelligent News Verification
          </h2>

          <p>
            Our Fake News Detector uses Natural Language Processing
            and Machine Learning techniques to analyze news content
            and classify it as Real or Fake.
          </p>
        </div>


        <div className="feature-grid">

          <div className="feature-card">
            <div className="feature-icon">AI</div>

            <h3>Machine Learning</h3>

            <p>
              Uses a trained machine learning classifier to
              identify patterns in news content.
            </p>
          </div>


          <div className="feature-card">
            <div className="feature-icon">NLP</div>

            <h3>Natural Language Processing</h3>

            <p>
              Cleans and processes textual news data before
              making a prediction.
            </p>
          </div>


          <div className="feature-card">
            <div className="feature-icon">TF</div>

            <h3>TF-IDF Analysis</h3>

            <p>
              Converts news text into numerical features that
              can be analyzed by the ML model.
            </p>
          </div>

        </div>

      </section>


      {/* =====================================================
          HOW IT WORKS
      ===================================================== */}

      <section className="how" id="how">

        <div className="section-heading">

          <span>PROCESS</span>

          <h2>How It Works</h2>

          <p>
            The system follows a simple machine learning pipeline
            to analyze the submitted news.
          </p>

        </div>


        <div className="steps">

          <div className="step-card">
            <div className="step-number">1</div>
            <h3>News Input</h3>
            <p>
              User enters a news headline or article.
            </p>
          </div>


          <div className="step-card">
            <div className="step-number">2</div>
            <h3>Preprocessing</h3>
            <p>
              Text is cleaned and prepared using NLP.
            </p>
          </div>


          <div className="step-card">
            <div className="step-number">3</div>
            <h3>TF-IDF</h3>
            <p>
              Text is converted into numerical features.
            </p>
          </div>


          <div className="step-card">
            <div className="step-number">4</div>
            <h3>ML Model</h3>
            <p>
              The trained classifier analyzes the features.
            </p>
          </div>


          <div className="step-card">
            <div className="step-number">5</div>
            <h3>Prediction</h3>
            <p>
              System displays the Real or Fake result.
            </p>
          </div>

        </div>

      </section>


      {/* =====================================================
          FOOTER
      ===================================================== */}

      <footer className="footer">

        <h3>Fake News Detector</h3>

        <p>
          AI / ML Based Major Project
        </p>

        <p className="copyright">
          © 2026 Fake News Detector. All rights reserved.
        </p>

      </footer>


      {/* =====================================================
          LOGIN MODAL
      ===================================================== */}

      {showLogin && (

        <div className="modal-overlay">

          <div className="modal">

            <button
              className="close-btn"
              onClick={() => setShowLogin(false)}
            >
              ×
            </button>

            <div className="modal-header">
              <h2>Welcome Back</h2>

              <p>
                Login to continue
              </p>
            </div>


            <input
              type="email"
              placeholder="Email address"
            />

            <input
              type="password"
              placeholder="Password"
            />

            <button className="modal-btn">
              Login
            </button>


            <p className="switch-text">

              Don't have an account?

              <button
                className="text-btn"
                onClick={() => {
                  setShowLogin(false);
                  setShowRegister(true);
                }}
              >
                Register
              </button>

            </p>

          </div>

        </div>

      )}


      {/* =====================================================
          REGISTER MODAL
      ===================================================== */}

      {showRegister && (

        <div className="modal-overlay">

          <div className="modal">

            <button
              className="close-btn"
              onClick={() => setShowRegister(false)}
            >
              ×
            </button>

            <div className="modal-header">

              <h2>Create Account</h2>

              <p>
                Register to get started
              </p>

            </div>


            <input
              type="text"
              placeholder="Full name"
            />

            <input
              type="email"
              placeholder="Email address"
            />

            <input
              type="password"
              placeholder="Password"
            />

            <button className="modal-btn">
              Register
            </button>


            <p className="switch-text">

              Already have an account?

              <button
                className="text-btn"
                onClick={() => {
                  setShowRegister(false);
                  setShowLogin(true);
                }}
              >
                Login
              </button>

            </p>

          </div>

        </div>

      )}

    </div>
  );
}

export default App;