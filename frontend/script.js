// Frontend logic for the single-page Fake News Detection demo UI.
// Talks to the FastAPI backend's POST /predict endpoint on the same origin.

const articleTextEl = document.getElementById("articleText");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const errorAlertEl = document.getElementById("errorAlert");
const resultSectionEl = document.getElementById("resultSection");
const predictionBadgeEl = document.getElementById("predictionBadge");
const confidenceValueEl = document.getElementById("confidenceValue");
const modelValueEl = document.getElementById("modelValue");

function showError(message) {
  errorAlertEl.textContent = message;
  errorAlertEl.style.display = "block";
  resultSectionEl.style.display = "none";
}

function hideError() {
  errorAlertEl.style.display = "none";
}

function showResult(data) {
  resultSectionEl.style.display = "block";

  const isReal = data.prediction === "Real";
  predictionBadgeEl.textContent = data.prediction.toUpperCase();
  predictionBadgeEl.className = "fw-bold mt-1 " + (isReal ? "text-success" : "text-danger");

  confidenceValueEl.textContent = data.confidence.toFixed(2) + " %";
  modelValueEl.textContent = data.model;
}

async function handlePredict() {
  const text = articleTextEl.value.trim();
  hideError();

  if (!text) {
    showError("Please enter a news article.");
    return;
  }

  predictBtn.disabled = true;
  const originalLabel = predictBtn.textContent;
  predictBtn.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      let detail = "Prediction failed.";
      try {
        const errorBody = await response.json();
        if (errorBody && errorBody.detail) {
          detail = errorBody.detail;
        }
      } catch (parseError) {
        // response wasn't valid JSON - fall back to the generic message above
      }
      showError(detail);
      return;
    }

    const data = await response.json();
    showResult(data);
  } catch (networkError) {
    showError("Unable to contact server.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = originalLabel;
  }
}

function handleClear() {
  articleTextEl.value = "";
  hideError();
  resultSectionEl.style.display = "none";
}

predictBtn.addEventListener("click", handlePredict);
clearBtn.addEventListener("click", handleClear);
