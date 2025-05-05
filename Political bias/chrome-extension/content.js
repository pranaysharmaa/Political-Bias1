const API_URL = "https://cors-anywhere.herokuapp.com/https://biascheck-api.onrender.com/predict_bias";

async function checkAndHighlight(text, element) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const errText = await response.text();
      throw new Error(`Non-JSON response: ${errText.slice(0, 100)}...`);
    }

    const result = await response.json();
    const bias = result.bias_label;

    if (bias === "Left") {
      element.style.backgroundColor = "#ffcccc"; // red
    } else if (bias === "Right") {
      element.style.backgroundColor = "#cce5ff"; // blue
    } else if (bias === "Center") {
      element.style.backgroundColor = "#ffffcc"; // yellow
    }

    element.style.transition = "background-color 0.4s ease";
    console.log(`BiasCheck: "${text.slice(0, 60)}..." → ${bias}`);
  } catch (error) {
    console.error("BiasCheck error:", error);
  }
}

function getVisibleTextElements() {
  return Array.from(document.querySelectorAll("p, span, li, div"))
    .filter(el => el.innerText.trim().length > 40 && isVisible(el));
}

function isVisible(el) {
  const style = window.getComputedStyle(el);
  return style.display !== "none" && style.visibility !== "hidden" && el.offsetParent !== null;
}

async function processPage() {
  const elements = getVisibleTextElements();
  for (const el of elements) {
    await checkAndHighlight(el.innerText, el);
  }
}

window.addEventListener("load", processPage);
