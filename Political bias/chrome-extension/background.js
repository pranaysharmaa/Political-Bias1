// background.js

const API_URL = "https://biascheck-api.onrender.com/predict_bias";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "getApiUrl") {
    sendResponse({ apiUrl: API_URL });
  }
});
