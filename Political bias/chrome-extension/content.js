// content.js

(function() {
  console.log("Political Bias Highlighter content script loaded.");

  // Function to send status updates back to the popup
  function updateStatus(status) {
    chrome.runtime.sendMessage({ type: "statusUpdate", status: status });
  }

  // Function to extract text from the page (here, from paragraph tags)
  function extractText() {
    updateStatus("Extracting content...");
    const paragraphs = document.querySelectorAll("p");
    let fullText = "";
    paragraphs.forEach(p => {
      fullText += p.innerText + "\n";
    });
    return { paragraphs, fullText };
  }

  // Helper function to highlight paragraphs (simple yellow background)
  function highlightParagraphs(paragraphs) {
    paragraphs.forEach(p => {
      p.style.backgroundColor = "yellow";
    });
    updateStatus("Bias highlighted.");
  }

  // Listen for the command from the popup to start the bias detection
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "startBiasDetection") {
      updateStatus("Starting bias detection...");
      
      // Extract content
      const { paragraphs, fullText } = extractText();
      if (fullText.trim().length === 0) {
        updateStatus("No text content found.");
        sendResponse({ status: "noContent" });
        return;
      }

      // Request API URL from the background script
      chrome.runtime.sendMessage({ type: "getApiUrl" }, (response) => {
        const apiUrl = response.apiUrl;
        if (!apiUrl) {
          updateStatus("Error: API URL not found.");
          sendResponse({ status: "apiUrlNotFound" });
          return;
        }

        updateStatus("Detecting bias...");

        // Call the bias detection API with the extracted text
        fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ text: fullText })
        })
        .then(res => res.json())
        .then(data => {
          console.log("Bias detection response:", data);
          // Assuming the API returns { biasFound: true } when bias is detected.
          if (data && data.biasFound) {
            highlightParagraphs(paragraphs);
            sendResponse({ status: "biasHighlighted" });
          } else {
            updateStatus("No bias detected.");
            sendResponse({ status: "noBias" });
          }
        })
        .catch(err => {
          console.error("Error during bias detection:", err);
          updateStatus("Error during bias detection.");
          sendResponse({ status: "error", error: err.toString() });
        });
      });

      // Indicate that the response will be sent asynchronously.
      return true;
    }
  });
})();
