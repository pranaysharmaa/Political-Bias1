// popup.js

document.addEventListener("DOMContentLoaded", function() {
    const startButton = document.getElementById("startButton");
    const statusText = document.getElementById("statusText");
    const loader = document.getElementById("loader");
  
    // Listen for status updates from the content script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === "statusUpdate") {
        statusText.textContent = message.status;
        // Hide loader if the process is complete or an error occurred
        if (
          message.status === "Bias highlighted." ||
          message.status === "No bias detected." ||
          message.status.startsWith("Error")
        ) {
          loader.style.display = "none";
        } else {
          loader.style.display = "inline-block";
        }
      }
    });
  
    startButton.addEventListener("click", function() {
      // Reset status and show loader
      statusText.textContent = "Initializing...";
      loader.style.display = "inline-block";
  
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length === 0) {
          statusText.textContent = "No active tab found.";
          return;
        }
        const tabId = tabs[0].id;
        
        // Dynamically inject the content script into the active tab
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ["content.js"]
        }, () => {
          if (chrome.runtime.lastError) {
            statusText.textContent = "Error injecting content script: " + chrome.runtime.lastError.message;
            console.error("Error injecting content script:", chrome.runtime.lastError.message);
          } else {
            // After injecting the content script, send a message to start bias detection
            chrome.tabs.sendMessage(tabId, { type: "startBiasDetection" }, (response) => {
              if (chrome.runtime.lastError) {
                statusText.textContent = "Error: " + chrome.runtime.lastError.message;
                console.error("Error sending message to content script:", chrome.runtime.lastError.message);
              }
            });
          }
        });
      });
    });
  });
  
