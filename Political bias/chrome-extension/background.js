chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeText") {
        fetch("https://your-api-url.com/analyze", {  // Replace with your deployed API URL
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: request.text })
        })
        .then(response => response.json())
        .then(data => {
            chrome.storage.local.set({ biasResult: data }, () => {
                console.log("Bias result saved:", data);
            });
        })
        .catch(error => console.error("API Error:", error));
    }
});
