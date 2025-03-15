document.addEventListener("DOMContentLoaded", () => {
    chrome.storage.local.get("biasResult", (data) => {
        if (data.biasResult) {
            document.getElementById("biasResult").innerText = 
                `Bias Detected: ${data.biasResult.bias}`;
        } else {
            document.getElementById("biasResult").innerText = "No data available.";
        }
    });
});

