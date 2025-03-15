// Extracts article text from a webpage
function getArticleText() {
    let paragraphs = document.querySelectorAll("p");
    let articleText = Array.from(paragraphs).map(p => p.innerText).join(" ");
    return articleText;
}

// Send extracted text to background script
chrome.runtime.sendMessage({ action: "analyzeText", text: getArticleText() });
