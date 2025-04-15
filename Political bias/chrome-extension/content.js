// Example of content script that extracts text from an article
function getArticleText() {
  let articleText = "";
  
  // Assuming the article is inside <article> tag
  let articleElement = document.querySelector('article');
  if (articleElement) {
    articleText = articleElement.innerText;
  }
  
  return articleText;
}

// Function to highlight text based on bias
function highlightBiasText(biasType) {
  let articleElement = document.querySelector('article');
  if (!articleElement) return;

  // Define the style for highlighting text
  const highlightStyle = 'background-color: yellow; font-weight: bold;';
  
  // Based on the detected bias, apply styles to the article
  if (biasType === 'Left-leaning') {
    highlightText(articleElement, /left|liberal|progressive/gi, highlightStyle);  // Example keywords
  } else if (biasType === 'Right-leaning') {
    highlightText(articleElement, /right|conservative|republican/gi, highlightStyle);  // Example keywords
  } else if (biasType === 'Neutral') {
    highlightText(articleElement, /neutral|balanced|unbiased/gi, highlightStyle);
  }
}

// Helper function to highlight matching text
function highlightText(element, regex, style) {
  const textNodes = getTextNodesIn(element);

  textNodes.forEach(node => {
    const newText = node.nodeValue.replace(regex, match => `<span style="${style}">${match}</span>`);
    const span = document.createElement('span');
    span.innerHTML = newText;
    node.replaceWith(span);
  });
}

// Helper function to get all text nodes within an element
function getTextNodesIn(element) {
  const textNodes = [];
  const walk = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );
  
  let node;
  while (node = walk.nextNode()) {
    textNodes.push(node);
  }
  return textNodes;
}

// Listen for messages from the popup and highlight based on bias detected
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'highlightBias') {
    highlightBiasText(request.bias);  // 'request.bias' is the bias type (e.g., 'Left-leaning', 'Right-leaning', etc.)
  }
});
