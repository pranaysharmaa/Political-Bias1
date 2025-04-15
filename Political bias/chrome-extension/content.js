// Function to extract text from the article
function getArticleText() {
  let articleText = "";
  
  // Assuming the article is inside <article> tag
  let articleElement = document.querySelector('article');
  if (articleElement) {
    articleText = articleElement.innerText;
  }
  
  return articleText;
}

// Function to highlight text based on the detected bias
function highlightBiasText(biasType) {
  let articleElement = document.querySelector('article');
  if (!articleElement) return;

  // Define the style for highlighting text based on the bias
  if (biasType === 'Left-leaning') {
    highlightText(articleElement, /left|liberal|progressive|democrat/gi, 'highlight-left');
  } else if (biasType === 'Right-leaning') {
    highlightText(articleElement, /right|conservative|republican|gop/gi, 'highlight-right');
  } else if (biasType === 'Neutral') {
    highlightText(articleElement, /neutral|balanced|unbiased|moderate/gi, 'highlight-neutral');
  }
}

// Helper function to highlight matching text with a specific CSS class
function highlightText(element, regex, className) {
  const textNodes = getTextNodesIn(element);

  textNodes.forEach(node => {
    const newText = node.nodeValue.replace(regex, match => `<span class="${className}">${match}</span>`);
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

// Listen for messages from the popup to highlight the bias text
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'highlightBias') {
    highlightBiasText(request.bias);  // 'request.bias' is the bias type
  }
});
