const playwright = require('playwright');

(async () => {
  const browser = await playwright.chromium.launch();
  const page = await browser.newPage();
  await page.goto('file:///app/interactive_code/Neural_Dashboard_Interactive.html');
  await page.screenshot({ path: 'generated_images/screenshot.png' });
  await browser.close();
})();
