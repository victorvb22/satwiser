// Capture pages of the running app with the locally installed Chrome (dev tool).
// Usage: node scripts/screenshot.mjs <url> <out.png> [width] [height] [waitSelector] [buttonText]
import puppeteer from "puppeteer-core";

const [url, out, width = "1440", height = "1300", waitFor, clickText] = process.argv.slice(2);
const executablePath = process.env.CHROME_PATH
  ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const browser = await puppeteer.launch({ executablePath, headless: true,
                                         defaultViewport: { width: +width, height: +height } });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(url, { waitUntil: "networkidle2" });
if (waitFor) await page.waitForSelector(waitFor, { timeout: 20000 });
if (clickText) {
  const buttons = await page.$$("button");
  for (const b of buttons) {
    if ((await b.evaluate((el) => el.textContent)) === clickText) await b.click();
  }
}
await new Promise((r) => setTimeout(r, 1200));
await page.screenshot({ path: out, fullPage: true });
if (errors.length) console.log("PAGE ERRORS:\n" + errors.join("\n"));
await browser.close();
