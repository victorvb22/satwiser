// Capture pages at phone size and report horizontal overflow (dev tool).
// Usage: node scripts/mobile-check.mjs <base url> <out dir> [width]
import puppeteer from "puppeteer-core";

const [base, outDir, width = "390"] = process.argv.slice(2);
const executablePath = process.env.CHROME_PATH
  ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const browser = await puppeteer.launch({ executablePath, headless: true });
const page = await browser.newPage();
await page.setViewport({ width: +width, height: 844, isMobile: true, hasTouch: true,
                         deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
for (const path of ["/", "/labo", "/methode"]) {
  await page.goto(base + path, { waitUntil: "networkidle2" });
  await new Promise((r) => setTimeout(r, 1500));
  // Scroll through the page so that blocks revealed on scroll are shown in the capture.
  const total = await page.evaluate(() => document.documentElement.scrollHeight);
  for (let y = 0; y <= total; y += 400) {
    await page.evaluate((v) => window.scrollTo(0, v), y);
    await new Promise((r) => setTimeout(r, 120));
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await new Promise((r) => setTimeout(r, 1000));
  const overflow = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    const wide = [...document.querySelectorAll("main *, nav *, footer *")]
      .filter((el) => el.getBoundingClientRect().right > vw + 1)
      .filter((el) => !el.closest(".pills")).slice(-8)
      .map((el) => `${el.tagName.toLowerCase()}.${el.className} → ${Math.round(el.getBoundingClientRect().right)}`);
    return { scrollWidth: document.documentElement.scrollWidth, vw, wide };
  });
  console.log(path, JSON.stringify(overflow));
  const name = path === "/" ? "mission" : path.slice(1);
  await page.screenshot({ path: `${outDir}/m_${name}.png`, fullPage: true });
}
if (errors.length) console.log("PAGE ERRORS:\n" + errors.join("\n"));
await browser.close();
