const MAX_DIMENSION = 720;
const SAMPLE_URL = "../examples/input.jpg";

const els = {
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  browseBtn: document.getElementById("browse-btn"),
  sampleBtn: document.getElementById("sample-btn"),
  resultArea: document.getElementById("result-area"),
  inputCanvas: document.getElementById("input-canvas"),
  outputCanvas: document.getElementById("output-canvas"),
  downloadLink: document.getElementById("download-link"),
  status: document.getElementById("status"),
  error: document.getElementById("error"),
  loading: document.getElementById("loading"),
};

let cvReady = false;
let currentImage = null;

function onOpenCvReady() {
  cv["onRuntimeInitialized"] = () => {
    cvReady = true;
    els.status.textContent = "OpenCV loaded — drop a photo to start.";
    if (currentImage) processCurrent();
  };
}

function onOpenCvError() {
  showError("Could not load OpenCV.js. Check your internet connection and reload.");
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
}

function selectedStyle() {
  return document.querySelector('input[name="style"]:checked').value;
}

function loadImage(src) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    currentImage = img;
    processCurrent();
  };
  img.onerror = () => showError("Could not load that image.");
  img.src = src;
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please choose an image file.");
    return;
  }
  clearError();
  loadImage(URL.createObjectURL(file));
}

function drawToCanvas(canvas, img) {
  const maxDim = MAX_DIMENSION;
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
}

function matFromCanvas(canvas) {
  return cv.imread(canvas);
}

function toGray(src) {
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  return gray;
}

function toRGB(src) {
  const rgb = new cv.Mat();
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  return rgb;
}

function bilateralN(rgb, iterations, sigmaColor, sigmaSpace) {
  let result = rgb;
  for (let i = 0; i < iterations; i++) {
    const next = new cv.Mat();
    cv.bilateralFilter(result, next, 9, sigmaColor, sigmaSpace);
    if (i > 0) result.delete();
    result = next;
  }
  return result;
}

function classicCartoon(src) {
  const gray = toGray(src);
  const smooth = new cv.Mat();
  cv.medianBlur(gray, smooth, 5);

  const edges = new cv.Mat();
  cv.adaptiveThreshold(
    smooth, edges, 255,
    cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9
  );

  const rgb = toRGB(src);
  const flat = bilateralN(rgb, 2, 250, 250);

  const cartoonRGB = new cv.Mat();
  cv.bitwise_and(flat, flat, cartoonRGB, edges);

  const cartoon = new cv.Mat();
  cv.cvtColor(cartoonRGB, cartoon, cv.COLOR_RGB2RGBA);

  gray.delete(); smooth.delete(); edges.delete(); rgb.delete(); flat.delete();
  cartoonRGB.delete();
  return cartoon;
}

function sketchCartoon(src) {
  const gray = toGray(src);
  const inverted = new cv.Mat();
  cv.bitwise_not(gray, inverted);

  const blurred = new cv.Mat();
  cv.GaussianBlur(inverted, blurred, new cv.Size(21, 21), 0);

  const divisor = new cv.Mat();
  cv.bitwise_not(blurred, divisor);

  const sketch = new cv.Mat();
  cv.divide(gray, divisor, sketch, 256, -1);
  cv.normalize(sketch, sketch, 0, 255, cv.NORM_MINMAX);

  const result = new cv.Mat();
  cv.cvtColor(sketch, result, cv.COLOR_GRAY2RGBA);

  gray.delete(); inverted.delete(); blurred.delete(); divisor.delete(); sketch.delete();
  return result;
}

function posterize(rgb, k) {
  const rows = rgb.rows * rgb.cols;
  const samples = new cv.Mat(rows, 3, cv.CV_32F);
  const data = rgb.data;
  for (let i = 0; i < rows; i++) {
    samples.data32F[i * 3] = data[i * 3];
    samples.data32F[i * 3 + 1] = data[i * 3 + 1];
    samples.data32F[i * 3 + 2] = data[i * 3 + 2];
  }

  const labels = new cv.Mat();
  const centers = new cv.Mat();
  const criteria = new cv.TermCriteria(
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0
  );
  cv.kmeans(samples, k, labels, criteria, 3, cv.KMEANS_PP_CENTERS, centers);

  const centers8 = new cv.Mat();
  centers.convertTo(centers8, cv.CV_8U);

  const flat = new cv.Mat(rows, 1, cv.CV_8UC3);
  for (let i = 0; i < rows; i++) {
    const c = labels.data32S[i] * 3;
    flat.data[i * 3] = centers8.data[c];
    flat.data[i * 3 + 1] = centers8.data[c + 1];
    flat.data[i * 3 + 2] = centers8.data[c + 2];
  }

  const result = new cv.Mat(rgb.rows, rgb.cols, cv.CV_8UC3);
  result.data.set(flat.data);

  samples.delete(); labels.delete(); centers.delete(); centers8.delete(); flat.delete();
  return result;
}

function boostSaturation(rgb, factor) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const n = hsv.rows * hsv.cols;
  for (let i = 0; i < n; i++) {
    const idx = i * 3 + 1;
    const s = hsv.data[idx] * factor;
    hsv.data[idx] = s > 255 ? 255 : s;
  }
  const out = new cv.Mat();
  cv.cvtColor(hsv, out, cv.COLOR_HSV2RGB);
  hsv.delete();
  return out;
}

function comicCartoon(src) {
  const gray = toGray(src);
  const smooth = new cv.Mat();
  cv.medianBlur(gray, smooth, 5);

  let edges = new cv.Mat();
  cv.adaptiveThreshold(
    smooth, edges, 255,
    cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9
  );
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const thickEdges = new cv.Mat();
  cv.dilate(edges, thickEdges, kernel, new cv.Point(-1, -1), 2);
  edges.delete();
  edges = thickEdges;

  const rgb = toRGB(src);
  const preSmooth = bilateralN(rgb, 1, 60, 60);
  const flat0 = posterize(preSmooth, 10);
  preSmooth.delete();
  const flat = boostSaturation(flat0, 1.4);
  flat0.delete();

  const cartoonRGB = new cv.Mat();
  cv.bitwise_and(flat, flat, cartoonRGB, edges);

  const cartoon = new cv.Mat();
  cv.cvtColor(cartoonRGB, cartoon, cv.COLOR_RGB2RGBA);

  gray.delete(); smooth.delete(); edges.delete(); rgb.delete();
  flat.delete(); cartoonRGB.delete(); kernel.delete();
  return cartoon;
}

function hdrCartoon(src) {
  const rgb = toRGB(src);
  let smooth = new cv.Mat();
  cv.bilateralFilter(rgb, smooth, 9, 60, 60);
  const smooth2 = new cv.Mat();
  cv.bilateralFilter(smooth, smooth2, 9, 60, 60);
  smooth.delete();

  const boosted = new cv.Mat();
  cv.addWeighted(rgb, 1.6, smooth2, -0.6, 0, boosted);

  const vivid = boostSaturation(boosted, 1.35);

  const out = new cv.Mat();
  cv.cvtColor(vivid, out, cv.COLOR_RGB2RGBA);
  rgb.delete(); smooth2.delete(); boosted.delete(); vivid.delete();
  return out;
}

function cartoonify(src, style) {
  switch (style) {
    case "sketch": return sketchCartoon(src);
    case "comic": return comicCartoon(src);
    case "hdr": return hdrCartoon(src);
    default: return classicCartoon(src);
  }
}

function processCurrent() {
  if (!cvReady || !currentImage) return;

  clearError();
  drawToCanvas(els.inputCanvas, currentImage);
  els.resultArea.hidden = false;
  els.downloadLink.hidden = true;
  els.status.textContent = "";
  els.loading.hidden = false;

  setTimeout(() => {
    try {
      const src = matFromCanvas(els.inputCanvas);
      const cartoon = cartoonify(src, selectedStyle());
      cv.imshow(els.outputCanvas, cartoon);
      updateDownloadLink();
      src.delete();
      cartoon.delete();
    } catch (err) {
      showError("Processing failed: " + err.message);
    } finally {
      els.loading.hidden = true;
    }
  }, 30);
}

function updateDownloadLink() {
  els.outputCanvas.toBlob((blob) => {
    if (els.downloadLink.href.startsWith("blob:")) {
      URL.revokeObjectURL(els.downloadLink.href);
    }
    els.downloadLink.href = URL.createObjectURL(blob);
    els.downloadLink.hidden = false;
    els.status.textContent = "Done.";
  }, "image/png");
}

els.dropZone.addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") els.fileInput.click();
});
els.dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.browseBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => loadFile(els.fileInput.files[0]));
els.sampleBtn.addEventListener("click", () => {
  clearError();
  loadImage(SAMPLE_URL);
});

["dragover", "dragenter"].forEach((event) =>
  els.dropZone.addEventListener(event, (e) => {
    e.preventDefault();
    els.dropZone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((event) =>
  els.dropZone.addEventListener(event, (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("dragover");
  })
);

els.dropZone.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
document.querySelectorAll('input[name="style"]').forEach((radio) =>
  radio.addEventListener("change", processCurrent)
);

const THEME_KEY = "cartoonify-theme";
const themeBtn = document.getElementById("theme-btn");
const themePopup = document.getElementById("theme-popup");
let savedTheme = localStorage.getItem(THEME_KEY) || "neon";

function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  themePopup.querySelectorAll("button[data-theme]").forEach((b) =>
    b.classList.toggle("active", b.dataset.theme === name)
  );
}

themeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const open = themePopup.classList.toggle("open");
  themeBtn.setAttribute("aria-expanded", String(open));
});

document.addEventListener("click", (e) => {
  if (!themePopup.contains(e.target) && e.target !== themeBtn) {
    themePopup.classList.remove("open");
    themeBtn.setAttribute("aria-expanded", "false");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    themePopup.classList.remove("open");
    themeBtn.setAttribute("aria-expanded", "false");
  }
});

themePopup.querySelectorAll("button[data-theme]").forEach((btn) => {
  btn.addEventListener("mouseenter", () => applyTheme(btn.dataset.theme));
  btn.addEventListener("mouseleave", () => applyTheme(savedTheme));
  btn.addEventListener("click", () => {
    savedTheme = btn.dataset.theme;
    localStorage.setItem(THEME_KEY, savedTheme);
    applyTheme(savedTheme);
    themePopup.classList.remove("open");
    themeBtn.setAttribute("aria-expanded", "false");
  });
});

applyTheme(savedTheme);

const bgCanvas = document.getElementById("bg-canvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let bgNodes = [];
let bgAnimId = null;

function plexusColor() {
  return (
    getComputedStyle(document.documentElement)
      .getPropertyValue("--plexus-rgb")
      .trim() || "165, 215, 255"
  );
}

function resizeBg() {
  bgCanvas.width = window.innerWidth;
  bgCanvas.height = window.innerHeight;
  const target = Math.min(
    110,
    Math.max(40, Math.round((bgCanvas.width * bgCanvas.height) / 22000))
  );
  while (bgNodes.length < target) {
    bgNodes.push({
      x: Math.random() * bgCanvas.width,
      y: Math.random() * bgCanvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: 1.2 + Math.random() * 1.6,
    });
  }
  bgNodes.length = Math.min(bgNodes.length, target);
}

function drawBg() {
  const ctx = bgCanvas.getContext("2d");
  const rgb = plexusColor();
  const LINK = 130;
  ctx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);

  for (const n of bgNodes) {
    n.x += n.vx;
    n.y += n.vy;
    if (n.x < -20) n.x = bgCanvas.width + 20;
    if (n.x > bgCanvas.width + 20) n.x = -20;
    if (n.y < -20) n.y = bgCanvas.height + 20;
    if (n.y > bgCanvas.height + 20) n.y = -20;
  }

  ctx.lineWidth = 1;
  for (let i = 0; i < bgNodes.length; i++) {
    for (let j = i + 1; j < bgNodes.length; j++) {
      const dx = bgNodes[i].x - bgNodes[j].x;
      const dy = bgNodes[i].y - bgNodes[j].y;
      const d2 = dx * dx + dy * dy;
      if (d2 < LINK * LINK) {
        const a = (1 - Math.sqrt(d2) / LINK) * 0.5;
        ctx.strokeStyle = `rgba(${rgb}, ${a.toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(bgNodes[i].x, bgNodes[i].y);
        ctx.lineTo(bgNodes[j].x, bgNodes[j].y);
        ctx.stroke();
      }
    }
  }

  ctx.shadowBlur = 6;
  for (const n of bgNodes) {
    ctx.fillStyle = `rgba(${rgb}, 0.9)`;
    ctx.shadowColor = `rgba(${rgb}, 0.9)`;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function startBg() {
  resizeBg();
  if (reduceMotion) {
    drawBg();
    return;
  }
  cancelAnimationFrame(bgAnimId);
  const loop = () => {
    if (!document.hidden) drawBg();
    bgAnimId = requestAnimationFrame(loop);
  };
  loop();
}

window.addEventListener("resize", () => {
  resizeBg();
  if (reduceMotion) drawBg();
});

startBg();
