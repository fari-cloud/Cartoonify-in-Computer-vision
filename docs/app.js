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
  const flat = new cv.Mat();
  cv.bilateralFilter(rgb, flat, 9, 300, 300);

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

  const flat = new cv.Mat(rows, 1, cv.CV_8UC3);
  for (let i = 0; i < rows; i++) {
    const c = labels.data32S[i] * 3;
    flat.data[i * 3] = centers.data[c];
    flat.data[i * 3 + 1] = centers.data[c + 1];
    flat.data[i * 3 + 2] = centers.data[c + 2];
  }

  const result = new cv.Mat(rgb.rows, rgb.cols, cv.CV_8UC3);
  result.data.set(flat.data);

  samples.delete(); labels.delete(); centers.delete(); flat.delete();
  return result;
}

function comicCartoon(src) {
  const gray = toGray(src);
  const smooth = new cv.Mat();
  cv.medianBlur(gray, smooth, 5);

  const edges = new cv.Mat();
  cv.adaptiveThreshold(
    smooth, edges, 255,
    cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9
  );

  const rgb = toRGB(src);
  const flat = posterize(rgb, 8);

  const cartoonRGB = new cv.Mat();
  cv.bitwise_and(flat, flat, cartoonRGB, edges);

  const cartoon = new cv.Mat();
  cv.cvtColor(cartoonRGB, cartoon, cv.COLOR_RGB2RGBA);

  gray.delete(); smooth.delete(); edges.delete(); rgb.delete();
  flat.delete(); cartoonRGB.delete();
  return cartoon;
}

function hdrCartoon(src) {
  const rgb = toRGB(src);
  let result = rgb;
  for (let i = 0; i < 3; i++) {
    const next = new cv.Mat();
    cv.bilateralFilter(result, next, 9, 200, 200);
    if (i > 0) result.delete();
    result = next;
  }
  const out = new cv.Mat();
  cv.cvtColor(result, out, cv.COLOR_RGB2RGBA);
  result.delete();
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
