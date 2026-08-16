// ═══════════════════════════════════════════════
// Chest X-ray Classifier -- runs 100% in the browser
// Uses ONNX Runtime Web to run the neural network directly
// on the visitor's device. No backend server needed at all.
// ═══════════════════════════════════════════════

const MODEL_URL = "https://huggingface.co/bach88/chest_xray_2.0/resolve/main/chest_xray_model_single.onnx";
const CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]; // must match training order!
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const fileInput = document.getElementById("fileInput");
const uploadText = document.getElementById("uploadText");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const originalImage = document.getElementById("originalImage");
const verdict = document.getElementById("verdict");
const confidencesDiv = document.getElementById("confidences");

let session = null;
let selectedFile = null;

// ── Load the ONNX model once, when the page loads ──
async function loadModel() {
  statusEl.textContent = "Loading model...";
  try {
    session = await ort.InferenceSession.create(MODEL_URL);
    statusEl.textContent = "Model ready.";
  } catch (err) {
    statusEl.textContent = "Failed to load model: " + err.message;
    console.error(err);
  }
}
loadModel();

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    selectedFile = fileInput.files[0];
    uploadText.textContent = selectedFile.name;
    predictBtn.disabled = false;
  }
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile || !session) return;

  results.classList.add("hidden");
  statusEl.textContent = "Analyzing...";
  predictBtn.disabled = true;

  try {
    const inputTensor = await preprocessImage(selectedFile);
    const feeds = { input: inputTensor };
    const output = await session.run(feeds);
    const logits = output.output.data; // raw scores, one per class

    const probs = softmax(logits);
    displayResults(probs, selectedFile);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
    console.error(err);
  } finally {
    predictBtn.disabled = false;
  }
});

// ── Preprocessing: resize to 224x224, normalize like training did ──
async function preprocessImage(file) {
  const img = await loadImage(file);

  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 224, 224);

  const imageData = ctx.getImageData(0, 0, 224, 224);
  const { data } = imageData; // RGBA, 0-255

  // Convert to CHW format (channels, height, width) as float32,
  // normalized using ImageNet mean/std -- same as the Python training pipeline.
  const floatData = new Float32Array(3 * 224 * 224);
  const numPixels = 224 * 224;

  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];               // R channel
    floatData[numPixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];   // G channel
    floatData[2 * numPixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]; // B channel
  }

  return new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits).map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function displayResults(probs, originalFile) {
  originalImage.src = URL.createObjectURL(originalFile);

  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  const predictedClass = CLASS_NAMES[maxIdx];
  const isAbnormal = predictedClass !== "Normal";

  verdict.textContent = isAbnormal
    ? `⚠️ Abnormality detected: ${predictedClass}`
    : "✅ No abnormality detected (Normal)";
  verdict.className = "verdict " + (isAbnormal ? "abnormal" : "normal");

  confidencesDiv.innerHTML = "";
  const sorted = CLASS_NAMES
    .map((name, i) => ({ name, score: probs[i] }))
    .sort((a, b) => b.score - a.score);

  for (const { name, score } of sorted) {
    const pct = (score * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "confidence-row";
    row.innerHTML = `
      <div class="confidence-label"><span>${name}</span><span>${pct}%</span></div>
      <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:${pct}%"></div></div>
    `;
    confidencesDiv.appendChild(row);
  }

  results.classList.remove("hidden");
}
