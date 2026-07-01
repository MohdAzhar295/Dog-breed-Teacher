const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const cameraVideo = document.getElementById("cameraVideo");
const captureCanvas = document.getElementById("captureCanvas");
const errorMessage = document.getElementById("errorMessage");
const loadingState = document.getElementById("loadingState");
const emptyState = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");
const dropZone = document.getElementById("dropZone");
const breedCount = document.getElementById("breedCount");
const apiStatus = document.getElementById("apiStatus");

let cameraStream = null;
let selectedImageBlob = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTag(tag) {
  return escapeHtml(String(tag).replaceAll("_", " "));
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function setPreviewFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  previewImage.src = url;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  cameraVideo.hidden = true;
  selectedImageBlob = blob;
  analyzeBtn.disabled = false;
  showError("");
}

function renderDetailRow(label, value) {
  if (!value || value === "N/A") {
    return "";
  }

  return `
    <div class="detail-row">
      <div class="detail-label">${escapeHtml(label)}</div>
      <div class="detail-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderDetailGrid(rows) {
  const content = rows.filter(Boolean).join("");
  if (!content) {
    return '<p class="notes-block">Not available.</p>';
  }
  return `<div class="detail-grid">${content}</div>`;
}

function renderStatCard(label, value) {
  return `
    <div class="stat-card">
      <span class="stat-label">${escapeHtml(label)}</span>
      <span class="stat-value">${escapeHtml(value || "N/A")}</span>
    </div>
  `;
}

function renderSection(title, icon, body, extraClass = "") {
  return `
    <article class="info-card ${extraClass}">
      <div class="card-head">
        <div class="card-icon" aria-hidden="true">${icon}</div>
        <h3>${escapeHtml(title)}</h3>
      </div>
      ${body}
    </article>
  `;
}

function renderProfile(profile) {
  if (!profile) {
    return '<p class="notes-block">No profile available for this prediction.</p>';
  }

  const characteristics = profile.characteristics || {};
  const environment = profile.suitable_environment || {};
  const food = profile.food_priority || {};
  const origin = profile.origin || {};
  const originCountry = origin.origin_country || origin.api_origin || "N/A";
  const foundIn = origin.commonly_found_in || [];
  const aliases = profile.aliases || [];
  const colors = (characteristics.colors || []).join(", ");
  const sources = profile.sources || [];

  const originSection = renderSection(
    "Origin & Where Found",
    "⌖",
    `
      ${renderDetailGrid([
        renderDetailRow("Country of origin", originCountry),
        renderDetailRow("Region", origin.region),
        renderDetailRow("Originally bred for", origin.historical_role || profile.bred_for),
      ])}
      ${
        foundIn.length
          ? `<ul class="pill-list">${foundIn.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
          : '<p class="notes-block">Regional presence data not available.</p>'
      }
    `,
    "origin-card"
  );

  const characteristicsSection = renderSection(
    "Physical Characteristics",
    "◆",
    `
      <div class="stat-grid">
        ${renderStatCard("Size", characteristics.size)}
        ${renderStatCard("Weight", characteristics.weight_range || profile.weight)}
        ${renderStatCard("Height", characteristics.height_range || profile.height)}
      </div>
      ${renderDetailGrid([
        renderDetailRow("Coat", characteristics.coat),
        renderDetailRow("Colors", colors),
        renderDetailRow("Life span", characteristics.life_span || profile.life_span),
        renderDetailRow("Breed group", profile.breed_group),
      ])}
    `
  );

  const temperamentSection = renderSection(
    "Temperament & Behavior",
    "☺",
    `
      <ul class="tag-list">
        ${(profile.temperament_tags || []).map((tag) => `<li>${formatTag(tag)}</li>`).join("")}
      </ul>
      <p class="notes-block">${escapeHtml(profile.behavior_notes || profile.temperament_summary || "No behavior notes available.")}</p>
      ${
        aliases.length
          ? `<p class="notes-block"><strong>Also known as:</strong> ${escapeHtml(aliases.join(", "))}</p>`
          : ""
      }
    `
  );

  const environmentSection = renderSection(
    "Living Environment",
    "⌂",
    renderDetailGrid([
      renderDetailRow("Housing", environment.housing),
      renderDetailRow("Climate", environment.climate),
      renderDetailRow("Exercise needs", environment.exercise_needs || profile.exercise),
      renderDetailRow("Family fit", environment.family_fit),
    ])
  );

  const nutritionSection = renderSection(
    "Nutrition & Diet",
    "✦",
    renderDetailGrid([
      renderDetailRow("Diet focus", food.diet_focus),
      renderDetailRow("Guidance", food.notes),
    ])
  );

  const careSection = renderSection(
    "Care & Maintenance",
    "✓",
    renderDetailGrid([
      renderDetailRow("Grooming", profile.grooming),
      renderDetailRow("Exercise", profile.exercise),
      renderDetailRow("Bred for", profile.bred_for),
    ])
  );

  const sourcesSection =
    sources.length > 0
      ? `
        <div class="result-footer">
          <p class="disclaimer"><strong>References</strong></p>
          <ul class="source-list">
            ${sources.map((source) => `<li><a href="${escapeHtml(source)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source)}</a></li>`).join("")}
          </ul>
        </div>
      `
      : "";

  return `
    <div class="profile-sections">
      ${originSection}
      <div class="section-grid-two">
        ${characteristicsSection}
        ${temperamentSection}
      </div>
      <div class="section-grid-two">
        ${environmentSection}
        ${nutritionSection}
      </div>
      ${careSection}
      ${sourcesSection}
    </div>
  `;
}

function renderTopMatches(topK) {
  return `
    <article class="match-card">
      <div class="card-head">
        <div class="card-icon" aria-hidden="true">#</div>
        <h3>Top breed matches</h3>
      </div>
      <div class="match-list">
        ${topK
          .map((item, index) => {
            const percent = Math.round(item.confidence * 100);
            return `
              <div class="match-item">
                <div class="match-row">
                  <span><span class="match-rank">#${index + 1}</span> ${escapeHtml(item.display_name || item.breed_id)}</span>
                  <strong>${percent}%</strong>
                </div>
                <div class="match-bar" aria-hidden="true"><span style="width: ${percent}%"></span></div>
              </div>
            `;
          })
          .join("")}
      </div>
    </article>
  `;
}

function renderResults(payload) {
  const prediction = payload.prediction;
  const confidencePercent = Math.round(prediction.confidence * 100);
  const badgeClass = prediction.is_confident ? "badge" : "badge warning";
  const badgeText = prediction.is_confident ? "Confident match" : "Low confidence";

  resultsContent.innerHTML = `
    <div class="result-hero">
      <div>
        <span class="${badgeClass}">${badgeText}</span>
        ${payload.api_enrichment_enabled ? '<span class="badge live">Live enrichment active</span>' : ""}
        <h3>${escapeHtml(prediction.display_name)}</h3>
        <p class="result-meta">Predicted breed: ${escapeHtml(prediction.predicted_breed)}</p>
      </div>
      <div class="confidence-ring" style="--confidence: ${confidencePercent}" aria-label="Confidence ${confidencePercent} percent">
        <span>${confidencePercent}%</span>
      </div>
    </div>

    ${renderTopMatches(prediction.top_k)}
    ${renderProfile(payload.profile)}

    <div class="result-footer">
      <p class="disclaimer">${escapeHtml(payload.disclaimer)}</p>
      ${
        payload.api_enrichment_enabled
          ? "<p class='disclaimer'>Additional breed details are enriched live through TheDogAPI.</p>"
          : ""
      }
    </div>
  `;

  resultsContent.hidden = false;
  emptyState.hidden = true;
}

async function loadAppMeta() {
  try {
    const [breedsResponse, healthResponse] = await Promise.all([
      fetch("/api/breeds"),
      fetch("/api/health"),
    ]);

    if (breedsResponse.ok) {
      const breedsData = await breedsResponse.json();
      breedCount.textContent = String(breedsData.breeds?.length || 30);
    }

    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      apiStatus.textContent = healthData.api_enrichment_enabled
        ? "TheDogAPI connected"
        : "Offline profiles only";
    } else {
      apiStatus.textContent = "Profile data ready";
    }
  } catch (error) {
    apiStatus.textContent = "Profile data ready";
  }
}

async function analyzeImage() {
  if (!selectedImageBlob) {
    showError("Select or capture an image first.");
    return;
  }

  loadingState.hidden = false;
  resultsContent.hidden = true;
  emptyState.hidden = true;
  analyzeBtn.disabled = true;
  showError("");

  try {
    const formData = new FormData();
    formData.append("file", selectedImageBlob, "capture.jpg");

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    renderResults(data);
  } catch (error) {
    showError(error.message);
    emptyState.hidden = false;
  } finally {
    loadingState.hidden = true;
    analyzeBtn.disabled = false;
  }
}

function handleDroppedFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please drop a valid image file.");
    return;
  }
  setPreviewFromBlob(file);
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    return;
  }
  setPreviewFromBlob(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragover");
  handleDroppedFile(event.dataTransfer?.files?.[0]);
});

startCameraBtn.addEventListener("click", async () => {
  try {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });

    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    previewImage.hidden = true;
    previewPlaceholder.hidden = true;
    captureBtn.hidden = false;
    showError("");
  } catch (error) {
    showError("Unable to access camera. Check browser permissions.");
  }
});

captureBtn.addEventListener("click", () => {
  const width = cameraVideo.videoWidth;
  const height = cameraVideo.videoHeight;
  captureCanvas.width = width;
  captureCanvas.height = height;

  const context = captureCanvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, width, height);

  captureCanvas.toBlob((blob) => {
    if (blob) {
      setPreviewFromBlob(blob);
      captureBtn.hidden = true;
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
      }
    }
  }, "image/jpeg", 0.92);
});

analyzeBtn.addEventListener("click", analyzeImage);
loadAppMeta();
