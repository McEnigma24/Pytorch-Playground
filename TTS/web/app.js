import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";
import { TextLayer } from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.mjs";

const PDF_SCALE = 1.4;
const ANCHOR_K = 3;
const USER_SCROLL_COOLDOWN_MS = 3000;

const appEl = document.getElementById("app");
const btnHome = document.getElementById("btn-home");
const btnUpload = document.getElementById("btn-upload");

let listPollTimer = null;
let readerState = null;

btnHome.addEventListener("click", () => navigateTo({ view: "list" }));
btnUpload.addEventListener("click", () => navigateTo({ view: "upload" }));

window.addEventListener("hashchange", () => renderFromHash());
renderFromHash();

function renderFromHash() {
  const hash = location.hash.replace(/^#/, "");
  if (!hash) {
    navigateTo({ view: "list" }, false);
    return;
  }
  if (hash === "upload") {
    showUpload();
    return;
  }
  if (hash.startsWith("read/")) {
    const id = hash.slice("read/".length);
    showReader(id);
    return;
  }
  showList();
}

function navigateTo({ view, id = null }, pushHash = true) {
  if (view === "list") {
    if (pushHash) location.hash = "";
    showList();
  } else if (view === "upload") {
    if (pushHash) location.hash = "upload";
    showUpload();
  } else if (view === "reader" && id) {
    if (pushHash) location.hash = `read/${id}`;
    showReader(id);
  }
}

function clearPoll() {
  if (listPollTimer) {
    clearInterval(listPollTimer);
    listPollTimer = null;
  }
  stopReaderPlayback();
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function tokenKey(raw) {
  const nfkc = raw.normalize("NFKC");
  const lower = nfkc.toLowerCase();
  return lower.normalize("NFD").replace(/\p{M}/gu, "");
}

const WORD_RE = /[\p{L}\p{N}]+(?:-[\p{L}\p{N}]*)?/gu;

/** @returns {{ text: string, key: string, start: number, end: number }[]} */
function tokenizeString(text) {
  const out = [];
  let match;
  WORD_RE.lastIndex = 0;
  while ((match = WORD_RE.exec(text)) !== null) {
    let t = match[0];
    let key = tokenKey(t.replace(/-$/, ""));
    out.push({
      text: t,
      key,
      start: match.index,
      end: match.index + t.length,
    });
  }
  return out;
}

function parseVtt(vttText) {
  const lines = vttText.replace(/\r/g, "").split("\n");
  /** @type {{ id: number, start: number, end: number, text: string }[]} */
  const cues = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line || line === "WEBVTT" || /^\d+$/.test(line)) {
      i += 1;
      continue;
    }
    const timeMatch = line.match(
      /^(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})$/
    );
    if (timeMatch) {
      const start = vttTimeToSec(timeMatch[1]);
      const end = vttTimeToSec(timeMatch[2]);
      i += 1;
      const textParts = [];
      while (i < lines.length && lines[i].trim() !== "") {
        textParts.push(lines[i].trim());
        i += 1;
      }
      const text = textParts.join(" ");
      if (text) {
        cues.push({ id: cues.length + 1, start, end, text });
      }
    }
    i += 1;
  }
  return cues;
}

function vttTimeToSec(ts) {
  const [h, m, rest] = ts.split(":");
  const [s, ms] = rest.split(".");
  return (
    Number(h) * 3600 +
    Number(m) * 60 +
    Number(s) +
    Number(ms) / 1000
  );
}

/**
 * @param {{ key: string }[]} pdfWords
 * @param {{ key: string, cueIndex: number }[]} vttWords
 */
function alignWords(pdfWords, vttWords) {
  /** @type {(null | { pdfFrom: number, pdfTo: number })[]} */
  const cueMap = [];
  let p = 0;

  for (let i = 0; i < vttWords.length; i += 1) {
    const cueIndex = vttWords[i].cueIndex;
    while (cueMap.length <= cueIndex) cueMap.push(null);

    const anchorLen = Math.min(ANCHOR_K, vttWords.length - i);
    let matchedAt = -1;

    for (let j = p; j < pdfWords.length; j += 1) {
      let ok = true;
      for (let a = 0; a < anchorLen; a += 1) {
        if (j + a >= pdfWords.length || pdfWords[j + a].key !== vttWords[i + a].key) {
          ok = false;
          break;
        }
      }
      if (ok) {
        matchedAt = j;
        break;
      }
    }

    if (matchedAt < 0) {
      for (let j = p; j < pdfWords.length; j += 1) {
        if (pdfWords[j].key === vttWords[i].key) {
          matchedAt = j;
          break;
        }
      }
    }

    if (matchedAt >= 0) {
      if (!cueMap[cueIndex]) {
        cueMap[cueIndex] = { pdfFrom: matchedAt, pdfTo: matchedAt };
      } else {
        cueMap[cueIndex].pdfTo = matchedAt;
      }
      p = matchedAt + 1;
    }
  }

  let mapped = 0;
  for (const m of cueMap) if (m) mapped += 1;
  const coverage = cueMap.length ? (mapped / cueMap.length) * 100 : 0;
  return { cueMap, coverage, mappedCues: mapped, totalCues: cueMap.length };
}

/** @param {HTMLElement} pageEl @param {number} pageNum */
function buildPdfWords(pageEl, pageNum) {
  const layer = pageEl.querySelector(".textLayer");
  if (!layer) return [];
  const spans = layer.querySelectorAll("span");
  /** @type {{ key: string, span: HTMLElement, start: number, end: number, page: number }[]} */
  const words = [];
  for (const span of spans) {
    const text = span.textContent || "";
    for (const tok of tokenizeString(text)) {
      words.push({
        key: tok.key,
        span,
        start: tok.start,
        end: tok.end,
        page: pageNum,
      });
    }
  }
  return words;
}

async function fetchMaterials() {
  const res = await fetch("/api/materials");
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function showList() {
  clearPoll();
  appEl.innerHTML = `<p class="empty">Ładowanie listy…</p>`;
  loadList();
  listPollTimer = setInterval(loadList, 4000);
}

async function loadList() {
  try {
    const items = await fetchMaterials();
    if (!location.hash || location.hash === "#") {
      /* still on list */
    } else if (!location.hash.startsWith("#read/") && location.hash !== "#upload") {
      return;
    }
    if (location.hash.startsWith("#read/") || location.hash === "#upload") return;

    if (!items.length) {
      appEl.innerHTML = `
        <div class="empty">
          <p>Brak materiałów.</p>
          <p><button type="button" class="btn" id="inline-upload">Dodaj pierwszy PDF</button></p>
        </div>`;
      document.getElementById("inline-upload")?.addEventListener("click", () =>
        navigateTo({ view: "upload" })
      );
      return;
    }

    appEl.innerHTML = `<div class="grid" id="material-grid"></div>`;
    const grid = document.getElementById("material-grid");
    for (const m of items) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "card";
      const progress = m.progress ? ` · postęp ${m.progress}` : "";
      const dur =
        m.duration_sec != null ? ` · ${m.duration_sec.toFixed(1)} s` : "";
      card.innerHTML = `
        <h2>${escapeHtml(m.title)}</h2>
        <div class="meta">
          <span class="status-${m.status}">${escapeHtml(m.status)}</span>${progress}${dur}
        </div>`;
      if (m.status === "error" && m.error) {
        card.innerHTML += `<p class="meta">${escapeHtml(m.error.split("\n")[0])}</p>`;
      }
      card.addEventListener("click", () => {
        if (m.status === "ready") navigateTo({ view: "reader", id: m.id });
        else alert(`Status: ${m.status}${m.error ? "\n" + m.error.slice(0, 200) : ""}`);
      });
      grid.appendChild(card);
    }
  } catch (err) {
    appEl.innerHTML = `<p class="message">Błąd listy: ${escapeHtml(String(err))}</p>`;
  }
}

function showUpload() {
  clearPoll();
  appEl.innerHTML = `
    <form class="form-panel" id="upload-form">
      <label>Tytuł (opcjonalnie)
        <input type="text" name="title" autocomplete="off" />
      </label>
      <label>Plik PDF (wymagany)
        <input type="file" name="pdf" accept="application/pdf" required />
      </label>
      <label>Tekst po filtracji (opcjonalnie — tylko usunięcia względem PDF)
        <textarea name="text" placeholder="Zostaw puste, aby wyciągnąć tekst z PDF na serwerze"></textarea>
      </label>
      <label>Język
        <select name="language">
          <option value="English" selected>English</option>
          <option value="Polish">Polish</option>
        </select>
      </label>
      <label>Speaker
        <select name="speaker">
          <option value="ryan" selected>ryan</option>
          <option value="aiden">aiden</option>
          <option value="dylan">dylan</option>
          <option value="eric">eric</option>
          <option value="serena">serena</option>
          <option value="sohee">sohee</option>
          <option value="vivian">vivian</option>
          <option value="uncle_fu">uncle_fu</option>
          <option value="ono_anna">ono_anna</option>
        </select>
      </label>
      <button type="submit" class="btn">Wyślij do przetwarzania</button>
      <p class="meta" id="upload-status"></p>
    </form>`;

  document.getElementById("upload-form").addEventListener("submit", onUploadSubmit);
}

async function onUploadSubmit(ev) {
  ev.preventDefault();
  const form = ev.target;
  const statusEl = document.getElementById("upload-status");
  const fd = new FormData(form);
  const pdf = fd.get("pdf");
  if (!(pdf instanceof File) || !pdf.size) {
    statusEl.textContent = "Wybierz plik PDF.";
    return;
  }

  const submitBtn = form.querySelector('[type="submit"]');
  submitBtn.disabled = true;
  statusEl.textContent = "Wysyłanie…";

  try {
    const res = await fetch("/api/materials", { method: "POST", body: fd });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || res.statusText);
    statusEl.textContent = `W kolejce: ${body.id}`;
    navigateTo({ view: "list" });
  } catch (err) {
    statusEl.textContent = `Błąd: ${err.message}`;
    submitBtn.disabled = false;
  }
}

function stopReaderPlayback() {
  if (readerState?.rafId) cancelAnimationFrame(readerState.rafId);
  readerState = null;
  if (CSS.highlights) CSS.highlights.delete("tts-word");
}

async function showReader(materialId) {
  clearPoll();
  stopReaderPlayback();
  appEl.innerHTML = `
    <p class="empty">Ładowanie czytnika…</p>
    <div id="reader-root" hidden></div>`;

  const metaList = await fetchMaterials();
  const meta = metaList.find((m) => m.id === materialId);
  const title = meta?.title || materialId;

  appEl.innerHTML = `
    <div id="reader-root">
      <div class="reader-toolbar">
        <strong>${escapeHtml(title)}</strong>
        <audio id="reader-audio" controls preload="metadata"></audio>
        <div class="reader-stats">
          <span id="coverage-label">Pokrycie dopasowania: —</span>
          <span id="time-label"></span>
        </div>
      </div>
      <div class="pdf-pages" id="pdf-pages"></div>
    </div>`;

  const audio = document.getElementById("reader-audio");
  audio.src = `/api/materials/${encodeURIComponent(materialId)}/wav`;

  const pdfUrl = `/api/materials/${encodeURIComponent(materialId)}/pdf`;
  const vttUrl = `/api/materials/${encodeURIComponent(materialId)}/vtt`;

  const [vttRes, pdf] = await Promise.all([
    fetch(vttUrl),
    pdfjsLib.getDocument(pdfUrl).promise,
  ]);
  const vttText = await vttRes.text();
  const cues = parseVtt(vttText);

  const pagesEl = document.getElementById("pdf-pages");
  const pdfWords = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum += 1) {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale: PDF_SCALE });

    const pageWrap = document.createElement("div");
    pageWrap.className = "pdf-page";
    pageWrap.dataset.page = String(pageNum);

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "textLayer";

    pageWrap.append(canvas, textLayerDiv);
    pagesEl.appendChild(pageWrap);

    await page.render({ canvasContext: ctx, viewport }).promise;
    const textContent = await page.getTextContent();
    const textLayer = new TextLayer({
      textContentSource: textContent,
      container: textLayerDiv,
      viewport,
    });
    await textLayer.render();

    pdfWords.push(...buildPdfWords(pageWrap, pageNum));
  }

  const vttWords = [];
  cues.forEach((cue, cueIndex) => {
    for (const tok of tokenizeString(cue.text)) {
      vttWords.push({ key: tok.key, cueIndex });
    }
  });

  const { cueMap, coverage, mappedCues, totalCues } = alignWords(pdfWords, vttWords);
  document.getElementById("coverage-label").textContent =
    `Pokrycie dopasowania: ${coverage.toFixed(1)}% (${mappedCues}/${totalCues} cue)`;

  readerState = {
    audio,
    cues,
    cueMap,
    pdfWords,
    lastUserScroll: 0,
    rafId: null,
    lastCueIndex: -1,
  };

  window.addEventListener(
    "scroll",
    () => {
      if (readerState) readerState.lastUserScroll = performance.now();
    },
    { passive: true }
  );

  pagesEl.addEventListener("click", (ev) => {
    const span = ev.target.closest(".textLayer span");
    if (!span || !readerState) return;
    const idx = pdfWords.findIndex((w) => w.span === span);
    if (idx < 0) return;
    const cueIdx = findCueForPdfIndex(readerState.cueMap, idx);
    if (cueIdx >= 0 && readerState.cues[cueIdx]) {
      readerState.audio.currentTime = readerState.cues[cueIdx].start;
    }
  });

  audio.addEventListener("play", () => tickHighlight());
  audio.addEventListener("pause", () => {
    if (readerState?.rafId) {
      cancelAnimationFrame(readerState.rafId);
      readerState.rafId = null;
    }
  });
  audio.addEventListener("ended", () => {
    if (CSS.highlights) CSS.highlights.delete("tts-word");
  });
}

function findCueForPdfIndex(cueMap, pdfIdx) {
  for (let i = 0; i < cueMap.length; i += 1) {
    const m = cueMap[i];
    if (!m) continue;
    if (pdfIdx >= m.pdfFrom && pdfIdx <= m.pdfTo) return i;
  }
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < cueMap.length; i += 1) {
    const m = cueMap[i];
    if (!m) continue;
    const dist = Math.min(Math.abs(pdfIdx - m.pdfFrom), Math.abs(pdfIdx - m.pdfTo));
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

function findActiveCueIndex(cues, t) {
  let lo = 0;
  let hi = cues.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (ans < 0) return -1;
  const c = cues[ans];
  if (t > c.end + 0.05) {
    for (let i = ans + 1; i < cues.length; i += 1) {
      if (t >= cues[i].start && t <= cues[i].end + 0.05) return i;
    }
    return -1;
  }
  return ans;
}

function rangesForCue(cueIndex, cueMap, pdfWords) {
  const m = cueMap[cueIndex];
  if (!m) return [];
  const ranges = [];
  for (let i = m.pdfFrom; i <= m.pdfTo; i += 1) {
    const w = pdfWords[i];
    if (!w) continue;
    const range = document.createRange();
    range.setStart(w.span, w.start);
    range.setEnd(w.span, w.end);
    ranges.push(range);
  }
  return ranges;
}

function tickHighlight() {
  if (!readerState) return;
  const { audio, cues, cueMap, pdfWords } = readerState;
  const t = audio.currentTime;
  const cueIdx = findActiveCueIndex(cues, t);

  document.getElementById("time-label").textContent =
    `${t.toFixed(2)} s` + (cueIdx >= 0 ? ` · cue ${cueIdx + 1}` : "");

  if (cueIdx !== readerState.lastCueIndex) {
    readerState.lastCueIndex = cueIdx;
    if (cueIdx >= 0 && CSS.highlights) {
      const ranges = rangesForCue(cueIdx, cueMap, pdfWords);
      if (ranges.length) {
        CSS.highlights.set("tts-word", new Highlight(...ranges));
        maybeScrollToCue(cueIdx, cueMap, pdfWords);
      } else {
        CSS.highlights.delete("tts-word");
      }
    } else if (CSS.highlights) {
      CSS.highlights.delete("tts-word");
    }
  }

  if (!audio.paused) {
    readerState.rafId = requestAnimationFrame(tickHighlight);
  }
}

function maybeScrollToCue(cueIdx, cueMap, pdfWords) {
  if (!readerState) return;
  if (performance.now() - readerState.lastUserScroll < USER_SCROLL_COOLDOWN_MS) return;
  const m = cueMap[cueIdx];
  if (!m) return;
  const w = pdfWords[m.pdfFrom];
  if (!w) return;
  w.span.scrollIntoView({ block: "center", behavior: "smooth" });
}
