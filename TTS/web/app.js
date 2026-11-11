import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";
import { TextLayer } from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.mjs";

const PDF_SCALE = 1.4;
const ANCHOR_K = 3;
const USER_SCROLL_COOLDOWN_MS = 3000;
const CHUNK_CHARS = 500;

const appEl = document.getElementById("app");
const btnHome = document.getElementById("btn-home");
const btnNew = document.getElementById("btn-new");

let listPollTimer = null;
let etaTickTimer = null;
let lastListItems = [];
let listActionInFlight = false;
let readerState = null;
let listLoadGeneration = 0;
/** @type {"list"|"new"|"edit"|"read"} */
let activeView = "list";

btnHome.addEventListener("click", () => navigateTo({ view: "list" }));
btnNew.addEventListener("click", () => navigateTo({ view: "new" }));

window.addEventListener("hashchange", () => renderFromHash());

/** Po pełnym przeładowaniu strony zawsze startujemy od listy (ignoruj stary hash). */
function bootstrap() {
  activeView = "list";
  const path = `${location.pathname}${location.search}#list`;
  if (location.hash.replace(/^#/, "") !== "list") {
    history.replaceState(null, "", path);
  }
  showList();
}

bootstrap();

const STATUS_HINT = {
  uploaded: "Wgrano PDF — wybierz strony i filtry, potem wyciągnij tekst (lub otwórz Edytuj).",
  draft: "Tekst wyciągnięty — usuń zbędne fragmenty i kliknij Generuj.",
  queued: "W kolejce do syntezy — możesz zatrzymać.",
  processing: "Trwa synteza i alignment — zajmie to chwilę (GPU).",
  paused: "Wstrzymano — wznów albo edytuj tekst.",
  ready: "Gotowe — otwórz czytnik.",
  error: "Błąd przetwarzania — wznów lub edytuj.",
};

function renderFromHash() {
  const hash = location.hash.replace(/^#/, "");
  if (!hash || hash === "list") {
    activeView = "list";
    showList();
    return;
  }
  if (hash === "new") {
    activeView = "new";
    showNew();
    return;
  }
  if (hash.startsWith("edit/")) {
    activeView = "edit";
    showEdit(hash.slice("edit/".length));
    return;
  }
  if (hash.startsWith("read/")) {
    activeView = "read";
    showReader(hash.slice("read/".length));
    return;
  }
  activeView = "list";
  location.replace("#list");
}

function hashForView(view, id) {
  if (view === "list") return "list";
  if (view === "new") return "new";
  if (view === "edit" && id) return `edit/${id}`;
  if (view === "reader" && id) return `read/${id}`;
  return "list";
}

function navigateTo({ view, id = null }, pushHash = true) {
  const target = hashForView(view, id);
  const current = location.hash.replace(/^#/, "");
  if (pushHash && current !== target) {
    location.hash = target;
    return;
  }
  activeView = view;
  if (view === "list") showList();
  else if (view === "new") showNew();
  else if (view === "edit" && id) showEdit(id);
  else if (view === "reader" && id) showReader(id);
}

function clearEtaTick() {
  if (etaTickTimer) {
    clearInterval(etaTickTimer);
    etaTickTimer = null;
  }
}

function clearPoll() {
  if (listPollTimer) {
    clearInterval(listPollTimer);
    listPollTimer = null;
  }
  clearEtaTick();
  stopReaderPlayback();
}

function formatDuration(sec) {
  if (sec == null || Number.isNaN(sec)) return "—";
  const totalMin = Math.max(0, Math.ceil(sec / 60));
  const d = Math.floor(totalMin / (24 * 60));
  const h = Math.floor((totalMin % (24 * 60)) / 60);
  const m = totalMin % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d)}d - ${pad(h)}h - ${pad(m)}m`;
}

function countdownRemaining(etaSec, etaAt, nowMs = Date.now()) {
  if (etaSec == null) return null;
  let rem = etaSec;
  if (etaAt) rem -= (nowMs - new Date(etaAt).getTime()) / 1000;
  return Math.max(0, rem);
}

function projectedFinishMs(etaSec, etaAt, finishAt, nowMs = Date.now()) {
  if (finishAt) {
    const skew = etaAt ? nowMs - new Date(etaAt).getTime() : 0;
    return new Date(finishAt).getTime() + skew;
  }
  const rem = countdownRemaining(etaSec, etaAt, nowMs);
  if (rem == null) return null;
  return nowMs + rem * 1000;
}

function formatFinishClock(whenMs, nowMs = Date.now()) {
  const date = new Date(whenMs);
  const now = new Date(nowMs);
  const pad = (n) => String(n).padStart(2, "0");
  const clock = `${pad(date.getHours())}h-${pad(date.getMinutes())}m`;
  if (date.toDateString() === now.toDateString()) {
    return clock;
  }
  const tomorrow = new Date(now);
  tomorrow.setDate(tomorrow.getDate() + 1);
  if (date.toDateString() === tomorrow.toDateString()) {
    return `jutro ${clock}`;
  }
  return `${pad(date.getDate())}.${pad(date.getMonth() + 1)}. ${clock}`;
}

function etaLinesForMaterial(m, nowMs = Date.now()) {
  if (!["queued", "processing"].includes(m.status)) return [];
  if (m.status === "queued") {
    return ["estymacja po pierwszym fragmencie"];
  }
  if (m.status !== "processing") return [];

  const lines = [];
  const chunkRem = countdownRemaining(m.chunk_eta_sec, m.chunk_eta_at, nowMs);
  if (chunkRem != null && chunkRem > 0) {
    const n = m.progress ? String(m.progress).split("/")[0] : "?";
    lines.push(`kawałek ${n}: ${formatDuration(chunkRem)}`);
  }

  const totalRem = countdownRemaining(m.eta_sec, m.eta_at, nowMs);
  if (totalRem != null && totalRem > 0) {
    lines.push(`całość: ${formatDuration(totalRem)}`);
  }

  if (!lines.length) {
    if (m.chunk_eta_sec == null && m.eta_sec == null) {
      return ["estymacja po pierwszym fragmencie"];
    }
    return ["kończę bieżący kawałek…"];
  }

  const chunkFinishMs = projectedFinishMs(
    m.chunk_eta_sec,
    m.chunk_eta_at,
    m.chunk_finish_at,
    nowMs
  );
  if (chunkRem != null && chunkRem > 0 && chunkFinishMs != null) {
    lines.push(`koniec kawałku: ${formatFinishClock(chunkFinishMs, nowMs)}`);
  }

  const totalFinishMs = projectedFinishMs(m.eta_sec, m.eta_at, m.eta_finish_at, nowMs);
  if (totalRem != null && totalRem > 0 && totalFinishMs != null) {
    lines.push(`koniec całości: ${formatFinishClock(totalFinishMs, nowMs)}`);
  }

  return lines;
}

function etaBlockInnerHtml(m, nowMs = Date.now()) {
  const lines = etaLinesForMaterial(m, nowMs);
  if (!lines.length) return "";
  return lines.map((ln) => `<p class="meta eta-line">${escapeHtml(ln)}</p>`).join("");
}

function progressSectionHtml(m) {
  const total = m.chars_total;
  const done = m.chars_done ?? 0;
  if (!total || !["queued", "processing", "paused"].includes(m.status)) return "";
  const pct = Math.min(100, Math.round((done / Math.max(1, total)) * 100));
  const etaBlock = etaBlockInnerHtml(m);
  return `
        <div class="progress-wrap" aria-hidden="true">
          <div class="progress-bar" style="width:${pct}%"></div>
        </div>
        ${etaBlock ? `<div class="eta-countdown" data-id="${escapeHtml(m.id)}">${etaBlock}</div>` : ""}`;
}

function refreshEtaLabels() {
  for (const el of document.querySelectorAll(".eta-countdown")) {
    const id = el.dataset.id;
    const m = lastListItems.find((x) => x.id === id);
    if (m) el.innerHTML = etaBlockInnerHtml(m);
  }
}

function startEtaTick(items) {
  clearEtaTick();
  lastListItems = items;
  if (!items.some((x) => ["queued", "processing"].includes(x.status))) return;
  etaTickTimer = setInterval(refreshEtaLabels, 1000);
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sanitizeTitlePreview(title) {
  return title
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^\w.\-]/g, "")
    .replace(/^_+|_+$/g, "") || "untitled";
}

function keptCharKey(ch) {
  const nfkc = ch.normalize("NFKC");
  if (nfkc === "'") return "'";
  if (/\p{L}/u.test(nfkc) || /\p{N}/u.test(nfkc)) {
    return nfkc.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
  }
  return "";
}

function keyChars(text) {
  let out = "";
  for (const ch of text) {
    out += keptCharKey(ch);
  }
  return out;
}

function parseVtt(vttText) {
  const lines = vttText.replace(/\r/g, "").split("\n");
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
      if (text) cues.push({ start, end, text, key: keyChars(text) });
    }
    i += 1;
  }
  return cues;
}

function vttTimeToSec(ts) {
  const [h, m, rest] = ts.split(":");
  const [s, ms] = rest.split(".");
  return Number(h) * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000;
}

function buildCharStream(pageEls) {
  const streamPos = [];
  let keyStream = "";
  for (const pageEl of pageEls) {
    const layer = pageEl.querySelector(".textLayer");
    if (!layer) continue;
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    let textNode = walker.nextNode();
    while (textNode) {
      const text = textNode.textContent || "";
      for (let off = 0; off < text.length; off += 1) {
        const k = keptCharKey(text[off]);
        if (k) {
          keyStream += k;
          streamPos.push({ node: textNode, offset: off });
        }
      }
      textNode = walker.nextNode();
    }
  }
  return { keyStream, streamPos };
}

function alignCuesToStream(cues, keyStream) {
  const cueStarts = new Array(cues.length).fill(-1);
  const cueEnds = new Array(cues.length).fill(-1);
  let p = 0;

  for (let ci = 0; ci < cues.length; ci += 1) {
    const key = cues[ci].key;
    if (!key) continue;

    const anchorLen = Math.min(ANCHOR_K, cues.length - ci);
    let anchorKeys = "";
    for (let a = 0; a < anchorLen; a += 1) {
      anchorKeys += cues[ci + a].key;
    }

    let pos = keyStream.indexOf(anchorKeys, p);
    if (pos < 0) pos = keyStream.indexOf(key, p);
    if (pos >= 0) {
      cueStarts[ci] = pos;
      cueEnds[ci] = pos + key.length - 1;
      p = pos + key.length;
    }
  }

  let mapped = cueStarts.filter((x) => x >= 0).length;
  const coverage = cues.length ? (mapped / cues.length) * 100 : 0;
  return { cueStarts, cueEnds, coverage, mappedCues: mapped, totalCues: cues.length };
}

function rangesForCue(cueIndex, cueStarts, cueEnds, streamPos) {
  const start = cueStarts[cueIndex];
  const end = cueEnds[cueIndex];
  if (start < 0 || end < start) return [];

  const ranges = [];
  let range = null;
  for (let i = start; i <= end; i += 1) {
    const pos = streamPos[i];
    if (!pos) continue;
    if (
      range &&
      range.endContainer === pos.node &&
      range.endOffset === pos.offset
    ) {
      range.setEnd(pos.node, pos.offset + 1);
    } else {
      if (range) ranges.push(range);
      range = document.createRange();
      range.setStart(pos.node, pos.offset);
      range.setEnd(pos.node, pos.offset + 1);
    }
  }
  if (range) ranges.push(range);
  return ranges;
}

function rangesForLine(cueIndex, cueStarts, cueEnds, streamPos) {
  const start = cueStarts[cueIndex];
  const end = cueEnds[cueIndex];
  if (start < 0 || end < start) return [];

  const nodes = new Set();
  for (let i = start; i <= end; i += 1) {
    const pos = streamPos[i];
    if (pos?.node) nodes.add(pos.node);
  }
  const ranges = [];
  for (const node of nodes) {
    const r = document.createRange();
    try {
      r.selectNodeContents(node);
      ranges.push(r);
    } catch {
      /* skip invalid nodes */
    }
  }
  return ranges;
}

async function renderPdfPages(pdf, pagesEl, pageFrom, pageTo, scale = PDF_SCALE) {
  const pageEls = [];
  const last = Math.min(pageTo, pdf.numPages);
  for (let pageNum = pageFrom; pageNum <= last; pageNum += 1) {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale });

    const pageWrap = document.createElement("div");
    pageWrap.className = "pdf-page";
    pageWrap.dataset.page = String(pageNum);
    pageWrap.style.setProperty("--scale-factor", String(viewport.scale));

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "textLayer";

    pageWrap.append(canvas, textLayerDiv);
    pagesEl.appendChild(pageWrap);
    pageEls.push(pageWrap);

    await page.render({ canvasContext: ctx, viewport }).promise;
    const textContent = await page.getTextContent();
    const textLayer = new TextLayer({
      textContentSource: textContent,
      container: textLayerDiv,
      viewport,
    });
    await textLayer.render();
  }
  return pageEls;
}

async function fetchMaterials() {
  const res = await fetch("/api/materials");
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function fetchMaterial(id) {
  const res = await fetch(`/api/materials/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function fetchFilters() {
  const res = await fetch("/api/filters");
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function fetchStats() {
  const res = await fetch("/api/stats");
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function formatApiError(data, fallback) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join("; ");
  return fallback;
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(formatApiError(data, res.statusText));
  return data;
}

async function apiDelete(id) {
  const res = await fetch(`/api/materials/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function showList() {
  clearPoll();
  activeView = "list";
  listActionInFlight = false;
  const generation = ++listLoadGeneration;
  appEl.innerHTML = `<p class="empty">Ładowanie listy…</p>`;
  loadList(generation);
  listPollTimer = setInterval(() => loadList(listLoadGeneration), 4000);
}

async function loadList(generation) {
  if (generation !== listLoadGeneration || activeView !== "list") return;
  if (listActionInFlight) return;

  try {
    const items = await fetchMaterials();
    if (generation !== listLoadGeneration || activeView !== "list") return;

    if (!items.length) {
      appEl.innerHTML = `
        <div class="empty">
          <p>Brak materiałów.</p>
          <p><button type="button" class="btn" id="inline-new">Dodaj pierwszy PDF</button></p>
        </div>`;
      document.getElementById("inline-new")?.addEventListener("click", () =>
        navigateTo({ view: "new" })
      );
      return;
    }

    if (generation !== listLoadGeneration || activeView !== "list") return;

    appEl.innerHTML = `<div class="grid" id="material-grid"></div>`;
    const grid = document.getElementById("material-grid");

    for (const m of items) {
      const card = document.createElement("div");
      card.className = "card";
      const progress = m.progress ? ` · ${m.progress}` : "";
      const dur = m.duration_sec != null ? ` · ${m.duration_sec.toFixed(1)} s` : "";
      const stopping =
        m.stop_requested === "pause" && m.active
          ? " · wstrzymam po bieżącym fragmencie…"
          : m.stop_requested === "abort" && m.active
            ? " · zatrzymuję…"
            : m.stop_requested === "delete"
              ? " · usuwam po bieżącym fragmencie…"
              : "";
      const hint = STATUS_HINT[m.status] || "";
      card.innerHTML = `
        <h2>${escapeHtml(m.title)}</h2>
        <div class="meta">
          <span class="status-${m.status}">${escapeHtml(m.status)}</span>${progress}${dur}${stopping}
        </div>
        ${hint ? `<p class="meta">${escapeHtml(hint)}</p>` : ""}
        ${progressSectionHtml(m)}
        <p class="message card-error" hidden role="alert"></p>
        <div class="card-actions" data-id="${escapeHtml(m.id)}"></div>`;
      if (m.status === "error" && m.error) {
        card.innerHTML += `<p class="meta">${escapeHtml(String(m.error).split("\n")[0])}</p>`;
      }

      const actions = card.querySelector(".card-actions");
      const errEl = card.querySelector(".card-error");

      async function runCardAction(fn) {
        listActionInFlight = true;
        try {
          await fn();
          listActionInFlight = false;
          await loadList(listLoadGeneration);
        } catch (e) {
          if (errEl) {
            errEl.hidden = false;
            errEl.textContent = String(e.message || e);
          }
        } finally {
          listActionInFlight = false;
        }
      }

      if (m.status === "ready") {
        actions.append(makeBtn("Otwórz", () => navigateTo({ view: "reader", id: m.id })));
      }
      if (["uploaded", "draft"].includes(m.status)) {
        actions.append(makeBtn("Edytuj", () => navigateTo({ view: "edit", id: m.id })));
      }
      if (["queued", "processing"].includes(m.status)) {
        const stopNow = () =>
          runCardAction(async () => {
            try {
              await apiPost(`/api/materials/${encodeURIComponent(m.id)}/abort`);
            } catch (e) {
              if (!String(e.message || e).includes("paused")) throw e;
            }
          });
        if (m.stop_requested === "pause" && m.active) {
          actions.append(
            makeBtn("Anuluj wstrzymanie", () =>
              runCardAction(() => apiPost(`/api/materials/${encodeURIComponent(m.id)}/resume`))
            )
          );
          actions.append(makeBtn("Zatrzymaj natychmiast", stopNow, false, "danger"));
        } else if (!m.stop_requested) {
          actions.append(
            makeBtn("Wstrzymaj", () =>
              runCardAction(() => apiPost(`/api/materials/${encodeURIComponent(m.id)}/pause`))
            )
          );
        }
      }
      if (["paused", "error"].includes(m.status)) {
        actions.append(
          makeBtn("Wznów", () =>
            runCardAction(() => apiPost(`/api/materials/${encodeURIComponent(m.id)}/resume`))
          )
        );
        actions.append(makeBtn("Edytuj tekst", () => navigateTo({ view: "edit", id: m.id })));
      }
      actions.append(
        makeBtn(
          "Usuń",
          () => {
            if (!confirm(`Usunąć „${m.title}"?`)) return;
            return runCardAction(() => apiDelete(m.id));
          },
          false,
          "danger"
        )
      );

      grid.appendChild(card);
    }
    startEtaTick(items);
  } catch (err) {
    if (generation !== listLoadGeneration || activeView !== "list") return;
    appEl.innerHTML = `<p class="message">Błąd listy: ${escapeHtml(String(err))}</p>`;
  }
}

function makeBtn(label, onClick, disabled = false, extraClass = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `btn sm ${extraClass}`.trim();
  b.textContent = label;
  b.disabled = !!disabled;
  b.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    if (b.disabled) return;
    b.disabled = true;
    try {
      await onClick();
    } finally {
      b.disabled = false;
    }
  });
  return b;
}

async function showNew() {
  clearPoll();
  activeView = "new";
  appEl.innerHTML = `<p class="empty">Ładowanie formularza…</p>`;

  let filters = [];
  try {
    filters = await fetchFilters();
  } catch (err) {
    appEl.innerHTML = `<p class="message">Błąd: ${escapeHtml(String(err))}</p>`;
    return;
  }

  const filterHtml = filters
    .map(
      (f) => `
    <label>
      <input type="checkbox" name="filter" value="${escapeHtml(f.id)}" ${f.default ? "checked" : ""} />
      ${escapeHtml(f.label)}
    </label>`
    )
    .join("");

  appEl.innerHTML = `
    <form class="form-panel" id="new-form">
      <h2 class="section-title">Nowy materiał</h2>
      <p class="meta">Po wgraniu od razu wyciągniemy tekst — potem możesz go przycinać i wygenerować audio.</p>
      <label>Tytuł pracy (spacje → _)
        <input type="text" name="title" id="title-input" autocomplete="off" placeholder="Attention Free Transformer" />
      </label>
      <p class="meta">ID zasobu: <span class="mono" id="id-preview">—</span></p>
      <fieldset>
        <legend>Źródło PDF</legend>
        <p class="meta">Wybierz plik albo wklej link — wystarczy jedno. Serwer sprawdza, że to PDF.</p>
        <div class="source-row">
          <label>Plik PDF
            <input type="file" name="pdf" id="pdf-input" accept="application/pdf" />
          </label>
          <span class="source-or" aria-hidden="true">lub</span>
          <label>Link do PDF (http/https)
            <input type="url" name="pdf_url" id="pdf-url-input" inputmode="url" placeholder="https://…/paper.pdf" autocomplete="off" />
          </label>
        </div>
      </fieldset>
      <fieldset id="pages-fieldset" disabled>
        <legend>Zakres stron</legend>
        <div class="row-2">
          <label>Od
            <input type="number" id="page-from" min="1" value="1" />
          </label>
          <label>Do
            <input type="number" id="page-to" min="1" value="1" />
          </label>
        </div>
        <p class="meta" id="page-hint">Wybierz plik lub podaj link — liczba stron pojawi się przed ekstrakcją.</p>
      </fieldset>
      <fieldset>
        <legend>Filtry tekstu</legend>
        <div class="filter-grid">${filterHtml}</div>
      </fieldset>
      <button type="submit" class="btn" id="new-submit">Wgraj i wyciągnij tekst</button>
      <p class="message" id="new-status" role="status"></p>
    </form>`;

  const titleInput = document.getElementById("title-input");
  const idPreview = document.getElementById("id-preview");
  const pdfInput = document.getElementById("pdf-input");
  const pdfUrlInput = document.getElementById("pdf-url-input");
  const pagesFieldset = document.getElementById("pages-fieldset");
  const pageFrom = document.getElementById("page-from");
  const pageTo = document.getElementById("page-to");
  const pageHint = document.getElementById("page-hint");
  const statusEl = document.getElementById("new-status");

  let localNumPages = null;

  function titleFromUrl(url) {
    try {
      const u = new URL(url.trim());
      const base = u.pathname.split("/").filter(Boolean).pop() || "";
      return base.replace(/\.pdf$/i, "");
    } catch {
      return "";
    }
  }

  function refreshId() {
    let t = titleInput.value.trim();
    if (!t && pdfInput.files?.[0]) {
      t = pdfInput.files[0].name.replace(/\.pdf$/i, "");
    }
    if (!t && pdfUrlInput.value.trim()) {
      t = titleFromUrl(pdfUrlInput.value);
    }
    idPreview.textContent = sanitizeTitlePreview(t);
  }

  titleInput.addEventListener("input", refreshId);

  pdfUrlInput.addEventListener("input", () => {
    if (pdfUrlInput.value.trim()) {
      pdfInput.value = "";
      localNumPages = null;
      pagesFieldset.disabled = true;
      pageHint.textContent =
        "Po pobraniu z linku ustawimy liczbę stron — przed ekstrakcją możesz zmienić zakres.";
    }
    refreshId();
  });

  pdfInput.addEventListener("change", async () => {
    if (pdfInput.files?.[0]) {
      pdfUrlInput.value = "";
    }
    refreshId();
    const file = pdfInput.files?.[0];
    if (!file) return;
    statusEl.textContent = "Liczenie stron w PDF…";
    try {
      const buf = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
      localNumPages = pdf.numPages;
      pagesFieldset.disabled = false;
      pageFrom.min = "1";
      pageFrom.max = String(localNumPages);
      pageTo.min = "1";
      pageTo.max = String(localNumPages);
      pageFrom.value = "1";
      pageTo.value = String(localNumPages);
      pageHint.textContent = `Ten plik ma ${localNumPages} stron (numeracja od 1).`;
      statusEl.textContent = "";
    } catch (err) {
      statusEl.textContent = `Nie można odczytać PDF: ${err.message}`;
      pagesFieldset.disabled = true;
    }
  });

  document.getElementById("new-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const submitBtn = document.getElementById("new-submit");
    const filterEls = document.querySelectorAll('input[name="filter"]:checked');
    const filtersSelected = [...filterEls].map((el) => el.value);

    const pdf = pdfInput.files?.[0];
    const pdfUrl = pdfUrlInput.value.trim();
    const hasFile = !!(pdf?.size);
    const hasUrl = /^https?:\/\//i.test(pdfUrl);
    if (hasFile === hasUrl) {
      statusEl.textContent = "Wybierz plik PDF albo wklej link (nie oba naraz).";
      return;
    }

    const fd = new FormData();
    const title = titleInput.value.trim();
    if (title) fd.append("title", title);
    if (hasFile) fd.append("pdf", pdf);
    else fd.append("pdf_url", pdfUrl);

    submitBtn.disabled = true;

    try {
      statusEl.textContent = hasUrl
        ? "1/3 Pobieranie PDF z linku…"
        : "1/3 Wgrywanie PDF na serwer…";
      const res = await fetch("/api/materials", { method: "POST", body: fd });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(formatApiError(body, res.statusText));

      const numPages = body.num_pages;
      localNumPages = numPages;
      pagesFieldset.disabled = false;
      pageFrom.min = "1";
      pageFrom.max = String(numPages);
      pageTo.min = "1";
      pageTo.max = String(numPages);
      if (!pageTo.value || Number(pageTo.value) === 1) {
        pageTo.value = String(numPages);
      }
      pageHint.textContent = `Materiał ma ${numPages} stron (numeracja od 1).`;

      let page_from = Number(pageFrom.value) || 1;
      let page_to = Number(pageTo.value) || numPages;
      page_from = Math.max(1, Math.min(page_from, numPages));
      const clampedTo = Math.min(page_to, numPages);
      statusEl.textContent = `2/3 Ekstrakcja tekstu (strony ${page_from}–${clampedTo})…`;

      const extracted = await apiPost(
        `/api/materials/${encodeURIComponent(body.id)}/extract`,
        { filters: filtersSelected, page_from, page_to: clampedTo }
      );

      statusEl.textContent = `3/3 Gotowe (${extracted.chars} znaków). Otwieram edytor…`;
      navigateTo({ view: "edit", id: body.id });
    } catch (err) {
      statusEl.textContent = `Błąd: ${err.message}`;
      submitBtn.disabled = false;
    }
  });
}

async function runExtract(materialId, { page_from, page_to, filters, statusEl, editor, editorFieldset, updateStats }) {
  statusEl.textContent = `Ekstrakcja stron ${page_from}–${page_to}…`;
  const data = await apiPost(`/api/materials/${encodeURIComponent(materialId)}/extract`, {
    filters,
    page_from,
    page_to,
  });
  editorFieldset.hidden = false;
  editor.value = data.text;
  editor.defaultValue = data.text;
  editor.scrollTop = 0;
  editor.dispatchEvent(new Event("input", { bubbles: true }));
  updateStats();
  const hint = document.getElementById("edit-hint");
  if (hint) {
    hint.textContent = STATUS_HINT.draft;
  }
  statusEl.textContent = `Wyciągnięto ${data.chars} znaków, ${data.words} słów — usuń co trzeba i kliknij Generuj.`;
  editorFieldset.scrollIntoView({ behavior: "smooth", block: "start" });
  return data;
}

async function showEdit(materialId) {
  clearPoll();
  activeView = "edit";
  appEl.innerHTML = `<p class="empty">Ładowanie…</p>`;

  let meta;
  let filters;
  try {
    [meta, filters] = await Promise.all([fetchMaterial(materialId), fetchFilters()]);
  } catch (err) {
    appEl.innerHTML = `<p class="message">Błąd: ${escapeHtml(String(err))}</p>`;
    return;
  }

  const numPages = meta.num_pages || 1;
  const pageFrom = meta.page_from || 1;
  const pageTo = meta.page_to || numPages;
  const activeFilters = new Set(meta.filters || filters.filter((f) => f.default).map((f) => f.id));

  let editorText = "";
  const preferExtracted = ["draft", "uploaded"].includes(meta.status);
  const extractedRes = await fetch(
    `/api/materials/${encodeURIComponent(materialId)}/extracted`
  );
  const textRes = await fetch(`/api/materials/${encodeURIComponent(materialId)}/text`);

  if (preferExtracted && extractedRes.ok) {
    editorText = await extractedRes.text();
  } else if (textRes.ok) {
    editorText = await textRes.text();
  } else if (extractedRes.ok) {
    editorText = await extractedRes.text();
  }

  const filterHtml = filters
    .map(
      (f) => `
    <label>
      <input type="checkbox" name="filter" value="${escapeHtml(f.id)}" ${activeFilters.has(f.id) ? "checked" : ""} />
      ${escapeHtml(f.label)}
    </label>`
    )
    .join("");

  appEl.innerHTML = `
    <div class="form-panel" id="edit-panel">
      <h2 class="section-title">${escapeHtml(meta.title)}</h2>
      <p class="meta">ID: <span class="mono">${escapeHtml(materialId)}</span> · status: ${escapeHtml(meta.status)}</p>
      <p class="message" id="edit-hint">${escapeHtml(STATUS_HINT[meta.status] || "")}</p>

      <fieldset>
        <legend>Ekstrakcja z PDF</legend>
        <div class="row-2">
          <label>Strona od
            <input type="number" id="page-from" min="1" max="${numPages}" value="${pageFrom}" />
          </label>
          <label>Strona do
            <input type="number" id="page-to" min="1" max="${numPages}" value="${pageTo}" />
          </label>
        </div>
        <p class="meta">PDF ma ${numPages} stron (numeracja od 1).</p>
        <div class="filter-grid">${filterHtml}</div>
        <button type="button" class="btn" id="btn-extract">Wyciągnij tekst</button>
      </fieldset>

      <fieldset id="editor-fieldset" ${editorText ? "" : 'hidden'}>
        <legend>Tekst do TTS (tylko wycinanie)</legend>
        <p class="meta">Możesz usuwać fragmenty; dopisywanie jest zablokowane.</p>
        <textarea id="text-editor" class="editor">${escapeHtml(editorText)}</textarea>
        <div class="row-2">
          <button type="button" class="btn ghost sm" id="btn-cut-sel">Usuń zaznaczenie</button>
          <span class="meta" id="text-stats">—</span>
        </div>
        <div class="row-2">
          <label>Język
            <select id="gen-language">
              <option value="English" ${meta.language === "English" ? "selected" : ""}>English</option>
              <option value="Polish" ${meta.language === "Polish" ? "selected" : ""}>Polish</option>
            </select>
          </label>
          <label>Speaker
            <select id="gen-speaker">
              ${["ryan", "aiden", "dylan", "eric", "serena", "sohee", "vivian", "uncle_fu", "ono_anna"]
                .map(
                  (s) =>
                    `<option value="${s}" ${meta.speaker === s ? "selected" : ""}>${s}</option>`
                )
                .join("")}
            </select>
          </label>
        </div>
        <button type="button" class="btn" id="btn-generate">Generuj audio + WebVTT</button>
      </fieldset>

      <p class="meta" id="edit-status"></p>
    </div>`;

  const editor = document.getElementById("text-editor");
  const editorFieldset = document.getElementById("editor-fieldset");
  const statusEl = document.getElementById("edit-status");
  const statsEl = document.getElementById("text-stats");

  function updateStats() {
    if (!editor) return;
    const t = editor.value;
    const words = t.trim() ? t.trim().split(/\s+/).length : 0;
    const chunks = Math.max(1, Math.ceil(t.length / CHUNK_CHARS));
    statsEl.textContent = `${t.length} znaków · ${words} słów · ~${chunks} kawałków TTS`;
  }

  if (editor) {
    editor.addEventListener("beforeinput", (ev) => {
      const allowed = new Set([
        "deleteContentBackward",
        "deleteContentForward",
        "deleteByCut",
        "deleteWordBackward",
        "deleteWordForward",
        "historyUndo",
        "historyRedo",
      ]);
      if (!allowed.has(ev.inputType)) {
        ev.preventDefault();
      }
    });
    editor.addEventListener("input", updateStats);
    editor.addEventListener("keydown", (ev) => {
      if (ev.key.length === 1 && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        ev.preventDefault();
      }
    });
    editor.addEventListener("paste", (ev) => ev.preventDefault());
    updateStats();
  }

  document.getElementById("btn-cut-sel")?.addEventListener("click", () => {
    if (!editor) return;
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    if (start === end) return;
    editor.setRangeText("", start, end, "end");
    updateStats();
  });

  document.getElementById("btn-extract").addEventListener("click", async () => {
    const page_from = Number(document.getElementById("page-from").value);
    const page_to = Number(document.getElementById("page-to").value);
    const filterEls = document.querySelectorAll(
      "#edit-panel input[name='filter']:checked"
    );
    const filtersSelected = [...filterEls].map((el) => el.value);
    const btn = document.getElementById("btn-extract");
    btn.disabled = true;
    if (
      editor?.value &&
      !confirm(
        "Ponowna ekstrakcja zastąpi cały tekst w edytorze (ręczne wycięcia przepadną). Kontynuować?"
      )
    ) {
      btn.disabled = false;
      return;
    }
    try {
      await runExtract(materialId, {
        page_from,
        page_to,
        filters: filtersSelected,
        statusEl,
        editor,
        editorFieldset,
        updateStats,
      });
    } catch (err) {
      statusEl.textContent = `Błąd: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  });

  if (meta.status === "uploaded" && !editorText) {
    const filterEls = document.querySelectorAll('input[name="filter"]:checked');
    const filtersSelected = [...filterEls].map((el) => el.value);
    try {
      await runExtract(materialId, {
        page_from: pageFrom,
        page_to: pageTo,
        filters: filtersSelected,
        statusEl,
        editor,
        editorFieldset,
        updateStats,
      });
    } catch (err) {
      statusEl.textContent = `Auto-ekstrakcja nie powiodła się: ${err.message}. Użyj „Wyciągnij tekst”.`;
    }
  }

  document.getElementById("btn-generate")?.addEventListener("click", async () => {
    if (!editor) return;
    const text = editor.value;
    statusEl.textContent = "Kolejkowanie…";
    try {
      await apiPost(`/api/materials/${encodeURIComponent(materialId)}/generate`, {
        text,
        speaker: document.getElementById("gen-speaker").value,
        language: document.getElementById("gen-language").value,
      });
      statusEl.textContent = "W kolejce — wracam na listę.";
      navigateTo({ view: "list" });
    } catch (err) {
      statusEl.textContent = `Błąd: ${err.message}`;
    }
  });
}

function stopReaderPlayback() {
  if (readerState?.rafId) cancelAnimationFrame(readerState.rafId);
  readerState = null;
  if (CSS.highlights) {
    CSS.highlights.delete("tts-word");
    CSS.highlights.delete("tts-line");
  }
}

async function showReader(materialId) {
  clearPoll();
  activeView = "read";
  stopReaderPlayback();
  appEl.innerHTML = `<p class="empty">Ładowanie czytnika…</p>`;

  let meta;
  try {
    meta = await fetchMaterial(materialId);
  } catch (err) {
    appEl.innerHTML = `<p class="message">Błąd: ${escapeHtml(String(err))}</p>`;
    return;
  }
  const title = meta.title || materialId;
  const pageFrom = meta.page_from || 1;
  const pageTo = meta.page_to || meta.num_pages || 1;

  appEl.innerHTML = `
    <div id="reader-root">
      <div class="reader-toolbar">
        <strong>${escapeHtml(title)}</strong>
        <p class="meta">Strony ${pageFrom}–${pageTo}</p>
        <audio id="reader-audio" controls preload="metadata"></audio>
        <div class="reader-rate">
          <label for="playback-rate">Tempo</label>
          <select id="playback-rate" aria-label="Tempo odtwarzania">
            <option value="0.8">0.8×</option>
            <option value="1" selected>1×</option>
            <option value="1.25">1.25×</option>
            <option value="1.5">1.5×</option>
            <option value="2">2×</option>
          </select>
        </div>
        <div class="reader-stats">
          <span id="coverage-label">Pokrycie dopasowania: —</span>
          <span id="time-label"></span>
        </div>
      </div>
      <div class="pdf-pages" id="pdf-pages"></div>
    </div>`;

  const audio = document.getElementById("reader-audio");
  audio.src = `/api/materials/${encodeURIComponent(materialId)}/wav`;
  const rateEl = document.getElementById("playback-rate");
  rateEl?.addEventListener("change", () => {
    const rate = Number(rateEl.value);
    audio.playbackRate = rate;
    audio.preservesPitch = true;
  });
  if ("mediaSession" in navigator) {
    try {
      navigator.mediaSession.metadata = new MediaMetadata({
        title,
        artist: "TTS PDF Reader",
      });
      navigator.mediaSession.setActionHandler("play", () => audio.play());
      navigator.mediaSession.setActionHandler("pause", () => audio.pause());
      navigator.mediaSession.setActionHandler("seekbackward", () => {
        audio.currentTime = Math.max(0, audio.currentTime - 5);
      });
      navigator.mediaSession.setActionHandler("seekforward", () => {
        audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 5);
      });
    } catch {
      /* unsupported handlers on some browsers */
    }
  }

  const pdfUrl = `/api/materials/${encodeURIComponent(materialId)}/pdf`;
  const vttUrl = `/api/materials/${encodeURIComponent(materialId)}/vtt`;

  const [vttRes, pdf] = await Promise.all([
    fetch(vttUrl),
    pdfjsLib.getDocument(pdfUrl).promise,
  ]);
  const vttText = await vttRes.text();
  const cues = parseVtt(vttText);

  const pagesEl = document.getElementById("pdf-pages");
  const pageEls = await renderPdfPages(pdf, pagesEl, pageFrom, pageTo, PDF_SCALE);

  const { keyStream, streamPos } = buildCharStream(pageEls);
  const { cueStarts, cueEnds, coverage, mappedCues, totalCues } = alignCuesToStream(
    cues,
    keyStream
  );

  document.getElementById("coverage-label").textContent =
    `Pokrycie dopasowania: ${coverage.toFixed(1)}% (${mappedCues}/${totalCues} cue)`;

  readerState = {
    audio,
    cues,
    cueStarts,
    cueEnds,
    streamPos,
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

    let streamIdx = -1;
    for (let i = 0; i < readerState.streamPos.length; i += 1) {
      const n = readerState.streamPos[i].node;
      const parent = n.parentElement;
      if (parent === span || span.contains(n)) {
        streamIdx = i;
        break;
      }
    }
    if (streamIdx < 0) return;

    let bestCue = -1;
    for (let ci = 0; ci < readerState.cueStarts.length; ci += 1) {
      const s = readerState.cueStarts[ci];
      const e = readerState.cueEnds[ci];
      if (s >= 0 && streamIdx >= s && streamIdx <= e) {
        bestCue = ci;
        break;
      }
    }
    if (bestCue < 0) {
      for (let ci = 0; ci < readerState.cueStarts.length; ci += 1) {
        if (readerState.cueStarts[ci] >= 0 && readerState.cueStarts[ci] <= streamIdx) {
          bestCue = ci;
        }
      }
    }
    if (bestCue >= 0) {
      readerState.audio.currentTime = readerState.cues[bestCue].start;
    }
  });

  audio.addEventListener("play", () => tickHighlight());
  audio.addEventListener("seeked", () => {
    if (readerState) {
      readerState.lastCueIndex = -1;
      tickHighlight();
    }
  });
  audio.addEventListener("pause", () => {
    if (readerState?.rafId) {
      cancelAnimationFrame(readerState.rafId);
      readerState.rafId = null;
    }
  });
  audio.addEventListener("ended", () => {
    if (CSS.highlights) {
      CSS.highlights.delete("tts-word");
      CSS.highlights.delete("tts-line");
    }
  });
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
  if (t > cues[ans].end + 0.08) {
    for (let i = ans + 1; i < cues.length; i += 1) {
      if (t >= cues[i].start && t <= cues[i].end + 0.08) return i;
    }
    return -1;
  }
  return ans;
}

function tickHighlight() {
  if (!readerState) return;
  try {
    const { audio, cues, cueStarts, cueEnds, streamPos } = readerState;
    const t = audio.currentTime;
    const cueIdx = findActiveCueIndex(cues, t);

    document.getElementById("time-label").textContent =
      `${t.toFixed(2)} s` + (cueIdx >= 0 ? ` · cue ${cueIdx + 1}` : "");

    if (cueIdx !== readerState.lastCueIndex) {
      readerState.lastCueIndex = cueIdx;
      if (cueIdx >= 0 && CSS.highlights) {
        const wordRanges = rangesForCue(cueIdx, cueStarts, cueEnds, streamPos);
        const lineRanges = rangesForLine(cueIdx, cueStarts, cueEnds, streamPos);
        if (wordRanges.length) {
          CSS.highlights.set("tts-line", new Highlight(...lineRanges));
          CSS.highlights.set("tts-word", new Highlight(...wordRanges));
          maybeScrollToCue(cueIdx, cueStarts, streamPos);
        } else {
          CSS.highlights.delete("tts-word");
          CSS.highlights.delete("tts-line");
        }
      } else if (CSS.highlights) {
        CSS.highlights.delete("tts-word");
        CSS.highlights.delete("tts-line");
      }
    }

    if (!audio.paused) {
      readerState.rafId = requestAnimationFrame(tickHighlight);
    }
  } catch (err) {
    console.warn("tickHighlight:", err);
    if (!readerState?.audio?.paused) {
      readerState.rafId = requestAnimationFrame(tickHighlight);
    }
  }
}

function maybeScrollToCue(cueIdx, cueStarts, streamPos) {
  if (!readerState) return;
  if (performance.now() - readerState.lastUserScroll < USER_SCROLL_COOLDOWN_MS) return;
  const start = cueStarts[cueIdx];
  if (start < 0) return;
  const pos = streamPos[start];
  if (!pos) return;
  const el = pos.node.parentElement || pos.node;
  el.scrollIntoView({ block: "center", behavior: "smooth" });
}
