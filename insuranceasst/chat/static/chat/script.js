// script.js — split-workspace client: chat + sources (RAG), damage analysis (vision), documents, tools
document.addEventListener('DOMContentLoaded', () => {
  // ---------- STATE ----------
  let selectedDoc = null;   // active policy filename
  let visionFacts = null;   // photo-analysis findings, sent as known facts for the rest of the conversation
  const history = [];       // [{role, content}] completed turns, sent so follow-ups keep their context
  const HISTORY_TURNS = 6;
  let aborter = null;       // AbortController for the streaming answer
  let busy = false;
  const selectedNames = new Set();

  // Ultralytics' default palette, indexed by CarDD class id, so swatches match the mask colours
  const CLASS_COLORS = {
    'crack': '#042AFF', 'dent': '#0BDBEB', 'glass shatter': '#F3F3F3',
    'lamp broken': '#00DFB7', 'scratch': '#111F68', 'tire flat': '#FF6FDD'
  };

  // ---------- ELEMENTS ----------
  const $ = (id) => document.getElementById(id);
  const thread = $('thread'), emptyState = $('emptyState');
  const composer = $('composer'), input = $('message'), sendBtn = $('sendBtn');
  const imgInput = $('imgInput');
  const policySelect = $('policySelect');
  const visionBody = $('visionBody'), sourcesBody = $('sourcesBody'), sourcesHint = $('sourcesHint');
  const docsModalEl = $('docsModal');
  const filesDiv = $('files'), filesCount = $('files-count'), chkAll = $('chk-all');
  const deleteBtn = $('btn-delete'), reindexBtn = $('btn-reindex'), clearVdbBtn = $('btn-clear-vdb');
  const uploadForm = $('upload-form'), fileInput = $('file');

  // ---------- HELPERS ----------
  function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function store(key, val) {
    try { val == null ? localStorage.removeItem(key) : localStorage.setItem(key, val); } catch (e) {}
  }
  function recall(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  const toastEl = $('toast');
  function toast(msg) {
    $('toastBody').textContent = msg;
    bootstrap.Toast.getOrCreateInstance(toastEl, { delay: 3500 }).show();
  }

  function modal(el) { return bootstrap.Modal.getOrCreateInstance(el); }

  function scrollThread() { thread.scrollTop = thread.scrollHeight; }
  function nearBottom() { return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80; }

  // ---------- THEME ----------
  const themeBtn = $('themeToggle');
  function applyThemeIcon() {
    const dark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
    themeBtn.innerHTML = dark ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon-stars"></i>';
  }
  themeBtn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-bs-theme', next);
    store('theme', next);
    applyThemeIcon();
  });
  applyThemeIcon();

  // ---------- POLICY SELECTION ----------
  function setSelectedDoc(name) {
    selectedDoc = name || null;
    policySelect.value = selectedDoc || '';
    policySelect.parentElement.classList.toggle('unset', !selectedDoc);
    input.disabled = !selectedDoc;
    input.placeholder = selectedDoc ? `Ask about ${selectedDoc}…` : 'Select a policy to start';
    updateSend();
    store('policy', selectedDoc);
    filesDiv.querySelectorAll('.file-row').forEach(r => r.classList.toggle('active', r.dataset.name === selectedDoc));
  }

  function updateSend() { sendBtn.disabled = busy || !selectedDoc || !input.value.trim(); }

  async function fetchFiles() {
    const res = await fetch('/api/files/');
    const data = await res.json();
    return data.files || [];
  }

  async function loadPolicies(files) {
    files = files || await fetchFiles();
    const keep = selectedDoc || recall('policy');
    policySelect.innerHTML = '<option value="">Select a policy…</option>' +
      files.map(f => `<option value="${escapeHtml(f.name)}">${escapeHtml(f.name)}</option>`).join('');
    const exists = files.some(f => f.name === keep);
    setSelectedDoc(exists ? keep : null);
  }

  policySelect.addEventListener('change', () => setSelectedDoc(policySelect.value));

  // ---------- CHAT ----------
  function hideEmpty() { if (emptyState) emptyState.remove(); }

  function addUser(text, photoUrl) {
    hideEmpty();
    const row = document.createElement('div');
    row.className = 'msg user';
    const body = document.createElement('div');
    body.className = 'body';
    if (photoUrl) {
      const img = document.createElement('img');
      img.className = 'photo';
      img.src = photoUrl;
      img.alt = text || 'Uploaded photo';
      body.appendChild(img);
    } else {
      body.textContent = text;
    }
    row.appendChild(body);
    thread.appendChild(row);
    scrollThread();
    return row;
  }

  function addBot(text = '') {
    hideEmpty();
    const row = document.createElement('div');
    row.className = 'msg bot';
    row.innerHTML = '<div class="avatar"><i class="bi bi-shield-check"></i></div><div class="body"><div class="text"></div></div>';
    const textEl = row.querySelector('.text');
    textEl.textContent = text;
    thread.appendChild(row);
    scrollThread();
    return { row, textEl, body: row.querySelector('.body') };
  }

  function addMeta(body, chips) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    chips.forEach(c => meta.appendChild(c));
    body.appendChild(meta);
  }

  // Minimal, safe formatting for answers: escape first, then **bold**, "- " bullets and [n] citations
  function renderAnswer(text, nSources) {
    // drop an empty questions section ("To give you a precise answer: No questions are needed.")
    text = text.replace(/\n+\**to give you a precise answer:?\**:?\s*(?:[-•\d.\s]*)(no (additional |further )?questions[^\n]*|n\/a)\s*$/i, '');
    const cite = (m, nums) => nums.split(/\s*,\s*/).map(n => {
      const k = parseInt(n, 10);
      return k >= 1 && k <= nSources ? `<button type="button" class="cite" data-n="${k}">${k}</button>` : '';
    }).join('');
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/^\s*[-*]\s+/gm, '• ')
      .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, cite);
  }

  function sourceLabel(s) {
    if (!s.page) return s.name;
    return s.page_end && s.page_end > s.page ? `${s.name} · p.${s.page}–${s.page_end}` : `${s.name} · p.${s.page}`;
  }

  // "Part 6 Protection… / Coverage UU Rental Reimbursement Coverage | …" -> "Coverage UU Rental Reimbursement Coverage"
  function sectionName(s) {
    const first = (s.section || '').split(' | ')[0];
    return first.split(' / ').pop().replace(/\s*__\s*/g, ' — ');
  }

  function renderSources(sources, question) {
    if (!sources.length) {
      sourcesBody.innerHTML = '<p class="muted small mb-0">No matching passages for the latest question.</p>';
      sourcesHint.textContent = '';
      return;
    }
    sourcesHint.textContent = `${sources.length} passage${sources.length === 1 ? '' : 's'}`;
    sourcesBody.innerHTML = sources.map((s, i) => `
      <details class="source" id="src-${i + 1}" ${i === 0 ? 'open' : ''}>
        <summary>
          <span class="num">${i + 1}</span>
          <span class="name">${escapeHtml(sourceLabel(s))}
            ${sectionName(s) ? `<small class="section">${escapeHtml(sectionName(s))}</small>` : ''}</span>
          <i class="bi bi-chevron-right chev"></i>
        </summary>
        <div class="snippet">${escapeHtml(s.snippet)}…</div>
      </details>`).join('');
    sourcesBody.setAttribute('aria-label', `Sources for: ${question}`);
  }

  function focusSource(i) {
    const el = $(`src-${i}`);
    if (!el) return;
    el.open = true;
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    el.classList.remove('flash');
    void el.offsetWidth;  // restart the highlight transition
    el.classList.add('flash');
    setTimeout(() => el.classList.remove('flash'), 1200);
  }

  async function ask(raw) {
    raw = (raw || '').trim();
    if (!raw) return;
    if (!selectedDoc) {
      toast('Select a policy first.');
      policySelect.focus();
      return;
    }
    if (aborter) aborter.abort();
    addUser(raw);
    input.value = '';
    await streamAnswer(raw);
  }

  async function streamAnswer(q) {
    aborter = new AbortController();
    busy = true; updateSend();
    const { textEl, body } = addBot();
    textEl.classList.add('pending');

    let buffer = '';
    let scheduled = false;
    const flush = () => {
      scheduled = false;
      if (!buffer) return;
      const stick = nearBottom();
      textEl.textContent += buffer;
      buffer = '';
      if (stick) scrollThread();
    };

    try {
      const res = await fetch('/api/chat/stream/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: q,
          doc: selectedDoc,
          history: history.slice(-HISTORY_TURNS),
          facts: visionFacts
        }),
        signal: aborter.signal
      });

      let sources = [];
      try { sources = JSON.parse(res.headers.get('X-Sources') || '[]'); } catch (e) {}
      renderSources(sources, q);

      if (!res.ok || !res.body) {
        textEl.textContent = (await res.text()) || ('HTTP ' + res.status);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        if (!scheduled) { scheduled = true; requestAnimationFrame(flush); }
      }
      flush();
      const answer = textEl.textContent.trim();
      // only completed answers become context for follow-ups
      if (answer && !answer.includes('[Error:')) {
        history.push({ role: 'user', content: q }, { role: 'assistant', content: answer });
      }
      textEl.innerHTML = renderAnswer(answer, sources.length);
      // citations belong to this answer: show its sources (the panel may be showing a later answer's)
      const openSource = (n) => { renderSources(sources, q); focusSource(n); };
      textEl.querySelectorAll('.cite').forEach(b => b.addEventListener('click', () => openSource(+b.dataset.n)));

      if (sources.length) {
        addMeta(body, sources.map((s, i) => {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'chip';
          chip.innerHTML = `<i class="bi bi-journal-text"></i>${i + 1} · ${escapeHtml(sourceLabel(s))}`;
          chip.addEventListener('click', () => { renderSources(sources, q); focusSource(i + 1); });
          return chip;
        }));
      }
    } catch (e) {
      flush();
      if (e.name === 'AbortError') textEl.textContent += ' [stopped]';
      else textEl.textContent = 'Error: ' + e.message;
    } finally {
      textEl.classList.remove('pending');
      aborter = null;
      busy = false; updateSend();
      if (nearBottom()) scrollThread();
    }
  }

  composer.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
  input.addEventListener('input', updateSend);

  document.querySelectorAll('.suggestion').forEach(btn => btn.addEventListener('click', () => {
    if (btn.dataset.action === 'photo') imgInput.click();
    else ask(btn.dataset.q);
  }));

  // ---------- DAMAGE ANALYSIS ----------
  function renderVisionLoading() {
    visionBody.innerHTML = '<div class="vision-loading"><div class="spinner-border spinner-border-sm"></div>Detecting damage…</div>';
  }

  function renderVision(data) {
    const byClass = {};
    (data.detections || []).forEach(d => {
      const c = byClass[d.class_name] || (byClass[d.class_name] = { count: 0, best: 0 });
      c.count += 1;
      c.best = Math.max(c.best, d.score);
    });
    const rows = Object.entries(byClass).sort((a, b) => b[1].best - a[1].best).map(([name, c]) => `
      <tr>
        <td><span class="swatch" style="background:${CLASS_COLORS[name] || 'var(--muted)'}"></span>${escapeHtml(name)}</td>
        <td>${c.count}</td>
        <td><span class="conf-bar"><span style="width:${Math.round(c.best * 100)}%"></span></span>${c.best.toFixed(2)}</td>
      </tr>`).join('');

    visionBody.innerHTML = `
      ${data.annotated_url ? `<img class="vision-img" src="${escapeHtml(data.annotated_url)}" alt="Photo with detected damage outlined">` : ''}
      <p class="vision-summary">${escapeHtml(data.summary || '')}</p>
      ${rows ? `<table class="det-table"><thead><tr><th>Damage</th><th>Count</th><th>Confidence</th></tr></thead><tbody>${rows}</tbody></table>`
             : '<p class="muted small mb-0">No damage detected.</p>'}`;
  }

  async function analyzeImage(file) {
    if (!file) return;
    const localUrl = URL.createObjectURL(file);
    const row = addUser(file.name || 'photo', localUrl);
    row.querySelector('img').onload = () => URL.revokeObjectURL(localUrl);
    renderVisionLoading();
    const { textEl, body } = addBot('Analyzing photo…');
    textEl.classList.add('pending');

    const fd = new FormData();
    fd.append('image', file);
    try {
      const res = await fetch('/api/vision/analyze/', { method: 'POST', body: fd });
      const raw = await res.text();
      let data;
      try { data = JSON.parse(raw); } catch { throw new Error(raw || `HTTP ${res.status}`); }
      if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);

      renderVision(data);
      const found = Object.entries(data.counts || {}).map(([k, v]) => `${v} ${k}`).join(', ');
      visionFacts = `Photo of the customer's vehicle analyzed by a damage detector: ${found || 'no damage detected'}. ${data.summary || ''}`.trim();
      textEl.textContent = (data.summary || 'Analysis complete.') +
        "\n\nAdd a few details (when, where, how it happened) and I'll check them against your policy.";
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.innerHTML = '<i class="bi bi-camera"></i>View damage analysis';
      chip.addEventListener('click', () => visionBody.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
      addMeta(body, [chip]);
    } catch (e) {
      textEl.textContent = 'Image analysis failed: ' + e.message;
      visionBody.innerHTML = `<p class="text-danger small mb-0">${escapeHtml(e.message)}</p>`;
    } finally {
      textEl.classList.remove('pending');
      scrollThread();
    }
  }

  const pickPhoto = () => imgInput.click();
  $('photoBtn').addEventListener('click', pickPhoto);
  $('photoBtn2').addEventListener('click', pickPhoto);
  visionBody.addEventListener('click', (e) => { if (e.target.closest('#dropzone')) pickPhoto(); });
  imgInput.addEventListener('change', () => {
    analyzeImage(imgInput.files?.[0]);
    imgInput.value = '';  // allow re-selecting the same file
  });

  // drag & drop onto the evidence pane or the chat
  ['.evidence-pane', '.chat-pane'].forEach(sel => {
    const zone = document.querySelector(sel);
    zone.addEventListener('dragover', (e) => {
      if (![...e.dataTransfer.items].some(i => i.type.startsWith('image/'))) return;
      e.preventDefault();
      $('dropzone')?.classList.add('drag');
    });
    zone.addEventListener('dragleave', () => $('dropzone')?.classList.remove('drag'));
    zone.addEventListener('drop', (e) => {
      const file = [...e.dataTransfer.files].find(f => f.type.startsWith('image/'));
      if (!file) return;
      e.preventDefault();
      $('dropzone')?.classList.remove('drag');
      analyzeImage(file);
    });
  });

  // click any analysis or chat photo to enlarge
  document.addEventListener('click', (e) => {
    const img = e.target.closest('.vision-img, .msg .photo');
    if (!img) return;
    const box = document.createElement('div');
    box.className = 'lightbox';
    box.innerHTML = `<img src="${escapeHtml(img.src)}" alt="">`;
    box.addEventListener('click', () => box.remove());
    document.body.appendChild(box);
  });

  // ---------- DOCUMENTS ----------
  function updateDeleteUI() { deleteBtn.disabled = selectedNames.size === 0; }

  async function loadFiles() {
    filesDiv.innerHTML = '<div class="file-empty">Loading…</div>';
    const files = await fetchFiles();
    selectedNames.clear();
    updateDeleteUI();
    chkAll.checked = false;
    filesCount.textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;

    if (!files.length) {
      filesDiv.innerHTML = '<div class="file-empty">No documents yet. Upload a policy above, then rebuild the index.</div>';
      return files;
    }
    filesDiv.innerHTML = '';
    files.forEach(f => {
      const row = document.createElement('div');
      row.className = 'file-row' + (f.name === selectedDoc ? ' active' : '');
      row.dataset.name = f.name;
      const icon = f.ext === '.pdf' ? 'bi-file-earmark-pdf' : 'bi-file-earmark-text';
      row.innerHTML = `
        <input class="form-check-input file-chk" type="checkbox" aria-label="Select ${escapeHtml(f.name)}">
        <i class="bi ${icon}"></i>
        <span class="name">${escapeHtml(f.name)}</span>
        <span class="muted small">${(f.size / 1024).toFixed(0)} KB</span>
        <button class="btn btn-sm btn-outline-primary use-btn">Use</button>
        <a class="btn btn-sm btn-ghost" href="${escapeHtml(f.url)}" target="_blank" title="Open"><i class="bi bi-box-arrow-up-right"></i></a>`;
      row.querySelector('.use-btn').addEventListener('click', () => {
        setSelectedDoc(f.name);
        modal(docsModalEl).hide();
      });
      row.querySelector('.file-chk').addEventListener('change', (e) => {
        if (e.currentTarget.checked) selectedNames.add(f.name); else selectedNames.delete(f.name);
        updateDeleteUI();
      });
      filesDiv.appendChild(row);
    });
    return files;
  }

  async function refreshAll() { await loadPolicies(await loadFiles()); }

  $('docsBtn').addEventListener('click', () => { modal(docsModalEl).show(); loadFiles(); });

  chkAll.addEventListener('change', () => {
    selectedNames.clear();
    filesDiv.querySelectorAll('.file-row').forEach(row => {
      row.querySelector('.file-chk').checked = chkAll.checked;
      if (chkAll.checked) selectedNames.add(row.dataset.name);
    });
    updateDeleteUI();
  });

  deleteBtn.addEventListener('click', async () => {
    if (!selectedNames.size) return;
    if (!confirm(`Delete ${selectedNames.size} file(s)? This cannot be undone.`)) return;
    deleteBtn.disabled = true;
    try {
      const r = await fetch('/api/files/delete/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names: Array.from(selectedNames) })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || 'Delete failed');
      if (selectedDoc && selectedNames.has(selectedDoc)) setSelectedDoc(null);
      await refreshAll();
      toast(`Deleted ${j.deleted.length} file(s)${j.missing.length ? `; missing: ${j.missing.join(', ')}` : ''}. Rebuild the index to drop them from search.`);
    } catch (e) {
      toast(e.message);
    } finally {
      updateDeleteUI();
    }
  });

  async function busyButton(btn, label, fn) {
    const html = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${label}`;
    try { await fn(); } finally { btn.disabled = false; btn.innerHTML = html; }
  }

  reindexBtn.addEventListener('click', () => busyButton(reindexBtn, 'Rebuilding…', async () => {
    try {
      const res = await fetch('/api/reindex/', { method: 'POST' });
      const data = await res.json();
      toast(data.ok ? `Index rebuilt from ${data.doc_count} document pages.` : 'Reindex failed: ' + (data.error || 'unknown error'));
    } catch (e) {
      toast('Reindex failed: ' + e.message);
    }
  }));

  clearVdbBtn.addEventListener('click', () => {
    if (!confirm('Clear the vector index? Documents stay; search returns nothing until you rebuild.')) return;
    busyButton(clearVdbBtn, 'Clearing…', async () => {
      try {
        const r = await fetch('/api/vectors/clear/', { method: 'POST' });
        const j = await r.json();
        toast(j.ok ? 'Index cleared.' : (j.error || 'Failed to clear index'));
      } catch (e) {
        toast(e.message);
      }
    });
  });

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files?.length) return;
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    try {
      const res = await fetch('/api/files/upload/', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'Upload failed');
      fileInput.value = '';
      selectedDoc = data.saved_as || selectedDoc;
      await refreshAll();
      toast('Uploaded. Rebuild the index to make it searchable.');
    } catch (err) {
      toast('Upload error: ' + err.message);
    }
  });

  // ---------- TOOLS ----------
  $('checkClaimBtn').addEventListener('click', () => {
    const num = $('claimNumber').value.trim();
    $('claimStatusResult').textContent = num
      ? `Claim ${num}: Received, awaiting adjuster review. (Demo response)`
      : 'Please enter a claim number.';
  });

  $('calcRun').addEventListener('click', () => {
    const dmg = parseFloat($('calcDamage').value || '0');
    const ded = parseFloat($('calcDeductible').value || '0');
    const lim = parseFloat($('calcLimit').value || 'NaN');
    const out = $('calcResult');
    if (isNaN(dmg) || isNaN(ded)) {
      out.innerHTML = '<span class="text-warning">Enter valid numbers for damage and deductible.</span>';
      return;
    }
    let payable = Math.max(0, dmg - ded);
    if (!isNaN(lim)) payable = Math.min(payable, lim);
    // insurer covers damage above the deductible (capped at the limit); you pay the rest
    const insurerPays = payable;
    const youPay = dmg - insurerPays;
    const fmt = (n) => n.toLocaleString(undefined, { style: 'currency', currency: 'USD' });
    out.innerHTML = `
      <div class="calc-out">
        <div><div class="k">You pay</div><div class="v">${fmt(youPay)}</div></div>
        <div><div class="k">Insurer pays (est.)</div><div class="v">${fmt(insurerPays)}</div></div>
      </div>`;
  });

  $('faqList').addEventListener('click', (e) => {
    const item = e.target.closest('.list-group-item');
    if (!item) return;
    modal($('faqsModal')).hide();
    ask(item.textContent);
  });

  $('supportSend').addEventListener('click', (e) => {
    // raw values here; the body is encoded once below
    const subject = encodeURIComponent('Insurance Assistant Support');
    const body = encodeURIComponent(`Name: ${$('supportName').value}\nEmail: ${$('supportEmail').value}\n\n${$('supportMsg').value}`);
    e.currentTarget.href = `mailto:support@example.com?subject=${subject}&body=${body}`;
  });

  // ---------- GLOBAL ----------
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && aborter) aborter.abort(); });

  // ---------- INIT ----------
  loadPolicies().catch(() => setSelectedDoc(null));
});
