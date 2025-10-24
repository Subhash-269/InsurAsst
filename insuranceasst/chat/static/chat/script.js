// script.js — unified client logic (chat, RAG, image analyze, quick menu, calculators)
document.addEventListener('DOMContentLoaded', () => {
  // ---------- STATE ----------
  let selectedDoc = null;          // selected policy filename
  let lastVision = null;           // {counts, detections, summary, annotated_url}
  let aborter = null;              // AbortController for streaming

  // ---------- ELEMENTS ----------
  const body = document.body;

  // Hero inputs
  const heroInput    = document.getElementById('message');
  const heroSend     = document.getElementById('send');
  const docBtnHero   = document.getElementById('docBtnHero');
  const imgBtnHero   = document.getElementById('imgBtnHero');
  const imgInputHero = document.getElementById('imgInputHero');
  const docPillHero  = document.getElementById('docPillHero');

  // Chat area
  const thread       = document.getElementById('thread');
  const chatInput    = document.getElementById('message2');
  const chatSend     = document.getElementById('send2');
  const docBtnChat   = document.getElementById('docBtnChat');
  const imgBtnChat   = document.getElementById('imgBtnChat');
  const imgInputChat = document.getElementById('imgInputChat');
  const docPillChat  = document.getElementById('docPillChat');

  // Docs modal & file mgmt
  const docsModal  = document.getElementById('docsModal');
  const filesDiv   = document.getElementById('files');
  const refreshBtn = document.getElementById('btn-refresh');
  const reindexBtn = document.getElementById('btn-reindex');
  const uploadForm = document.getElementById('upload-form');
  const fileInput  = document.getElementById('file');

  // Quick menu + utilities
  const fab           = document.getElementById('fabMenu');
  const offcanvasEl   = document.getElementById('quickMenu');
  const quickMenu     = offcanvasEl ? bootstrap.Offcanvas.getOrCreateInstance(offcanvasEl) : null;
  const menuMyPolicy  = document.getElementById('menuMyPolicy');

  const claimNumber       = document.getElementById('claimNumber');
  const checkClaimBtn     = document.getElementById('checkClaimBtn');
  const claimStatusResult = document.getElementById('claimStatusResult');

  const calcDamage = document.getElementById('calcDamage');
  const calcDeductible = document.getElementById('calcDeductible');
  const calcLimit = document.getElementById('calcLimit');
  const calcRun = document.getElementById('calcRun');
  const calcResult = document.getElementById('calcResult');

  const faqList = document.getElementById('faqList');

  const supportName  = document.getElementById('supportName');
  const supportEmail = document.getElementById('supportEmail');
  const supportMsg   = document.getElementById('supportMsg');
  const supportSend  = document.getElementById('supportSend');

  // ---------- HELPERS ----------
  function setSelectedDoc(name) {
    selectedDoc = name || null;
    const has = !!selectedDoc;

    if (heroSend) heroSend.disabled = !has;
    if (chatSend) chatSend.disabled  = !has;

    if (docPillHero) {
      docPillHero.textContent = has ? selectedDoc : 'No policy selected';
      docPillHero.classList.toggle('bad', !has);
    }
    if (docPillChat) {
      docPillChat.textContent = has ? selectedDoc : 'No policy selected';
      docPillChat.classList.toggle('bad', !has);
    }
  }

  function addBubble(text, who = 'bot') {
    const div = document.createElement('div');
    div.className = 'bubble ' + (who === 'user' ? 'user' : 'bot');
    div.textContent = text || '';
    thread.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return div;
  }

  function addImage(url, who = 'bot', caption = '') {
    const wrap = document.createElement('div');
    wrap.className = 'bubble ' + (who === 'user' ? 'user' : 'bot');
    wrap.innerHTML =
      `<div><img src="${url}" style="max-width:100%; border-radius:10px; border:1px solid #27304a"></div>` +
      (caption ? `<div style="margin-top:.4rem">${caption}</div>` : '');
    thread.appendChild(wrap);
    wrap.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return wrap;
  }

  async function streamAnswer(q) {
    if (aborter) aborter.abort();
    aborter = new AbortController();

    const botDiv = addBubble('', 'bot');

    try {
      const res = await fetch('/api/chat/stream/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q, doc: selectedDoc }),
        signal: aborter.signal
      });
      if (!res.ok) {
        const t = await res.text();
        botDiv.textContent = t || ('HTTP ' + res.status);
        return botDiv;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        botDiv.textContent += decoder.decode(value, { stream: true });
        botDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
      return botDiv;
    } catch (e) {
      if (e.name === 'AbortError') {
        botDiv.textContent += ' [stopped]';
      } else {
        botDiv.textContent = 'Error: ' + e.message;
      }
      return botDiv;
    } finally {
      aborter = null;
    }
  }

  // ---------- MESSAGE FLOWS ----------
  async function firstAsk() {
    const raw = (heroInput?.value || '').trim();
    if (!raw || !selectedDoc) return;
    document.body.classList.add('chat-started');
    addBubble(raw, 'user');
    if (heroInput) heroInput.value = '';
    if (chatInput) chatInput.focus();

    const q = lastVision
      ? `Context: Recent photo analysis detected ${JSON.stringify(lastVision.counts)}. Summary: "${lastVision.summary}". User details: ${raw}\n\nQuestion: What should I know / do per my policy?`
      : raw;

    await streamAnswer(q);
  }

  async function nextAsk() {
    const raw = (chatInput?.value || '').trim();
    if (!raw || !selectedDoc) return;
    addBubble(raw, 'user');
    if (chatInput) chatInput.value = '';

    const q = lastVision
      ? `Context: Recent photo analysis detected ${JSON.stringify(lastVision.counts)}. Summary: "${lastVision.summary}". User details: ${raw}\n\nQuestion: What should I know / do per my policy?`
      : raw;

    await streamAnswer(q);
  }

  // ---------- IMAGE ANALYSIS ----------
  async function analyzeImage(file) {
    if (!file) return;

    // Ensure the chat area is visible even if user started from HERO
    document.body.classList.add('chat-started');

    // 1) Show the selected image in the chat as a USER bubble
    const localUrl = URL.createObjectURL(file);
    const userImgWrap = addImage(localUrl, 'user', file.name || 'photo');
    // Revoke object URL after the image loads (avoid memory leaks)
    const userImgEl = userImgWrap.querySelector('img');
    if (userImgEl) {
      userImgEl.onload = () => URL.revokeObjectURL(localUrl);
    } else {
      // Fallback: revoke soon even if not found (shouldn't happen)
      setTimeout(() => URL.revokeObjectURL(localUrl), 5000);
    }

    // 2) Send to backend
    const fd = new FormData();
    // IMPORTANT: server expects 'image' (change to 'file' if your endpoint uses that)
    fd.append('image', file);

    const uploading = addBubble('Analyzing photo…', 'bot');
    try {
      const res = await fetch('/api/vision/analyze/', { method: 'POST', body: fd });
      const text = await res.text();
      let data = null;
      try { data = JSON.parse(text); } catch { throw new Error(text || `HTTP ${res.status}`); }
      if (!res.ok || !data?.ok) throw new Error(data?.error || `HTTP ${res.status}`);

      // 3) Show annotated result + summary
      if (data.annotated_url) addImage(data.annotated_url, 'bot');
      addBubble(data.summary || 'Detected damage summarized.', 'bot');

      lastVision = {
        counts: data.counts || {},
        detections: data.detections || [],
        summary: data.summary || '',
        annotated_url: data.annotated_url
      };

      addBubble("Does this look right? Add a few details (when/where/how). I'll use this with your selected policy to guide next steps.", 'bot');
    } catch (e) {
      addBubble('Image analysis failed: ' + e.message, 'bot');
    } finally {
      uploading.remove();
    }
  }

  // ---------- DOCS MODAL / FILE MGMT ----------
  async function loadFiles() {
    if (!filesDiv) return;
    filesDiv.innerHTML = '<div class="list-group-item bg-transparent text-secondary">Loading…</div>';
    const res = await fetch('/api/files/');
    const data = await res.json();
    filesDiv.innerHTML = '';

    (data.files || []).forEach(f => {
      const row = document.createElement('div');
      row.className = 'list-group-item d-flex justify-content-between align-items-center bg-transparent file-item';
      row.innerHTML = `
        <span>📄 ${f.name}</span>
        <div class="d-flex gap-2">
          <button class="btn btn-sm btn-outline-success">Select</button>
          <a class="btn btn-sm btn-outline-secondary" href="${f.url}" target="_blank">Open</a>
        </div>
      `;
      row.querySelector('button').addEventListener('click', () => {
        setSelectedDoc(f.name);
        const modal = bootstrap.Modal.getInstance(docsModal) || bootstrap.Modal.getOrCreateInstance(docsModal);
        modal.hide();
      });
      filesDiv.appendChild(row);
    });

    if ((data.files || []).length === 0) {
      filesDiv.innerHTML = '<div class="list-group-item bg-transparent text-secondary">No documents found. Upload one below, then click Rebuild Index.</div>';
    }
  }

  // ---------- BINDINGS: SEND & DOCS ----------
  heroSend?.addEventListener('click', firstAsk);
  heroInput?.addEventListener('keydown', e => { if (e.key === 'Enter') firstAsk(); });
  chatSend?.addEventListener('click', nextAsk);
  chatInput?.addEventListener('keydown', e => { if (e.key === 'Enter') nextAsk(); });

  [docBtnHero, docBtnChat].forEach(btn => btn?.addEventListener('click', () => {
    const m = bootstrap.Modal.getOrCreateInstance(docsModal);
    m.show();
    loadFiles();
  }));

  // ---------- BINDINGS: IMAGE UPLOADS ----------
  imgBtnHero?.addEventListener('click', () => imgInputHero?.click());
  imgInputHero?.addEventListener('change', () => {
    const file = imgInputHero?.files?.[0];
    if (file) analyzeImage(file);
    // Reset so selecting the same file again re-triggers 'change'
    if (imgInputHero) imgInputHero.value = '';
  });

  imgBtnChat?.addEventListener('click', () => imgInputChat?.click());
  imgInputChat?.addEventListener('change', () => {
    const file = imgInputChat?.files?.[0];
    if (file) analyzeImage(file);
    if (imgInputChat) imgInputChat.value = '';
  });

  // ---------- BINDINGS: FILE ACTIONS ----------
  refreshBtn?.addEventListener('click', loadFiles);
  reindexBtn?.addEventListener('click', async () => {
    reindexBtn.disabled = true; reindexBtn.textContent = 'Rebuilding…';
    try {
      const res = await fetch('/api/reindex/', { method: 'POST' });
      const data = await res.json();
      alert(data.ok ? 'Index rebuilt.' : ('Reindex failed: ' + (data.error || 'Unknown error')));
      await loadFiles();
    } catch (e) {
      alert('Reindex failed: ' + e.message);
    } finally {
      reindexBtn.disabled = false; reindexBtn.textContent = 'Rebuild Index';
    }
  });
  uploadForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput?.files?.length) return;
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    try {
      const res = await fetch('/api/files/upload/', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'Upload failed');
      fileInput.value = '';
      alert('Uploaded. Click “Rebuild Index” to include it.');
      await loadFiles();
      setSelectedDoc(data.saved_as || null);
    } catch (err) {
      alert('Upload error: ' + err.message);
    }
  });

  // ---------- QUICK MENU & UTILITIES ----------
  fab?.addEventListener('click', () => quickMenu?.show());
  menuMyPolicy?.addEventListener('click', () => {
    quickMenu?.hide();
    const m = bootstrap.Modal.getOrCreateInstance(docsModal);
    m.show();
    loadFiles();
  });

  checkClaimBtn?.addEventListener('click', () => {
    const num = (claimNumber?.value || '').trim();
    if (!claimStatusResult) return;
    if (!num) {
      claimStatusResult.textContent = 'Please enter a claim number.';
      return;
    }
    // Placeholder: wire to backend later
    claimStatusResult.textContent = `Claim ${num}: Received — Awaiting adjuster review. (This is a demo response)`;
  });

  calcRun?.addEventListener('click', () => {
    const dmg = parseFloat(calcDamage?.value || '0');
    const ded = parseFloat(calcDeductible?.value || '0');
    const lim = parseFloat(calcLimit?.value || 'NaN');
    if (isNaN(dmg) || isNaN(ded)) {
      if (calcResult) calcResult.innerHTML = '<span class="text-warning">Enter valid numbers for damage and deductible.</span>';
      return;
    }
    let payable = Math.max(0, dmg - ded);
    if (!isNaN(lim)) payable = Math.min(payable, lim);
    const youPay = Math.min(dmg, ded + payable);
    const insurerPays = Math.max(0, dmg - youPay);
    if (calcResult) {
      calcResult.innerHTML = `
        <div class="alert alert-dark border" role="alert" style="border-color:#27304a;">
          <div><strong>Out-of-pocket:</strong> $${youPay.toFixed(2)}</div>
          <div><strong>Insurer pays (est.):</strong> $${insurerPays.toFixed(2)}</div>
        </div>
      `;
    }
  });

  faqList?.addEventListener('click', (e) => {
    if (e.target && e.target.matches('.list-group-item')) {
      const q = e.target.textContent.trim();
      const target = document.body.classList.contains('chat-started') ? chatInput : heroInput;
      if (target) target.value = q;
      if (selectedDoc) {
        if (document.body.classList.contains('chat-started')) {
          nextAsk();
        } else {
          firstAsk();
        }
      } else {
        const m = bootstrap.Modal.getOrCreateInstance(docsModal);
        m.show();
        loadFiles();
      }
      const faqsModal = document.getElementById('faqsModal');
      if (faqsModal) bootstrap.Modal.getInstance(faqsModal)?.hide();
      quickMenu?.hide();
    }
  });

  supportSend?.addEventListener('click', (e) => {
    const name  = encodeURIComponent(supportName?.value || '');
    const email = encodeURIComponent(supportEmail?.value || '');
    const msg   = encodeURIComponent(supportMsg?.value || '');
    const subject = encodeURIComponent('Insurance Assistant Support');
    const body   = encodeURIComponent(`Name: ${name}\nEmail: ${email}\n\n${msg}`);
    e.currentTarget.href = `mailto:support@example.com?subject=${subject}&body=${body}`;
  });

  // ---------- GLOBAL ----------
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && aborter) aborter.abort();
  });

  // ---------- INIT ----------
  setSelectedDoc(null);
});
