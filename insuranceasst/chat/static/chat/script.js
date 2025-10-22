const chat = document.getElementById('chat');
const inp = document.getElementById('message');
const send = document.getElementById('send');
const newChat = document.getElementById('newChat');
const suggestions = document.getElementById('suggestions');
const themeToggle = document.getElementById('themeToggle');

// ===== THEME =====
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeToggle.checked = (theme === 'dark');
  localStorage.setItem('theme', theme);
}
(function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  applyTheme(saved);
})();
themeToggle.addEventListener('change', () => {
  applyTheme(themeToggle.checked ? 'dark' : 'light');
});

// ===== CHAT HELPERS =====
function addBubble(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  chat.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function showSuggestions(show) {
  if (!suggestions) return;
  suggestions.style.display = show ? '' : 'none';
}

// ===== SEND MESSAGE =====
async function ask(q) {
  const question = q ?? inp.value.trim();
  if (!question) return;
  addBubble(question, 'user');
  showSuggestions(false);
  inp.value = '';
  send.disabled = true;

  try {
    const res = await fetch('/api/chat/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question })
    });
    const data = await res.json();
    addBubble(data.answer || ('Error: ' + (data.error || 'Unknown')), 'bot');
  } catch (e) {
    addBubble('Error: ' + e.message, 'bot');
  } finally {
    send.disabled = false;
  }
}

const docSelect = document.getElementById('docSelect');

async function loadIndexed() {
  try {
    const res = await fetch('/api/indexed/');
    const data = await res.json();
    docSelect.innerHTML = '<option value="">— All indexed documents —</option>';
    (data.docs || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.source;            // pass full source path back to server
      opt.textContent = `${d.name} (${d.count})`;
      docSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('indexed list error', e);
  }
}

async function ask() {
  const q = inp.value.trim();
  if (!q) return;
  add(q, 'user');
  inp.value = '';
  send.disabled = true;
  try {
    const payload = { message: q };
    const chosen = docSelect.value.trim();
    if (chosen) payload.source = chosen;   // filter to selected doc
    const res = await fetch('/api/chat/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    add(data.answer || ('Error: ' + (data.error || 'Unknown')), 'bot');
  } catch (e) {
    add('Error: ' + e.message, 'bot');
  } finally {
    send.disabled = false;
  }
}


send.addEventListener('click', () => ask());
inp.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

// ===== NEW CHAT =====
function resetChat() {
  chat.innerHTML = '';
  addBubble('New chat started. Ask about your policy (e.g., “Does my policy cover rental cars?”)', 'bot');
  showSuggestions(true);
}
newChat.addEventListener('click', resetChat);

// ===== SUGGESTION CHIPS =====
if (suggestions) {
  suggestions.addEventListener('click', (e) => {
    const btn = e.target.closest('.chip');
    if (!btn) return;
    const q = btn.getAttribute('data-q');
    if (q) ask(q);
  });
}

docSelect.addEventListener('change', () => {
  const opt = docSelect.options[docSelect.selectedIndex];
  const fname = opt ? opt.textContent : '';
  // Try to find the URL in the /api/files list we loaded:
  const rows = filesDiv.querySelectorAll('.file-item');
  // (Optional) you could map name->url on loadFiles to open automatically
});
