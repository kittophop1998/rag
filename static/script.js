/**
 * Company RAG — Frontend
 *
 * Architecture:
 *   API      → All HTTP / SSE calls (isolated; easy to swap backend)
 *   Storage  → Chat session persistence via localStorage
 *   md       → Lightweight inline Markdown renderer
 *   UI       → Pure DOM helpers (no business logic)
 *   App      → Orchestrates everything; event binding
 */
'use strict';

/* ================================================================
   CONFIGURATION
   ================================================================ */
const CFG = {
  STREAM_URL:    '/api/chat/stream',
  REINDEX_URL:   '/api/reindex',
  DOCS_URL:      '/api/documents',
  UPLOAD_URL:    '/api/upload',
  STORAGE_KEY:   'rag_sessions_v2',
  MAX_SESSIONS:  60,
  TITLE_MAX_LEN: 46,
};

/* ================================================================
   MARKDOWN RENDERER
   Handles the subset of Markdown that LLMs typically produce.
   ================================================================ */
function md(raw) {
  if (!raw) return '';

  // Split on fenced code blocks first so we don't mangle their content.
  const parts = raw.split(/(```[\w]*\n?[\s\S]*?```)/g);

  const rendered = parts.map((part, idx) => {
    // Code block
    if (idx % 2 === 1) {
      const m = part.match(/```([\w]*)\n?([\s\S]*?)```/);
      if (!m) return `<pre><code>${esc(part)}</code></pre>`;
      const lang = m[1] ? ` class="language-${esc(m[1])}"` : '';
      return `<pre><code${lang}>${esc(m[2].trim())}</code></pre>`;
    }

    let t = esc(part);

    // Inline code  (before bold/italic so backticks aren't processed further)
    t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');

    // Italic
    t = t.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    t = t.replace(/_([^_\n]+)_/g, '<em>$1</em>');

    // Headers
    t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    t = t.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered list lines → wrap in <ul>
    t = t.replace(/((?:^[ \t]*[-*] .+\n?)+)/gm, match => {
      const items = match.replace(/^[ \t]*[-*] (.+)$/gm, '<li>$1</li>');
      return `<ul>${items}</ul>`;
    });

    // Ordered list lines → wrap in <ol>
    t = t.replace(/((?:^[ \t]*\d+\. .+\n?)+)/gm, match => {
      const items = match.replace(/^[ \t]*\d+\. (.+)$/gm, '<li>$1</li>');
      return `<ol>${items}</ol>`;
    });

    // Paragraphs
    t = t.split(/\n{2,}/).map(p => {
      const trimmed = p.trim();
      if (!trimmed) return '';
      if (/^<(h[1-3]|ul|ol|pre|li)/.test(trimmed)) return trimmed;
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return t;
  });

  return rendered.join('');
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ================================================================
   API CLIENT
   All network calls are here. Swap endpoints or add auth in one place.
   ================================================================ */
const API = {

  /**
   * Stream a chat answer via SSE.
   * Yields objects: { type: 'token'|'sources'|'error', content: any }
   */
  async *stream(question) {
    const url = `${CFG.STREAM_URL}?q=${encodeURIComponent(question)}`;
    let resp;
    try {
      resp = await fetch(url, { headers: { Accept: 'text/event-stream' } });
    } catch (err) {
      yield { type: 'error', content: `ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: ${err.message}` };
      return;
    }
    if (!resp.ok) {
      yield { type: 'error', content: `Server error ${resp.status}` };
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();           // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') return;
        try { yield JSON.parse(data); } catch { /* skip malformed */ }
      }
    }
  },

  async reindex() {
    const r = await fetch(CFG.REINDEX_URL, { method: 'POST' });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async documents() {
    const r = await fetch(CFG.DOCS_URL);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async upload(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(CFG.UPLOAD_URL, { method: 'POST', body: fd });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },
};

/* ================================================================
   STORAGE  –  Chat session persistence
   Session schema: { id, title, createdAt, messages: [] }
   Message schema: { role: 'user'|'bot', content, sources?, ts }
   ================================================================ */
const Store = {
  _sessions: null,

  all() {
    if (!this._sessions) {
      try {
        this._sessions = JSON.parse(localStorage.getItem(CFG.STORAGE_KEY) || '[]');
      } catch {
        this._sessions = [];
      }
    }
    return this._sessions;
  },

  save() {
    try {
      localStorage.setItem(CFG.STORAGE_KEY, JSON.stringify(this._sessions));
    } catch { /* storage full – ignore */ }
  },

  create(firstMessage) {
    const session = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      title: firstMessage.slice(0, CFG.TITLE_MAX_LEN) + (firstMessage.length > CFG.TITLE_MAX_LEN ? '…' : ''),
      createdAt: Date.now(),
      messages: [],
    };
    this.all().unshift(session);
    if (this._sessions.length > CFG.MAX_SESSIONS) this._sessions.pop();
    this.save();
    return session;
  },

  get(id) {
    return this.all().find(s => s.id === id) || null;
  },

  addMessage(sessionId, msg) {
    const s = this.get(sessionId);
    if (!s) return;
    s.messages.push({ ...msg, ts: Date.now() });
    this.save();
  },

  updateBotMessage(sessionId, content, sources) {
    const s = this.get(sessionId);
    if (!s) return;
    const last = [...s.messages].reverse().find(m => m.role === 'bot');
    if (last) { last.content = content; last.sources = sources; }
    this.save();
  },

  delete(id) {
    this._sessions = this.all().filter(s => s.id !== id);
    this.save();
  },
};

/* ================================================================
   UI HELPERS
   Pure DOM operations — no business logic here.
   ================================================================ */
const $ = id => document.getElementById(id);

/** Show a toast notification. */
function toast(msg, type = 'info', durationMs = 3500) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warn: '⚠️' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span class="toast-text">${esc(msg)}</span>`;
  const container = $('toastContainer');
  container.appendChild(el);
  setTimeout(() => {
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, durationMs);
}

/** Format bytes to human-readable string. */
function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

/** Group sessions by relative date label. */
function groupByDate(sessions) {
  const now = Date.now();
  const ONE_DAY = 86_400_000;
  const groups = {};
  for (const s of sessions) {
    const diff = now - s.createdAt;
    let label;
    if (diff < ONE_DAY)           label = 'วันนี้';
    else if (diff < 2 * ONE_DAY)  label = 'เมื่อวาน';
    else if (diff < 7 * ONE_DAY)  label = 'สัปดาห์นี้';
    else                           label = 'เก่ากว่า';
    (groups[label] ??= []).push(s);
  }
  return groups;
}

/* ================================================================
   APP
   ================================================================ */
class App {
  constructor() {
    this.currentSessionId = null;
    this.isStreaming = false;
  }

  /** Boot: bind events, render history, load docs. */
  init() {
    this._bindSidebar();
    this._bindComposer();
    this._bindSettings();
    this._bindKnowledgeBase();
    this._bindSuggestions();
    this._renderHistory();
    this._loadDocs();
    $('chatInput').focus();
  }

  /* ── Sidebar ────────────────────────────────────────────────── */
  _bindSidebar() {
    $('sidebarCollapseBtn').addEventListener('click', () => {
      document.getElementById('app').classList.toggle('sidebar-collapsed');
    });

    // Mobile toggle
    const mobileBtn = $('mobileSidebarBtn');
    mobileBtn.addEventListener('click', () => {
      $('sidebar').classList.toggle('mobile-open');
    });

    // Close sidebar on backdrop click (mobile)
    document.addEventListener('click', e => {
      const sidebar = $('sidebar');
      if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
        if (!sidebar.contains(e.target) && e.target !== mobileBtn) {
          sidebar.classList.remove('mobile-open');
        }
      }
    });

    $('newChatBtn').addEventListener('click', () => {
      this._startNewChat();
      $('sidebar').classList.remove('mobile-open');
    });
  }

  /* ── Chat History ───────────────────────────────────────────── */
  _renderHistory() {
    const container = $('chatHistory');
    const sessions = Store.all();

    if (!sessions.length) {
      container.innerHTML = '<p class="history-empty">ยังไม่มีประวัติการสนทนา</p>';
      return;
    }

    const groups = groupByDate(sessions);
    const labels = ['วันนี้', 'เมื่อวาน', 'สัปดาห์นี้', 'เก่ากว่า'];
    let html = '';

    for (const label of labels) {
      if (!groups[label]?.length) continue;
      html += `<div class="history-label">${label}</div>`;
      for (const s of groups[label]) {
        const active = s.id === this.currentSessionId ? ' active' : '';
        html += `
          <div class="history-item${active}" data-id="${s.id}" role="button" tabindex="0">
            <span class="history-item-text">${esc(s.title)}</span>
            <button class="history-item-del" data-del="${s.id}" title="ลบ" aria-label="ลบการสนทนา">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/>
              </svg>
            </button>
          </div>`;
      }
    }

    container.innerHTML = html;

    // Event delegation for clicks
    container.addEventListener('click', e => {
      const delBtn = e.target.closest('[data-del]');
      if (delBtn) {
        e.stopPropagation();
        this._deleteSession(delBtn.dataset.del);
        return;
      }
      const item = e.target.closest('[data-id]');
      if (item) this._loadSession(item.dataset.id);
    });

    // Keyboard support
    container.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        const item = e.target.closest('[data-id]');
        if (item) this._loadSession(item.dataset.id);
      }
    });
  }

  _startNewChat() {
    this.currentSessionId = null;
    this._clearMessages();
    this._showEmptyState();
    this._renderHistory();
    $('chatInput').focus();
  }

  _loadSession(id) {
    const session = Store.get(id);
    if (!session) return;
    this.currentSessionId = id;
    this._clearMessages();
    this._hideEmptyState();

    for (const msg of session.messages) {
      if (msg.role === 'user') {
        this._appendUserBubble(msg.content);
      } else {
        const { bubble } = this._createBotBubble();
        bubble.innerHTML = md(msg.content);
        if (msg.sources?.length) this._appendSources(bubble.parentElement, msg.sources);
      }
    }
    this._renderHistory();
    this._scrollToBottom();
    $('sidebar').classList.remove('mobile-open');
  }

  _deleteSession(id) {
    if (id === this.currentSessionId) this._startNewChat();
    Store.delete(id);
    this._renderHistory();
  }

  /* ── Composer ───────────────────────────────────────────────── */
  _bindComposer() {
    const input  = $('chatInput');
    const sendBtn = $('sendBtn');
    const form   = $('composer');

    // Enable/disable send button
    input.addEventListener('input', () => {
      sendBtn.disabled = !input.value.trim() || this.isStreaming;
      // Auto-resize
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 180) + 'px';
    });

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._send();
      }
    });

    form.addEventListener('submit', e => {
      e.preventDefault();
      this._send();
    });
  }

  async _send() {
    if (this.isStreaming) return;
    const input = $('chatInput');
    const question = input.value.trim();
    if (!question) return;

    // Clear input
    input.value = '';
    input.style.height = 'auto';
    $('sendBtn').disabled = true;

    // Ensure we have a session
    if (!this.currentSessionId) {
      const session = Store.create(question);
      this.currentSessionId = session.id;
      this._renderHistory();
    }

    // Hide empty state, show message area
    this._hideEmptyState();

    // Append user message
    this._appendUserBubble(question);
    Store.addMessage(this.currentSessionId, { role: 'user', content: question });

    // Create bot bubble with typing indicator
    const { msgEl, bubble, setContent, finalize } = this._createBotBubble(true);
    this._scrollToBottom();

    // Stream
    this.isStreaming = true;
    let fullText = '';
    let sources = [];

    try {
      for await (const event of API.stream(question)) {
        if (event.type === 'token') {
          fullText += event.content;
          setContent(fullText, true /* streaming */);
          this._scrollToBottom();
        } else if (event.type === 'sources') {
          sources = event.content;
        } else if (event.type === 'error') {
          fullText = event.content;
          setContent(fullText, false);
        }
      }
    } catch (err) {
      fullText = `ขออภัยครับ เกิดข้อผิดพลาด: ${err.message}`;
      setContent(fullText, false);
    } finally {
      finalize();
      if (sources.length) this._appendSources(msgEl, sources);
      Store.addMessage(this.currentSessionId, { role: 'bot', content: fullText, sources });
      this.isStreaming = false;
      $('sendBtn').disabled = false;
      this._scrollToBottom();
      $('chatInput').focus();
    }
  }

  /* ── Chat Area DOM helpers ──────────────────────────────────── */
  _showEmptyState()  {
    $('emptyState').style.display = '';
    $('messages').style.display  = 'none';
  }

  _hideEmptyState() {
    $('emptyState').style.display = 'none';
    $('messages').style.display  = '';
  }

  _clearMessages() {
    $('messages').innerHTML = '';
  }

  _scrollToBottom() {
    const el = $('messages');
    el.scrollTop = el.scrollHeight;
  }

  _appendUserBubble(text) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message user';
    msgEl.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-body">
        <div class="bubble">${esc(text)}</div>
      </div>`;
    $('messages').appendChild(msgEl);
    return msgEl;
  }

  /**
   * Creates a bot message bubble.
   * Returns helpers: setContent(text, streaming), finalize(), and DOM refs.
   */
  _createBotBubble(showTyping = false) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message bot';

    const avatarSvg = `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
        <line x1="9" y1="9" x2="9.01" y2="9" stroke-width="3"/>
        <line x1="15" y1="9" x2="15.01" y2="9" stroke-width="3"/>
      </svg>`;

    msgEl.innerHTML = `
      <div class="msg-avatar">${avatarSvg}</div>
      <div class="msg-body">
        <div class="bubble">${showTyping ? '<div class="typing-dots"><span></span><span></span><span></span></div>' : ''}</div>
      </div>`;

    $('messages').appendChild(msgEl);
    const bubble = msgEl.querySelector('.bubble');

    let cursor = null;

    function setContent(text, streaming = false) {
      // Remove cursor before re-rendering
      if (cursor) cursor.remove();
      bubble.innerHTML = md(text);
      if (streaming) {
        cursor = document.createElement('span');
        cursor.className = 'stream-cursor';
        bubble.appendChild(cursor);
      }
    }

    function finalize() {
      if (cursor) { cursor.remove(); cursor = null; }
    }

    return { msgEl, bubble, setContent, finalize };
  }

  /**
   * Append sources as clickable chips below a bot message body.
   */
  _appendSources(msgEl, sources) {
    const body = msgEl.querySelector('.msg-body');
    if (!body) return;

    const bar = document.createElement('div');
    bar.className = 'sources-bar';

    const label = document.createElement('span');
    label.className = 'sources-label';
    label.textContent = '📎 อ้างอิง:';
    bar.appendChild(label);

    // Preview container (shared)
    const previewContainer = document.createElement('div');

    sources.forEach((src, i) => {
      const page = typeof src.page === 'number' ? ` หน้า ${src.page + 1}` : '';
      const chipText = `${src.source}${page}`;

      const chip = document.createElement('button');
      chip.className = 'source-chip';
      chip.title = chipText;
      chip.innerHTML = `📄 ${esc(chipText)}`;

      const preview = document.createElement('div');
      preview.className = 'source-preview';
      preview.hidden = true;
      preview.innerHTML = `
        <div class="source-preview-header">📄 ${esc(chipText)}</div>
        <div class="source-preview-text">${esc(src.snippet || '(ไม่มีข้อความตัวอย่าง)')}</div>`;

      chip.addEventListener('click', () => {
        const isOpen = !preview.hidden;
        // Close all other previews and chips
        previewContainer.querySelectorAll('.source-preview').forEach(p => p.hidden = true);
        bar.querySelectorAll('.source-chip').forEach(c => c.classList.remove('active'));
        // Toggle this one
        preview.hidden = isOpen;
        chip.classList.toggle('active', !isOpen);
      });

      bar.appendChild(chip);
      previewContainer.appendChild(preview);
    });

    const bubble = body.querySelector('.bubble');
    if (bubble) {
      bubble.appendChild(bar);
      bubble.appendChild(previewContainer);
    }
  }

  /* ── Suggestions (empty state) ──────────────────────────────── */
  _bindSuggestions() {
    $('suggestionGrid').addEventListener('click', e => {
      const card = e.target.closest('.suggestion-card');
      if (!card) return;
      const q = card.dataset.q;
      if (!q) return;
      $('chatInput').value = q;
      $('chatInput').dispatchEvent(new Event('input'));
      this._send();
    });
  }

  /* ── Knowledge Base ─────────────────────────────────────────── */
  _bindKnowledgeBase() {
    // Accordion toggle
    $('kbToggleBtn').addEventListener('click', () => {
      const panel = $('kbPanel');
      const isOpen = panel.classList.toggle('open');
      $('kbToggleBtn').setAttribute('aria-expanded', isOpen);
      panel.setAttribute('aria-hidden', !isOpen);
      if (isOpen) this._loadDocs();
    });

    // Reindex
    $('reindexBtn').addEventListener('click', async () => {
      const btn = $('reindexBtn');
      btn.disabled = true;
      btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 1s linear infinite"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> กำลัง Rebuild...`;
      try {
        await API.reindex();
        toast('สร้าง Index สำเร็จแล้ว — พร้อมตอบคำถามจากเอกสารใหม่!', 'success');
        await this._loadDocs();
      } catch (err) {
        toast(`Rebuild ไม่สำเร็จ: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Rebuild Index`;
      }
    });

    // File upload
    $('uploadInput').addEventListener('change', async e => {
      const files = Array.from(e.target.files || []);
      e.target.value = '';
      if (!files.length) return;
      await this._uploadFiles(files);
    });
  }

  async _loadDocs() {
    const list = $('docsList');
    try {
      const data = await API.documents();
      if (!data.documents.length) {
        list.innerHTML = '<p class="docs-empty">ยังไม่มีไฟล์ PDF<br>กดอัปโหลดเพื่อเพิ่มเอกสาร</p>';
        return;
      }
      list.innerHTML = data.documents.map(doc => `
        <div class="doc-item" title="${esc(doc.name)} (${fmtBytes(doc.size)})">
          <span class="doc-item-name">📄 ${esc(doc.name)}</span>
          <span class="doc-item-badge ${doc.indexed ? 'indexed' : 'pending'}">${doc.indexed ? 'Indexed' : 'Pending'}</span>
        </div>`).join('');
    } catch {
      list.innerHTML = '<p class="docs-empty">โหลดรายการไม่สำเร็จ</p>';
    }
  }

  async _uploadFiles(files) {
    for (const file of files) {
      toast(`กำลังอัปโหลด ${file.name}...`, 'info', 8000);
      try {
        await API.upload(file);
        toast(`อัปโหลด ${file.name} สำเร็จ`, 'success');
      } catch (err) {
        toast(`อัปโหลด ${file.name} ไม่สำเร็จ: ${err.message}`, 'error');
      }
    }
    await this._loadDocs();
  }

  /* ── Settings Modal ─────────────────────────────────────────── */
  _bindSettings() {
    const modal   = $('settingsModal');
    const backdrop = $('modalBackdrop');
    const closeBtn = $('modalCloseBtn');

    const open  = () => { modal.classList.add('open'); modal.removeAttribute('aria-hidden'); };
    const close = () => { modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); };

    $('settingsBtn').addEventListener('click', () => {
      $('webhookUrl').textContent = `${window.location.origin}/webhook`;
      open();
    });

    backdrop.addEventListener('click', close);
    closeBtn.addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

    // Copy webhook URL
    $('copyWebhookBtn').addEventListener('click', () => {
      const url = $('webhookUrl').textContent;
      navigator.clipboard.writeText(url).then(() => {
        toast('คัดลอก Webhook URL แล้ว', 'success');
      }).catch(() => {
        toast('คัดลอกไม่สำเร็จ', 'error');
      });
    });
  }
}

/* ================================================================
   CSS injection for spin animation (used during reindex button)
   ================================================================ */
const _spinStyle = document.createElement('style');
_spinStyle.textContent = `@keyframes spin { to { transform: rotate(360deg); } }`;
document.head.appendChild(_spinStyle);

/* ================================================================
   BOOT
   ================================================================ */
document.addEventListener('DOMContentLoaded', () => new App().init());
