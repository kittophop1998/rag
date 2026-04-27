/**
 * Company RAG — Frontend
 *
 * Architecture:
 *   Auth     → Login / logout / token management
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
  LOGIN_URL:     '/api/auth/login',
  LOGOUT_URL:    '/api/auth/logout',
  ME_URL:        '/api/auth/me',
  DB_URL:        '/api/settings/databases',
  USERS_URL:     '/api/users',
  STORAGE_KEY:   'rag_sessions_v2',
  TOKEN_KEY:     'rag_auth_token',
  ROLE_KEY:      'rag_auth_role',
  USER_KEY:      'rag_auth_user',
  MAX_SESSIONS:  60,
  TITLE_MAX_LEN: 46,
};

/* ================================================================
   AUTH — token + role management
   ================================================================ */
const Auth = {
  getToken() {
    return localStorage.getItem(CFG.TOKEN_KEY) || '';
  },

  setToken(token) {
    localStorage.setItem(CFG.TOKEN_KEY, token);
  },

  clearToken() {
    localStorage.removeItem(CFG.TOKEN_KEY);
    localStorage.removeItem(CFG.ROLE_KEY);
    localStorage.removeItem(CFG.USER_KEY);
  },

  getRole() {
    return localStorage.getItem(CFG.ROLE_KEY) || '';
  },

  getUsername() {
    return localStorage.getItem(CFG.USER_KEY) || '';
  },

  isAdmin() {
    return this.getRole() === 'admin';
  },

  isLoggedIn() {
    return !!this.getToken();
  },

  /** Returns the Authorization header object for fetch calls. */
  headers() {
    const t = this.getToken();
    return t ? { Authorization: `Bearer ${t}` } : {};
  },

  async login(username, password) {
    const r = await fetch(CFG.LOGIN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    this.setToken(json.token);
    if (json.role)     localStorage.setItem(CFG.ROLE_KEY, json.role);
    if (json.username) localStorage.setItem(CFG.USER_KEY, json.username);
    return json;
  },

  async logout() {
    try {
      await fetch(CFG.LOGOUT_URL, {
        method: 'POST',
        headers: { ...Auth.headers(), 'Content-Type': 'application/json' },
      });
    } catch { /* ignore network error on logout */ }
    this.clearToken();
  },

  /** Verify token and refresh role/username from server. */
  async verify() {
    if (!this.isLoggedIn()) return false;
    try {
      const r = await fetch(CFG.ME_URL, { headers: Auth.headers() });
      if (!r.ok) return false;
      const json = await r.json().catch(() => ({}));
      if (json.role)     localStorage.setItem(CFG.ROLE_KEY, json.role);
      if (json.username) localStorage.setItem(CFG.USER_KEY, json.username);
      return true;
    } catch {
      return false;
    }
  },
};

/* ================================================================
   MARKDOWN RENDERER
   Handles the subset of Markdown that LLMs typically produce.
   ================================================================ */
function md(raw) {
  if (!raw) return '';

  const parts = raw.split(/(```[\w]*\n?[\s\S]*?```)/g);

  const rendered = parts.map((part, idx) => {
    if (idx % 2 === 1) {
      const m = part.match(/```([\w]*)\n?([\s\S]*?)```/);
      if (!m) return `<pre><code>${esc(part)}</code></pre>`;
      const lang = m[1] ? ` class="language-${esc(m[1])}"` : '';
      return `<pre><code${lang}>${esc(m[2].trim())}</code></pre>`;
    }

    let t = esc(part);
    t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    t = t.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    t = t.replace(/_([^_\n]+)_/g, '<em>$1</em>');
    t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    t = t.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    t = t.replace(/((?:^[ \t]*[-*] .+\n?)+)/gm, match => {
      const items = match.replace(/^[ \t]*[-*] (.+)$/gm, '<li>$1</li>');
      return `<ul>${items}</ul>`;
    });

    t = t.replace(/((?:^[ \t]*\d+\. .+\n?)+)/gm, match => {
      const items = match.replace(/^[ \t]*\d+\. (.+)$/gm, '<li>$1</li>');
      return `<ol>${items}</ol>`;
    });

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

  async *stream(question) {
    const url = `${CFG.STREAM_URL}?q=${encodeURIComponent(question)}`;
    let resp;
    try {
      resp = await fetch(url, { headers: { Accept: 'text/event-stream', ...Auth.headers() } });
    } catch (err) {
      yield { type: 'error', content: `ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: ${err.message}` };
      return;
    }
    if (resp.status === 401) {
      Auth.clearToken();
      showLogin();
      yield { type: 'error', content: 'Session หมดอายุ กรุณาเข้าสู่ระบบอีกครั้ง' };
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
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') return;
        try { yield JSON.parse(data); } catch { /* skip malformed */ }
      }
    }
  },

  async reindex() {
    const r = await fetch(CFG.REINDEX_URL, { method: 'POST', headers: Auth.headers() });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async documents() {
    const r = await fetch(CFG.DOCS_URL, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async upload(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(CFG.UPLOAD_URL, { method: 'POST', body: fd, headers: Auth.headers() });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  // ── Database settings ───────────────────────────────────────────
  async listDatabases() {
    const r = await fetch(CFG.DB_URL, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async addDatabase(data) {
    const r = await fetch(CFG.DB_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async updateDatabase(id, data) {
    const r = await fetch(`${CFG.DB_URL}/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async deleteDatabase(id) {
    const r = await fetch(`${CFG.DB_URL}/${id}`, { method: 'DELETE', headers: Auth.headers() });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  // ── User management ─────────────────────────────────────────────
  async listUsers() {
    const r = await fetch(CFG.USERS_URL, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async createUser(data) {
    const r = await fetch(CFG.USERS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async updateUser(id, data) {
    const r = await fetch(`${CFG.USERS_URL}/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async deleteUser(id) {
    const r = await fetch(`${CFG.USERS_URL}/${id}`, { method: 'DELETE', headers: Auth.headers() });
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

function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

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
   LOGIN PAGE HELPERS
   ================================================================ */
function showLogin() {
  $('loginPage').classList.add('active');
  $('loginPage').removeAttribute('aria-hidden');
  $('app').setAttribute('aria-hidden', 'true');
  $('app').style.display = 'none';
  setTimeout(() => $('loginUsername').focus(), 100);
}

function showApp() {
  $('loginPage').classList.remove('active');
  $('loginPage').setAttribute('aria-hidden', 'true');
  $('app').removeAttribute('aria-hidden');
  $('app').style.display = '';
}

/* DB type icons / labels */
const DB_TYPES = {
  mysql:      { label: 'MySQL',         icon: '🐬' },
  postgresql: { label: 'PostgreSQL',    icon: '🐘' },
  mssql:      { label: 'MS SQL Server', icon: '🪟' },
  mongodb:    { label: 'MongoDB',       icon: '🍃' },
  redis:      { label: 'Redis',         icon: '🔴' },
  rest:       { label: 'REST API',      icon: '🌐' },
  other:      { label: 'อื่นๆ',          icon: '🗄️' },
};

/* ================================================================
   APP
   ================================================================ */
class App {
  constructor() {
    this.currentSessionId = null;
    this.isStreaming = false;
    this._dbEditingId   = null;
    this._userEditingId = null;
  }

  /** Boot: check auth first, then bind events. */
  async init() {
    const valid = await Auth.verify();
    if (!valid) {
      Auth.clearToken();
      showLogin();
      this._bindLogin();
      return;
    }
    this._launch();
  }

  _launch() {
    showApp();
    this._applyRoleVisibility();
    this._updateUserDisplay();
    this._bindSidebar();
    this._bindComposer();
    this._bindSettings();
    this._bindKnowledgeBase();
    this._bindSuggestions();
    this._renderHistory();
    if (Auth.isAdmin()) this._loadDocs();
    $('chatInput').focus();
  }

  /** Show or hide elements marked admin-only based on current role. */
  _applyRoleVisibility() {
    const isAdmin = Auth.isAdmin();
    document.querySelectorAll('.admin-only').forEach(el => {
      el.style.display = isAdmin ? '' : 'none';
    });
  }

  /** Populate the sidebar footer with username and role badge. */
  _updateUserDisplay() {
    const username = Auth.getUsername();
    const role     = Auth.getRole();
    const avatar   = $('sidebarUserAvatar');
    const nameEl   = $('sidebarUserName');
    const roleEl   = $('sidebarUserRole');

    if (avatar)  avatar.textContent  = (username[0] || '?').toUpperCase();
    if (nameEl)  nameEl.textContent  = username || '—';
    if (roleEl) {
      roleEl.textContent  = role === 'admin' ? 'Admin' : 'User';
      roleEl.className    = `sidebar-user-role role-badge role-${role}`;
    }
  }

  /* ── Login ──────────────────────────────────────────────────── */
  _bindLogin() {
    const form    = $('loginForm');
    const errEl   = $('loginError');
    const pwInput = $('loginPassword');
    const togglePwBtn = $('togglePasswordBtn');

    togglePwBtn.addEventListener('click', () => {
      const isPassword = pwInput.type === 'password';
      pwInput.type = isPassword ? 'text' : 'password';
      $('eyeIcon').innerHTML = isPassword
        ? `<line x1="1" y1="1" x2="23" y2="23"/><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>`
        : `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>`;
    });

    form.addEventListener('submit', async e => {
      e.preventDefault();
      errEl.textContent = '';
      const username = $('loginUsername').value.trim();
      const password = $('loginPassword').value;
      if (!username || !password) {
        errEl.textContent = 'กรุณากรอกชื่อผู้ใช้และรหัสผ่าน';
        return;
      }

      const btn = $('loginBtn');
      btn.disabled = true;
      $('loginBtnText').textContent = 'กำลังเข้าสู่ระบบ...';
      $('loginSpinner').classList.remove('hidden');

      try {
        await Auth.login(username, password);
        this._launch();
        $('loginPassword').value = '';
      } catch (err) {
        errEl.textContent = err.message || 'เข้าสู่ระบบไม่สำเร็จ';
        $('loginPassword').value = '';
        $('loginPassword').focus();
      } finally {
        btn.disabled = false;
        $('loginBtnText').textContent = 'เข้าสู่ระบบ';
        $('loginSpinner').classList.add('hidden');
      }
    });

    // Enter on username → focus password
    $('loginUsername').addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); $('loginPassword').focus(); }
    });
  }

  /* ── Sidebar ────────────────────────────────────────────────── */
  _bindSidebar() {
    $('sidebarCollapseBtn').addEventListener('click', () => {
      document.getElementById('app').classList.toggle('sidebar-collapsed');
    });

    const mobileBtn = $('mobileSidebarBtn');
    mobileBtn.addEventListener('click', () => {
      $('sidebar').classList.toggle('mobile-open');
    });

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

    $('logoutBtn').addEventListener('click', async () => {
      await Auth.logout();
      showLogin();
      this._bindLogin();
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

    container.addEventListener('click', e => {
      const delBtn = e.target.closest('[data-del]');
      if (delBtn) { e.stopPropagation(); this._deleteSession(delBtn.dataset.del); return; }
      const item = e.target.closest('[data-id]');
      if (item) this._loadSession(item.dataset.id);
    });

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
    const input   = $('chatInput');
    const sendBtn = $('sendBtn');
    const form    = $('composer');

    input.addEventListener('input', () => {
      sendBtn.disabled = !input.value.trim() || this.isStreaming;
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 180) + 'px';
    });

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._send(); }
    });

    form.addEventListener('submit', e => { e.preventDefault(); this._send(); });
  }

  async _send() {
    if (this.isStreaming) return;
    const input = $('chatInput');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    input.style.height = 'auto';
    $('sendBtn').disabled = true;

    if (!this.currentSessionId) {
      const session = Store.create(question);
      this.currentSessionId = session.id;
      this._renderHistory();
    }

    this._hideEmptyState();
    this._appendUserBubble(question);
    Store.addMessage(this.currentSessionId, { role: 'user', content: question });

    const { msgEl, bubble, setContent, finalize } = this._createBotBubble(true);
    this._scrollToBottom();

    this.isStreaming = true;
    let fullText = '';
    let sources = [];

    try {
      for await (const event of API.stream(question)) {
        if (event.type === 'token') {
          fullText += event.content;
          setContent(fullText, true);
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
  _showEmptyState()  { $('emptyState').style.display = ''; $('messages').style.display = 'none'; }
  _hideEmptyState()  { $('emptyState').style.display = 'none'; $('messages').style.display = ''; }
  _clearMessages()   { $('messages').innerHTML = ''; }
  _scrollToBottom()  { const el = $('messages'); el.scrollTop = el.scrollHeight; }

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

  _appendSources(msgEl, sources) {
    const body = msgEl.querySelector('.msg-body');
    if (!body) return;

    const bar = document.createElement('div');
    bar.className = 'sources-bar';

    const label = document.createElement('span');
    label.className = 'sources-label';
    label.textContent = '📎 อ้างอิง:';
    bar.appendChild(label);

    const previewContainer = document.createElement('div');

    sources.forEach(src => {
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
        previewContainer.querySelectorAll('.source-preview').forEach(p => p.hidden = true);
        bar.querySelectorAll('.source-chip').forEach(c => c.classList.remove('active'));
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
    $('kbToggleBtn').addEventListener('click', () => {
      const panel = $('kbPanel');
      const isOpen = panel.classList.toggle('open');
      $('kbToggleBtn').setAttribute('aria-expanded', isOpen);
      panel.setAttribute('aria-hidden', !isOpen);
      if (isOpen) this._loadDocs();
    });

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
    const modal    = $('settingsModal');
    const backdrop = $('modalBackdrop');
    const closeBtn = $('modalCloseBtn');

    const open = () => {
      modal.classList.add('open');
      modal.removeAttribute('aria-hidden');
      this._loadDatabases();
    };
    const close = () => {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      this._hideDbForm();
    };

    $('settingsBtn').addEventListener('click', () => {
      $('webhookUrl').textContent = `${window.location.origin}/webhook`;
      open();
    });

    backdrop.addEventListener('click', close);
    closeBtn.addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

    $('copyWebhookBtn').addEventListener('click', () => {
      const url = $('webhookUrl').textContent;
      navigator.clipboard.writeText(url).then(() => {
        toast('คัดลอก Webhook URL แล้ว', 'success');
      }).catch(() => {
        toast('คัดลอกไม่สำเร็จ', 'error');
      });
    });

    // Tab switching
    document.querySelectorAll('.settings-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.settings-tab-panel').forEach(p => {
          p.classList.remove('active');
          p.setAttribute('aria-hidden', 'true');
        });
        tab.classList.add('active');
        const panel = $(`tab${tab.dataset.tab.charAt(0).toUpperCase() + tab.dataset.tab.slice(1)}`);
        if (panel) {
          panel.classList.add('active');
          panel.removeAttribute('aria-hidden');
        }
        if (tab.dataset.tab === 'databases') this._loadDatabases();
        if (tab.dataset.tab === 'users')     this._loadUsers();
      });
    });

    // DB: add / edit
    $('addDbBtn').addEventListener('click', () => this._showDbForm());
    $('dbFormCancelBtn').addEventListener('click', () => this._hideDbForm());
    $('dbFormSaveBtn').addEventListener('click', () => this._saveDatabase());

    // Users: add / edit
    $('addUserBtn').addEventListener('click', () => this._showUserForm());
    $('userFormCancelBtn').addEventListener('click', () => this._hideUserForm());
    $('userFormSaveBtn').addEventListener('click', () => this._saveUser());
  }

  /* ── Database Management ────────────────────────────────────── */
  async _loadDatabases() {
    const list = $('dbList');
    try {
      const data = await API.listDatabases();
      const dbs = data.databases || [];

      // Update tab badge
      const badge = $('dbCountBadge');
      if (dbs.length > 0) {
        badge.textContent = dbs.length;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }

      if (!dbs.length) {
        list.innerHTML = `
          <div class="db-list-empty">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <ellipse cx="12" cy="5" rx="9" ry="3"/>
              <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
            <p>ยังไม่มีฐานข้อมูล<br>กด "เพิ่มฐานข้อมูล" เพื่อเริ่มต้น</p>
          </div>`;
        return;
      }

      list.innerHTML = dbs.map(db => {
        const typeInfo = DB_TYPES[db.db_type] || DB_TYPES.other;
        const maskedUrl = this._maskUrl(db.url);
        return `
          <div class="db-item" data-id="${db.id}">
            <div class="db-item-left">
              <span class="db-item-icon">${typeInfo.icon}</span>
              <div class="db-item-info">
                <div class="db-item-name">${esc(db.name)}</div>
                <div class="db-item-meta">
                  <span class="db-type-badge">${esc(typeInfo.label)}</span>
                  <span class="db-item-url" title="${esc(db.url)}">${esc(maskedUrl)}</span>
                </div>
                ${db.description ? `<div class="db-item-desc">${esc(db.description)}</div>` : ''}
              </div>
            </div>
            <div class="db-item-actions">
              <label class="db-toggle" title="${db.enabled ? 'ปิดใช้งาน' : 'เปิดใช้งาน'}">
                <input type="checkbox" class="db-toggle-input" data-db-id="${db.id}" ${db.enabled ? 'checked' : ''} />
                <span class="db-toggle-track"></span>
              </label>
              <button class="db-action-btn db-edit-btn" data-edit="${db.id}" title="แก้ไข">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
              </button>
              <button class="db-action-btn db-delete-btn" data-delete="${db.id}" title="ลบ">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                  <path d="M10 11v6M14 11v6"/>
                  <path d="M9 6V4h6v2"/>
                </svg>
              </button>
            </div>
          </div>`;
      }).join('');

      // Bind events on list items
      list.querySelectorAll('.db-toggle-input').forEach(chk => {
        chk.addEventListener('change', async () => {
          try {
            await API.updateDatabase(chk.dataset.dbId, { enabled: chk.checked });
            toast(chk.checked ? 'เปิดใช้งานแล้ว' : 'ปิดใช้งานแล้ว', 'success', 2000);
          } catch (err) {
            toast(`ไม่สำเร็จ: ${err.message}`, 'error');
            chk.checked = !chk.checked;
          }
        });
      });

      list.querySelectorAll('[data-edit]').forEach(btn => {
        btn.addEventListener('click', () => {
          const db = dbs.find(d => d.id === btn.dataset.edit);
          if (db) this._showDbForm(db);
        });
      });

      list.querySelectorAll('[data-delete]').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (!confirm('ต้องการลบฐานข้อมูลนี้ใช่หรือไม่?')) return;
          try {
            await API.deleteDatabase(btn.dataset.delete);
            toast('ลบฐานข้อมูลแล้ว', 'success');
            await this._loadDatabases();
          } catch (err) {
            toast(`ลบไม่สำเร็จ: ${err.message}`, 'error');
          }
        });
      });

    } catch (err) {
      list.innerHTML = `<p class="docs-empty">โหลดข้อมูลไม่สำเร็จ: ${esc(err.message)}</p>`;
    }
  }

  _maskUrl(url) {
    try {
      const u = new URL(url);
      if (u.password) u.password = '••••••';
      return u.toString();
    } catch {
      // Not a URL (e.g. connection string) — redact password-like parts
      return url.replace(/:[^@:/]+@/, ':••••••@');
    }
  }

  _showDbForm(db = null) {
    const form = $('dbForm');
    $('dbFormTitle').textContent = db ? 'แก้ไขฐานข้อมูล' : 'เพิ่มฐานข้อมูลใหม่';
    $('dbFormId').value   = db ? db.id : '';
    $('dbFormName').value = db ? db.name : '';
    $('dbFormType').value = db ? db.db_type : 'mysql';
    $('dbFormUrl').value  = db ? db.url : '';
    $('dbFormDesc').value = db ? db.description : '';
    this._dbEditingId = db ? db.id : null;
    form.classList.remove('hidden');
    $('dbFormName').focus();
  }

  _hideDbForm() {
    $('dbForm').classList.add('hidden');
    this._dbEditingId = null;
  }

  async _saveDatabase() {
    const name    = $('dbFormName').value.trim();
    const db_type = $('dbFormType').value;
    const url     = $('dbFormUrl').value.trim();
    const description = $('dbFormDesc').value.trim();

    if (!name || !url) {
      toast('กรุณากรอกชื่อและ URL ให้ครบ', 'warn');
      return;
    }

    const saveBtn = $('dbFormSaveBtn');
    saveBtn.disabled = true;

    try {
      if (this._dbEditingId) {
        await API.updateDatabase(this._dbEditingId, { name, db_type, url, description });
        toast('อัปเดตฐานข้อมูลเรียบร้อย', 'success');
      } else {
        await API.addDatabase({ name, db_type, url, description, enabled: true });
        toast('เพิ่มฐานข้อมูลเรียบร้อย', 'success');
      }
      this._hideDbForm();
      await this._loadDatabases();
    } catch (err) {
      toast(`บันทึกไม่สำเร็จ: ${err.message}`, 'error');
    } finally {
      saveBtn.disabled = false;
    }
  }

  /* ── User Management ─────────────────────────────────────────── */
  async _loadUsers() {
    const list = $('userList');
    try {
      const data = await API.listUsers();
      const users = data.users || [];

      const badge = $('userCountBadge');
      if (users.length > 0) {
        badge.textContent = users.length;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }

      if (!users.length) {
        list.innerHTML = `<div class="db-list-empty">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
          </svg>
          <p>ยังไม่มีผู้ใช้งาน</p></div>`;
        return;
      }

      const me = Auth.getUsername();
      list.innerHTML = users.map(u => {
        const roleLabel = u.role === 'admin' ? 'Admin' : 'User';
        const roleCls   = `role-badge role-${u.role}`;
        const statusCls = u.enabled ? 'user-status--active' : 'user-status--inactive';
        const statusTxt = u.enabled ? 'Active' : 'Disabled';
        const isSelf    = u.username === me;
        return `
          <div class="db-item user-item" data-id="${u.id}">
            <div class="db-item-left">
              <div class="user-avatar-sm">${(u.display_name || u.username)[0].toUpperCase()}</div>
              <div class="db-item-info">
                <div class="db-item-name">
                  ${esc(u.display_name || u.username)}
                  ${isSelf ? '<span class="user-self-badge">ฉัน</span>' : ''}
                </div>
                <div class="db-item-meta">
                  <span class="${roleCls}">${roleLabel}</span>
                  <span class="db-item-url">${esc(u.username)}</span>
                  <span class="user-status ${statusCls}">${statusTxt}</span>
                </div>
              </div>
            </div>
            <div class="db-item-actions">
              <label class="db-toggle" title="${u.enabled ? 'ปิดใช้งาน' : 'เปิดใช้งาน'}">
                <input type="checkbox" class="db-toggle-input user-toggle-input"
                  data-user-id="${u.id}" ${u.enabled ? 'checked' : ''} ${isSelf ? 'disabled' : ''} />
                <span class="db-toggle-track"></span>
              </label>
              <button class="db-action-btn db-edit-btn user-edit-btn" data-edit="${u.id}" title="แก้ไข">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
              </button>
              <button class="db-action-btn db-delete-btn user-delete-btn"
                data-delete="${u.id}" title="ลบ" ${isSelf ? 'disabled' : ''}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                  <path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/>
                </svg>
              </button>
            </div>
          </div>`;
      }).join('');

      list.querySelectorAll('.user-toggle-input').forEach(chk => {
        chk.addEventListener('change', async () => {
          try {
            await API.updateUser(chk.dataset.userId, { enabled: chk.checked });
            toast(chk.checked ? 'เปิดใช้งานแล้ว' : 'ปิดใช้งานแล้ว', 'success', 2000);
            await this._loadUsers();
          } catch (err) {
            toast(`ไม่สำเร็จ: ${err.message}`, 'error');
            chk.checked = !chk.checked;
          }
        });
      });

      list.querySelectorAll('.user-edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const u = users.find(x => x.id === btn.dataset.edit);
          if (u) this._showUserForm(u);
        });
      });

      list.querySelectorAll('.user-delete-btn:not([disabled])').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (!confirm('ต้องการลบผู้ใช้นี้ใช่หรือไม่?')) return;
          try {
            await API.deleteUser(btn.dataset.delete);
            toast('ลบผู้ใช้แล้ว', 'success');
            await this._loadUsers();
          } catch (err) {
            toast(`ลบไม่สำเร็จ: ${err.message}`, 'error');
          }
        });
      });

    } catch (err) {
      list.innerHTML = `<p class="docs-empty">โหลดข้อมูลไม่สำเร็จ: ${esc(err.message)}</p>`;
    }
  }

  _showUserForm(user = null) {
    const isEdit = !!user;
    $('userFormTitle').textContent   = isEdit ? 'แก้ไขผู้ใช้' : 'เพิ่มผู้ใช้ใหม่';
    $('userFormId').value            = isEdit ? user.id : '';
    $('userFormUsername').value      = isEdit ? user.username : '';
    $('userFormUsername').disabled   = isEdit;  // cannot change username
    $('userFormRole').value          = isEdit ? user.role : 'user';
    $('userFormDisplay').value       = isEdit ? (user.display_name || '') : '';
    $('userFormPassword').value      = '';
    $('userFormPassword').placeholder = isEdit ? 'เว้นว่างไว้หากไม่ต้องการเปลี่ยน' : 'อย่างน้อย 6 ตัวอักษร';
    $('userFormPwLabel').innerHTML   = isEdit
      ? 'รหัสผ่านใหม่ (เว้นว่างเพื่อคงเดิม)'
      : 'รหัสผ่าน <span class="required">*</span>';
    this._userEditingId = isEdit ? user.id : null;
    $('userForm').classList.remove('hidden');
    (isEdit ? $('userFormDisplay') : $('userFormUsername')).focus();
  }

  _hideUserForm() {
    $('userForm').classList.add('hidden');
    this._userEditingId = null;
  }

  async _saveUser() {
    const username     = $('userFormUsername').value.trim();
    const role         = $('userFormRole').value;
    const display_name = $('userFormDisplay').value.trim();
    const password     = $('userFormPassword').value;

    if (!this._userEditingId && !username) {
      toast('กรุณากรอกชื่อผู้ใช้', 'warn');
      return;
    }
    if (!this._userEditingId && !password) {
      toast('กรุณากรอกรหัสผ่าน', 'warn');
      return;
    }

    const saveBtn = $('userFormSaveBtn');
    saveBtn.disabled = true;

    try {
      if (this._userEditingId) {
        const patch = { role, display_name };
        if (password) patch.password = password;
        await API.updateUser(this._userEditingId, patch);
        toast('อัปเดตผู้ใช้เรียบร้อย', 'success');
      } else {
        await API.createUser({ username, password, role, display_name });
        toast('เพิ่มผู้ใช้เรียบร้อย', 'success');
      }
      this._hideUserForm();
      await this._loadUsers();
    } catch (err) {
      toast(`บันทึกไม่สำเร็จ: ${err.message}`, 'error');
    } finally {
      saveBtn.disabled = false;
    }
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
