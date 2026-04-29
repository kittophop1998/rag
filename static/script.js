/**
 * Ruangthong RAG — Frontend
 *
 * Architecture:
 *   Auth     → Login / logout / token management
 *   API      → All HTTP / SSE calls (isolated; easy to swap backend)
 *   Storage  → Chat session persistence via localStorage (per logged-in user)
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
  DB_LIST_URL:   '/api/databases',
  DB_QUERY_URL:  '/api/db-query',
  URLS_URL:      '/api/settings/urls',
  USERS_URL:     '/api/users',
  TOKEN_KEY:     'rag_auth_token',
  ROLE_KEY:      'rag_auth_role',
  USER_KEY:      'rag_auth_user',
  SESSIONS_URL:  '/api/sessions',
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
   Handles the subset of Markdown that LLMs typically produce,
   including image syntax: ![alt](url) and bare image URLs.

   Key design: images are extracted into placeholders FIRST, all
   markdown text processing runs on the placeholder-substituted text,
   then images are restored LAST — so italic/bold regexes never
   touch raw HTML attribute strings.
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

    // ── Step 1: extract images into placeholders (raw URLs, before esc) ──
    const imgSlots = [];
    const SLOT = (i) => `\x02IMGSLOT${i}\x03`;   // \x02/\x03 = rare control chars

    // ![alt](url)  — standard markdown image
    let work = part.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, url) => {
      const i = imgSlots.length;
      imgSlots.push(`<img src="${url.trim()}" alt="${alt.replace(/"/g, '&quot;')}" class="md-img">`);
      return SLOT(i);
    });

    // bare image URL on its own: http(s)://... ending with image extension
    work = work.replace(/(?<![(\[])https?:\/\/\S+\.(?:png|jpe?g|gif|webp|svg|bmp)(?:\?\S*)?(?![)\]])/gi, (url) => {
      const i = imgSlots.length;
      imgSlots.push(`<img src="${url.trim()}" alt="" class="md-img">`);
      return SLOT(i);
    });

    // ── Step 2: escape HTML in the remaining text ──
    let t = esc(work);

    // ── Step 3: inline Markdown on escaped text (never touches imgSlots) ──
    // Inline links [text](url) — must come before italic to avoid [_text_](url) issues
    t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) =>
      `<a href="${url.trim()}" target="_blank" rel="noopener noreferrer">${text}</a>`
    );

    t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    // _italic_ only at word-boundaries (avoids matching underscores in identifiers/filenames)
    t = t.replace(/(^|[\s([>])_([^_\n]+)_(?=[\s,.)!\]<]|$)/gm, '$1<em>$2</em>');
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

    // ── Step 4: restore images LAST (after all text processing) ──
    imgSlots.forEach((html, i) => { t = t.split(esc(SLOT(i))).join(html); });

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

  async listDatabasesForUser() {
    const r = await fetch(CFG.DB_LIST_URL, { headers: Auth.headers() });
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

  // ── DB Auto-Index ────────────────────────────────────────────────
  async indexDatabase(id) {
    const r = await fetch(`${CFG.DB_URL}/${id}/index`, { method: 'POST', headers: Auth.headers() });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async getIndexStatus(id) {
    const r = await fetch(`${CFG.DB_URL}/${id}/index-status`, { headers: Auth.headers() });
    if (!r.ok) return { status: 'none', message: '' };
    return r.json().catch(() => ({ status: 'none', message: '' }));
  },

  async deleteDbIndex(id) {
    const r = await fetch(`${CFG.DB_URL}/${id}/index`, { method: 'DELETE', headers: Auth.headers() });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  // ── Per-group control ────────────────────────────────────────────
  async listGroupStates(connId) {
    const r = await fetch(`${CFG.DB_URL}/${connId}/groups`, { headers: Auth.headers() });
    if (!r.ok) return { groups: [] };
    return r.json().catch(() => ({ groups: [] }));
  },

  async setGroupEnabled(connId, groupName, enabled) {
    const r = await fetch(
      `${CFG.DB_URL}/${encodeURIComponent(connId)}/groups/${encodeURIComponent(groupName)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...Auth.headers() },
        body: JSON.stringify({ enabled }),
      },
    );
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async indexGroup(connId, groupName) {
    const r = await fetch(
      `${CFG.DB_URL}/${encodeURIComponent(connId)}/groups/${encodeURIComponent(groupName)}/index`,
      { method: 'POST', headers: Auth.headers() },
    );
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async getGroupIndexStatus(connId, groupName) {
    const r = await fetch(
      `${CFG.DB_URL}/${encodeURIComponent(connId)}/groups/${encodeURIComponent(groupName)}/index-status`,
      { headers: Auth.headers() },
    );
    if (!r.ok) return { status: 'idle', message: '' };
    return r.json().catch(() => ({ status: 'idle', message: '' }));
  },

  // ── DB Natural-Language Query ────────────────────────────────────
  async dbQuery(question, db_connection_id) {
    const r = await fetch(CFG.DB_QUERY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify({ question, db_connection_id }),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  // ── URL Source management ────────────────────────────────────────
  async listUrlSources() {
    const r = await fetch(CFG.URLS_URL, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async addUrlSource(data) {
    const r = await fetch(CFG.URLS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async updateUrlSource(id, data) {
    const r = await fetch(`${CFG.URLS_URL}/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify(data),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async deleteUrlSource(id) {
    const r = await fetch(`${CFG.URLS_URL}/${id}`, { method: 'DELETE', headers: Auth.headers() });
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

  // ── Chat sessions (server-side, per user) ────────────────────
  async listSessions() {
    const r = await fetch(CFG.SESSIONS_URL, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async createSession(title) {
    const r = await fetch(CFG.SESSIONS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify({ title }),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async getMessages(sessionId) {
    const r = await fetch(`${CFG.SESSIONS_URL}/${sessionId}/messages`, { headers: Auth.headers() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  },

  async addMessage(sessionId, role, content, sources) {
    const r = await fetch(`${CFG.SESSIONS_URL}/${sessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify({ role, content, sources: sources || [] }),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },

  async deleteSession(sessionId) {
    const r = await fetch(`${CFG.SESSIONS_URL}/${sessionId}`, {
      method: 'DELETE',
      headers: Auth.headers(),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.detail || `HTTP ${r.status}`);
    return json;
  },
};

/* ================================================================
   STORAGE  –  Chat session persistence (server-side SQLite, per user)
   Session schema: { id, title, createdAt }
   Message schema: { role: 'user'|'bot', content, sources?, ts }
   All methods are async — they proxy to the backend /api/sessions API.
   ================================================================ */
const Store = {
  _sessions: null,

  /** Drop in-memory session list cache (call after login / logout). */
  invalidate() {
    this._sessions = null;
  },

  async all() {
    if (!this._sessions) {
      try {
        const data = await API.listSessions();
        this._sessions = data.sessions || [];
      } catch {
        this._sessions = [];
      }
    }
    return this._sessions;
  },

  async create(firstMessage) {
    const title = firstMessage.slice(0, CFG.TITLE_MAX_LEN)
                + (firstMessage.length > CFG.TITLE_MAX_LEN ? '…' : '');
    const session = await API.createSession(title);
    if (this._sessions) this._sessions.unshift(session);
    return session;
  },

  async addMessage(sessionId, msg) {
    await API.addMessage(sessionId, msg.role, msg.content, msg.sources || []);
  },

  async delete(id) {
    await API.deleteSession(id);
    if (this._sessions) this._sessions = this._sessions.filter(s => s.id !== id);
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
    this._dbEditingId    = null;
    this._urlEditingId   = null;
    this._userEditingId  = null;
    this._mode           = 'rag';   // 'rag' | 'db'
    this._selectedDbId   = null;    // selected DB connection id in db mode
    this._dbConnections  = [];      // cached list of enabled DB connections
    this._eventsBound    = false;   // guard against duplicate event binding on re-login
    this._dbListCache    = [];      // latest API list (for settings → group overview)
    this._dbIndexByConn  = {};       // conn_id → last GET index-status payload
    this._dbGroupStates  = {};       // conn_id → {groupName: {enabled, last_indexed_at}}
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
    await this._launch();
  }

  async _launch() {
    showApp();
    this._applyRoleVisibility();
    this._updateUserDisplay();
    if (!this._eventsBound) {
      this._bindSidebar();
      this._bindComposer();
      this._bindModeToggle();
      this._bindSettings();
      this._bindKnowledgeBase();
      this._bindSuggestions();
      this._bindTopbarNav();
      this._bindMobileBottomNav();
      this._eventsBound = true;
    }
    this._switchView('chat');
    await this._renderHistory();
    $('chatInput').focus();
  }

  /**
   * Switch the visible top-level view.
   * @param {'chat'|'knowledge'|'settings'} name
   */
  _switchView(name) {
    const app = document.getElementById('app');
    if (!app) return;

    // Don't allow non-admins into admin views
    if ((name === 'knowledge' || name === 'settings') && !Auth.isAdmin()) {
      name = 'chat';
    }

    app.dataset.view = name;

    // Show/hide each view
    document.querySelectorAll('.view').forEach(v => {
      const isActive = v.dataset.view === name;
      v.classList.toggle('active', isActive);
      if (isActive) {
        v.removeAttribute('hidden');
        v.style.display = '';
      } else {
        v.setAttribute('hidden', '');
        v.style.display = 'none';
      }
    });

    // Highlight nav items (topbar + bottom nav)
    document.querySelectorAll('.topbar-nav-item').forEach(b =>
      b.classList.toggle('active', b.dataset.view === name)
    );
    document.querySelectorAll('.bottom-nav-item').forEach(b => {
      if (b.dataset.view) b.classList.toggle('active', b.dataset.view === name);
      else b.classList.remove('active');
    });

    // Close mobile sidebar drawer when switching pages
    const sidebar = $('sidebar');
    if (sidebar) sidebar.classList.remove('mobile-open');

    // Lazy-load data for the view
    if (name === 'knowledge')      this._loadDocs();
    else if (name === 'settings')  this._loadDatabases();
  }

  /** Show or hide elements marked admin-only based on current role. */
  _applyRoleVisibility() {
    const isAdmin = Auth.isAdmin();
    document.querySelectorAll('.admin-only').forEach(el => {
      el.style.display = isAdmin ? '' : 'none';
    });
  }

  /** Populate the topbar with username and role badge. */
  _updateUserDisplay() {
    const username = Auth.getUsername();
    const role     = Auth.getRole();
    const avatar   = $('sidebarUserAvatar');
    const nameEl   = $('sidebarUserName');
    const roleEl   = $('sidebarUserRole');

    if (avatar)  avatar.textContent  = (username[0] || '?').toUpperCase();
    if (nameEl)  nameEl.textContent  = username || '—';
    if (roleEl) {
      roleEl.textContent = role === 'admin' ? 'Admin' : 'User';
      roleEl.className   = `topbar-user-role role-badge role-${role}`;
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

    const onSubmit = async e => {
      e.preventDefault();
      errEl.textContent = '';
      const username = $('loginUsername').value.trim();
      const password = $('loginPassword').value;
      if (!username || !password) {
        errEl.textContent = 'กรุณากรอกชื่อผู้ใช้และรหัสผ่าน';
        // Re-bind for next attempt since { once: true } removed listener on first call
        form.addEventListener('submit', onSubmit, { once: true });
        return;
      }

      const btn = $('loginBtn');
      btn.disabled = true;
      $('loginBtnText').textContent = 'กำลังเข้าสู่ระบบ...';
      $('loginSpinner').classList.remove('hidden');

      try {
        await Auth.login(username, password);
        Store.invalidate();
        await this._launch();
        $('loginPassword').value = '';
      } catch (err) {
        errEl.textContent = err.message || 'เข้าสู่ระบบไม่สำเร็จ';
        $('loginPassword').value = '';
        $('loginPassword').focus();
        // Re-bind for retry after failed login
        form.addEventListener('submit', onSubmit, { once: true });
      } finally {
        btn.disabled = false;
        $('loginBtnText').textContent = 'เข้าสู่ระบบ';
        $('loginSpinner').classList.add('hidden');
      }
    };
    form.addEventListener('submit', onSubmit, { once: true });

    // Enter on username → focus password
    $('loginUsername').addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); $('loginPassword').focus(); }
    });
  }

  /* ── Sidebar ────────────────────────────────────────────────── */
  _bindSidebar() {
    // Close/collapse button inside sidebar (X button on mobile, close on desktop)
    const collapseBtn = $('sidebarCollapseBtn');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', () => {
        const sidebar = $('sidebar');
        if (window.innerWidth <= 768) {
          sidebar.classList.remove('mobile-open');
        } else {
          document.getElementById('app').classList.toggle('sidebar-collapsed');
        }
      });
    }

    // Topbar hamburger — toggle sidebar on both desktop and mobile
    const mobileBtn = $('mobileSidebarBtn');
    if (mobileBtn) {
      mobileBtn.addEventListener('click', e => {
        e.stopPropagation();
        const app = document.getElementById('app');
        const sidebar = $('sidebar');
        if (window.innerWidth <= 768) {
          sidebar.classList.toggle('mobile-open');
        } else {
          app.classList.toggle('sidebar-collapsed');
        }
      });
    }

    document.addEventListener('click', e => {
      const sidebar = $('sidebar');
      if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
        if (!sidebar.contains(e.target) && !(mobileBtn && mobileBtn.contains(e.target))) {
          sidebar.classList.remove('mobile-open');
        }
      }
    });

    const backdrop = $('sidebarBackdrop');
    if (backdrop) {
      backdrop.addEventListener('click', () => {
        $('sidebar').classList.remove('mobile-open');
      });
    }

    $('newChatBtn').addEventListener('click', async () => {
      await this._startNewChat();
      $('sidebar').classList.remove('mobile-open');
    });

    $('logoutBtn').addEventListener('click', async () => {
      Store.invalidate();
      await Auth.logout();
      showLogin();
      this._bindLogin();
    });
  }

  /* ── Topbar Navigation ──────────────────────────────────────── */
  _bindTopbarNav() {
    document.querySelectorAll('.topbar-nav-item[data-view]').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        this._switchView(view);
        if (view === 'chat') $('chatInput').focus();
      });
    });
  }

  /* ── Mobile Bottom Navigation ───────────────────────────────── */
  _bindMobileBottomNav() {
    // Tabs that switch view
    document.querySelectorAll('.bottom-nav-item[data-view]').forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        this._switchView(view);
        if (view === 'chat') $('chatInput').focus();
      });
    });

    // History — toggle sidebar drawer (only meaningful in chat view)
    const btnHistory = $('bottomNavHistory');
    if (btnHistory) {
      btnHistory.addEventListener('click', () => {
        // Make sure we're in chat view, then open the sidebar drawer
        if (document.getElementById('app').dataset.view !== 'chat') {
          this._switchView('chat');
        }
        const sidebar = $('sidebar');
        sidebar.classList.add('mobile-open');
      });
    }
  }

  /* ── Chat History ───────────────────────────────────────────── */
  async _renderHistory() {
    const container = $('chatHistory');
    const sessions = await Store.all();

    if (!sessions.length) {
      container.innerHTML = '<p class="history-empty">ยังไม่มีประวัติการสนทนา</p>';
      container.onclick = null;
      container.onkeydown = null;
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

    container.onclick = async e => {
      const delBtn = e.target.closest('[data-del]');
      if (delBtn) { e.stopPropagation(); await this._deleteSession(delBtn.dataset.del); return; }
      const item = e.target.closest('[data-id]');
      if (item) await this._loadSession(item.dataset.id);
    };

    container.onkeydown = async e => {
      if (e.key === 'Enter') {
        const item = e.target.closest('[data-id]');
        if (item) await this._loadSession(item.dataset.id);
      }
    };
  }

  async _startNewChat() {
    this.currentSessionId = null;
    this._clearMessages();
    this._showEmptyState();
    await this._renderHistory();
    $('chatInput').focus();
  }

  async _loadSession(id) {
    this.currentSessionId = id;
    this._clearMessages();
    this._hideEmptyState();

    try {
      const data = await API.getMessages(id);
      for (const msg of (data.messages || [])) {
        if (msg.role === 'user') {
          this._appendUserBubble(msg.content);
        } else {
          const { bubble } = this._createBotBubble();
          bubble.innerHTML = md(msg.content);
          if (msg.sources?.length) this._appendSources(bubble.parentElement, msg.sources);
        }
      }
    } catch {
      toast('โหลดประวัติการสนทนาไม่สำเร็จ', 'error');
    }

    await this._renderHistory();
    this._scrollToBottom();
    $('sidebar').classList.remove('mobile-open');
  }

  async _deleteSession(id) {
    if (id === this.currentSessionId) await this._startNewChat();
    await Store.delete(id);
    await this._renderHistory();
  }

  /* ── Mode Toggle (RAG ↔ DB Query) ──────────────────────────── */
  _bindModeToggle() {
    const modeBtn = $('composerModeBtn');
    const dbBtn   = $('composerDbChip');
    const modePop = $('composerModePopup');
    const dbPop   = $('composerDbPopup');

    modeBtn.addEventListener('click', e => {
      e.stopPropagation();
      this._closePopup('db');
      this._togglePopup('mode');
    });

    dbBtn.addEventListener('click', e => {
      e.stopPropagation();
      this._closePopup('mode');
      this._togglePopup('db');
      this._refreshDbSelector();
    });

    modePop.querySelectorAll('.composer-popup-item').forEach(item => {
      item.addEventListener('click', e => {
        e.stopPropagation();
        const m = item.dataset.mode;
        this._setMode(m);
        this._closePopup('mode');
      });
    });

    document.addEventListener('click', e => {
      if (!e.target.closest('.composer-tools')) {
        this._closePopup('mode');
        this._closePopup('db');
      }
    });

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        this._closePopup('mode');
        this._closePopup('db');
      }
    });
  }

  _positionPopup(popup, btn) {
    const r   = btn.getBoundingClientRect();
    const vh  = window.innerHeight;
    const pw  = Math.max(popup.offsetWidth || 250, 250);
    // Show above the chip
    const spaceAbove = r.top;
    const spaceBelow = vh - r.bottom;
    let top;
    if (spaceAbove >= 200 || spaceAbove >= spaceBelow) {
      // Open upward
      popup.style.bottom = `${vh - r.top + 8}px`;
      popup.style.top    = 'auto';
    } else {
      // Open downward
      popup.style.top    = `${r.bottom + 8}px`;
      popup.style.bottom = 'auto';
    }
    // Align left with chip, but don't overflow right edge
    let left = r.left;
    if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
    popup.style.left = `${Math.max(8, left)}px`;
  }

  _togglePopup(which) {
    const popup = which === 'mode' ? $('composerModePopup') : $('composerDbPopup');
    const btn   = which === 'mode' ? $('composerModeBtn')   : $('composerDbChip');
    const isOpen = popup.style.display !== 'none' && popup.style.display !== '';
    if (isOpen) {
      popup.style.display = 'none';
      btn.setAttribute('aria-expanded', 'false');
    } else {
      popup.style.display = 'block';
      this._positionPopup(popup, btn);
      btn.setAttribute('aria-expanded', 'true');
    }
  }

  _closePopup(which) {
    const popup = which === 'mode' ? $('composerModePopup') : $('composerDbPopup');
    const btn   = which === 'mode' ? $('composerModeBtn')   : $('composerDbChip');
    if (popup.style.display !== 'none') {
      popup.style.display = 'none';
      btn.setAttribute('aria-expanded', 'false');
    }
  }

  _setMode(mode) {
    this._mode = mode;

    const modeLabel = $('composerModeLabel');
    const modeIcon  = $('composerModeIcon');
    const dbChip    = $('composerDbChip');
    const hint      = $('composerHint');
    const input     = $('chatInput');

    $('composerModePopup').querySelectorAll('.composer-popup-item').forEach(it => {
      it.classList.toggle('active', it.dataset.mode === mode);
    });

    if (mode === 'db') {
      modeLabel.textContent = 'ถามฐานข้อมูล';
      modeIcon.innerHTML =
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>' +
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>' +
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>';
      dbChip.classList.remove('hidden');
      hint.textContent = 'Enter ส่ง · Shift+Enter ขึ้นบรรทัดใหม่ · ถาม OpenAI เกี่ยวกับฐานข้อมูลที่เลือก';
      input.placeholder = 'เช่น "ยอดขายเดือนนี้เท่าไหร่?" หรือ "แสดงสินค้า 10 รายการล่าสุด"';
      this._refreshDbSelector();
    } else {
      modeLabel.textContent = 'ถามเอกสาร';
      modeIcon.innerHTML =
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>' +
        '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>';
      dbChip.classList.add('hidden');
      hint.textContent = 'Enter ส่ง · Shift+Enter ขึ้นบรรทัดใหม่ · ระบบตอบจากเอกสาร FAISS เท่านั้น';
      input.placeholder = 'ถามเกี่ยวกับเอกสารของบริษัท...';
    }
  }

  _updateDbChipLabel() {
    const chip  = $('composerDbChip');
    const label = $('composerDbLabel');
    const sel   = this._dbConnections.find(d => d.id === this._selectedDbId);
    if (sel) {
      const info = DB_TYPES[sel.db_type] || DB_TYPES.other;
      label.textContent = `${info.icon} ${sel.name}`;
      chip.classList.remove('is-empty');
    } else {
      label.textContent = 'เลือกฐานข้อมูล';
      chip.classList.add('is-empty');
    }
  }

  async _refreshDbSelector() {
    const list   = $('composerDbList');
    const hidden = $('dbSelector');
    list.innerHTML = '<div class="composer-popup-empty">กำลังโหลด...</div>';
    try {
      const data = await API.listDatabasesForUser();
      this._dbConnections = (data.databases || []).filter(d => d.enabled);

      hidden.innerHTML = '<option value="">— เลือกฐานข้อมูล —</option>' +
        this._dbConnections.map(db =>
          `<option value="${esc(db.id)}">${esc(db.name)}</option>`
        ).join('');

      if (this._selectedDbId && !this._dbConnections.find(d => d.id === this._selectedDbId)) {
        this._selectedDbId = null;
      }
      hidden.value = this._selectedDbId || '';

      if (!this._dbConnections.length) {
        list.innerHTML =
          '<div class="composer-popup-empty">ยังไม่มีฐานข้อมูลที่เปิดใช้งาน<br/>' +
          '<span style="font-size:11px">เพิ่มได้ที่ตั้งค่าระบบ → ฐานข้อมูล</span></div>';
      } else {
        list.innerHTML = this._dbConnections.map(db => {
          const info = DB_TYPES[db.db_type] || DB_TYPES.other;
          const active = db.id === this._selectedDbId ? 'active' : '';
          return `
            <button type="button" class="composer-popup-item ${active}" data-id="${esc(db.id)}">
              <span class="composer-popup-icon" style="font-size:16px">${info.icon}</span>
              <span class="composer-popup-text">
                <span class="composer-popup-title">${esc(db.name)}</span>
                <span class="composer-popup-desc">${esc(info.label)}</span>
              </span>
              <svg class="composer-popup-check" width="14" height="14" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            </button>`;
        }).join('');

        list.querySelectorAll('.composer-popup-item').forEach(it => {
          it.addEventListener('click', e => {
            e.stopPropagation();
            const id = it.dataset.id;
            this._selectedDbId = id;
            hidden.value = id;
            this._updateDbChipLabel();
            list.querySelectorAll('.composer-popup-item').forEach(x =>
              x.classList.toggle('active', x.dataset.id === id)
            );
            this._closePopup('db');
          });
        });
      }
    } catch {
      list.innerHTML = '<div class="composer-popup-empty">โหลดไม่สำเร็จ</div>';
    }
    this._updateDbChipLabel();
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

    // DB mode guard
    if (this._mode === 'db') {
      if (!this._selectedDbId) {
        toast('กรุณาเลือกฐานข้อมูลก่อนส่งคำถาม', 'warn');
        $('composerDbChip').click();
        return;
      }
      return this._sendDbQuery(question);
    }

    // ── RAG mode ─────────────────────────────────────────────────
    input.value = '';
    input.style.height = 'auto';
    $('sendBtn').disabled = true;

    if (!this.currentSessionId) {
      const session = await Store.create(question);
      this.currentSessionId = session.id;
      await this._renderHistory();
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

  async _sendDbQuery(question) {
    const input = $('chatInput');
    input.value = '';
    input.style.height = 'auto';
    $('sendBtn').disabled = true;

    if (!this.currentSessionId) {
      const session = await Store.create(question);
      this.currentSessionId = session.id;
      await this._renderHistory();
    }

    this._hideEmptyState();
    this._appendUserBubble(question);
    Store.addMessage(this.currentSessionId, { role: 'user', content: question });

    const { msgEl, bubble, setContent, finalize } = this._createBotBubble(true);
    this._scrollToBottom();

    this.isStreaming = true;
    let fullText = '';

    try {
      const result = await API.dbQuery(question, this._selectedDbId);
      fullText = result.answer || '(ไม่มีคำตอบ)';
      setContent(fullText, false);
      finalize();
    } catch (err) {
      if (err.message.includes('401') || err.message.toLowerCase().includes('session')) {
        Auth.clearToken(); showLogin(); return;
      }
      fullText = `ขออภัยครับ เกิดข้อผิดพลาด: ${err.message}`;
      setContent(fullText, false);
      finalize();
    } finally {
      Store.addMessage(this.currentSessionId, { role: 'bot', content: fullText });
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
    msgEl.className = 'message bot' + (showTyping ? ' is-thinking' : '');

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
      msgEl.classList.remove('is-thinking');
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

  /* ── DB Query Result panel ──────────────────────────────────── */
  _appendDbResult(msgEl, result) {
    const body = msgEl.querySelector('.msg-body');
    if (!body) return;

    const wrap = document.createElement('div');
    wrap.className = 'db-result-wrap';

    // ── Query panel (collapsible) ──────────────────────────────
    const queryLang = result.db_type === 'mongodb' ? 'MongoDB' : 'SQL';
    const queryToggle = document.createElement('button');
    queryToggle.className = 'db-result-toggle';
    queryToggle.innerHTML =
      `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>` +
      `<span>🔍 ${queryLang} ที่ใช้</span>`;

    const queryPanel = document.createElement('div');
    queryPanel.className = 'db-result-panel hidden';
    queryPanel.innerHTML = `<pre class="db-query-code"><code>${esc(result.query || '')}</code></pre>`;

    queryToggle.addEventListener('click', () => {
      const open = queryPanel.classList.toggle('hidden');
      queryToggle.querySelector('svg').style.transform = open ? '' : 'rotate(180deg)';
    });

    wrap.appendChild(queryToggle);
    wrap.appendChild(queryPanel);

    // ── Data table (collapsible) ───────────────────────────────
    if (result.rows && result.rows.length > 0) {
      const rowCount = result.row_count ?? result.rows.length;
      const shown = result.rows.length;
      const caption = rowCount > shown
        ? `📊 ผลลัพธ์ (แสดง ${shown} จาก ${rowCount} แถว)`
        : `📊 ผลลัพธ์ (${rowCount} แถว)`;

      const tableToggle = document.createElement('button');
      tableToggle.className = 'db-result-toggle';
      tableToggle.innerHTML =
        `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>` +
        `<span>${caption}</span>`;

      const tablePanel = document.createElement('div');
      tablePanel.className = 'db-result-panel hidden';

      const keys = Object.keys(result.rows[0]);
      const thead = `<thead><tr>${keys.map(k => `<th>${esc(k)}</th>`).join('')}</tr></thead>`;
      const tbody = `<tbody>${result.rows.map(row =>
        `<tr>${keys.map(k => `<td>${esc(row[k] === null || row[k] === undefined ? '' : String(row[k]))}</td>`).join('')}</tr>`
      ).join('')}</tbody>`;

      const tableWrap = document.createElement('div');
      tableWrap.className = 'db-table-wrap';
      tableWrap.innerHTML = `<table class="db-table">${thead}${tbody}</table>`;
      tablePanel.appendChild(tableWrap);

      tableToggle.addEventListener('click', () => {
        const open = tablePanel.classList.toggle('hidden');
        tableToggle.querySelector('svg').style.transform = open ? '' : 'rotate(180deg)';
      });

      wrap.appendChild(tableToggle);
      wrap.appendChild(tablePanel);
    }

    const bubble = body.querySelector('.bubble');
    if (bubble) bubble.appendChild(wrap);
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
    const reindexBtn  = $('reindexBtn');
    const uploadInput = $('uploadInput');
    if (!reindexBtn || !uploadInput) return;

    reindexBtn.addEventListener('click', async () => {
      reindexBtn.disabled = true;
      const original = reindexBtn.innerHTML;
      reindexBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 1s linear infinite"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> กำลัง Rebuild...`;
      try {
        await API.reindex();
        toast('สร้าง Index สำเร็จแล้ว — พร้อมตอบคำถามจากเอกสารใหม่!', 'success');
        await this._loadDocs();
      } catch (err) {
        toast(`Rebuild ไม่สำเร็จ: ${err.message}`, 'error');
      } finally {
        reindexBtn.disabled = false;
        reindexBtn.innerHTML = original;
      }
    });

    uploadInput.addEventListener('change', async e => {
      const files = Array.from(e.target.files || []);
      e.target.value = '';
      if (!files.length) return;
      await this._uploadFiles(files);
    });
  }

  async _loadDocs() {
    const list = $('docsList');
    if (!list) return;
    try {
      const data = await API.documents();
      if (!data.documents.length) {
        list.innerHTML = '<p class="docs-empty">ยังไม่มีไฟล์ PDF<br>กดอัปโหลดเพื่อเพิ่มเอกสาร</p>';
        return;
      }
      const fileSvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
      list.innerHTML = data.documents.map(doc => {
        const statusKey   = doc.indexed ? 'ready' : 'indexing';
        const statusLabel = doc.indexed ? 'Ready' : 'Indexing';
        return `
        <div class="doc-item" title="${esc(doc.name)}">
          <span class="doc-item-icon">${fileSvg}</span>
          <span class="doc-item-name">${esc(doc.name)}</span>
          <span class="doc-item-size">${fmtBytes(doc.size)}</span>
          <span class="doc-item-badge ${statusKey}">${statusLabel}</span>
        </div>`;
      }).join('');
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

  /* ── Settings Page ──────────────────────────────────────────── */
  _bindSettings() {
    // Settings opens via _switchView('settings') from topbar/bottom nav.
    // Tab switching within the page:
    document.querySelectorAll('.settings-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.settings-tab-panel').forEach(p => {
          p.classList.remove('active');
          p.style.display = 'none';
          p.setAttribute('aria-hidden', 'true');
        });
        tab.classList.add('active');
        const panel = $(`tab${tab.dataset.tab.charAt(0).toUpperCase() + tab.dataset.tab.slice(1)}`);
        if (panel) {
          panel.classList.add('active');
          panel.style.display = '';
          panel.removeAttribute('aria-hidden');
        }
        /* Collapse forms that belong to other tabs so state stays consistent */
        if (tab.dataset.tab !== 'users')      this._hideUserForm();
        if (tab.dataset.tab !== 'databases')  this._hideDbForm();
        if (tab.dataset.tab !== 'urls')       this._hideUrlForm();
        if (tab.dataset.tab === 'databases') this._loadDatabases();
        if (tab.dataset.tab === 'urls')      this._loadUrlSources();
        if (tab.dataset.tab === 'users')     this._loadUsers();
      });
    });

    // DB: add / edit
    $('addDbBtn').addEventListener('click', () => this._showDbForm());
    $('dbFormCancelBtn').addEventListener('click', () => this._hideDbForm());
    $('dbFormSaveBtn').addEventListener('click', () => this._saveDatabase());

    // URLs: add / edit
    $('addUrlBtn').addEventListener('click', () => this._showUrlForm());
    $('urlFormCancelBtn').addEventListener('click', () => this._hideUrlForm());
    $('urlFormSaveBtn').addEventListener('click', () => this._saveUrlSource());

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
        this._dbListCache = [];
        this._dbIndexByConn = {};
        this._renderDbGroupOverview();
        list.innerHTML = `
          <div class="db-list-empty">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
              <ellipse cx="12" cy="5" rx="9" ry="3" opacity=".55"/>
              <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" opacity=".75"/>
              <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
              <line x1="7" y1="9.4" x2="9" y2="9.4" stroke-width="1.2" opacity=".6"/>
              <line x1="7" y1="16" x2="11" y2="16" stroke-width="1.2" opacity=".4"/>
            </svg>
            <p><strong>ยังไม่มีฐานข้อมูล</strong><br>กด "เพิ่มฐานข้อมูล" เพื่อเริ่มต้น</p>
          </div>`;
        return;
      }

      this._dbListCache = dbs;
      this._dbIndexByConn = {};

      list.innerHTML = dbs.map(db => {
        const typeInfo = DB_TYPES[db.db_type] || DB_TYPES.other;
        const maskedUrl = this._maskUrl(db.url);
        return `
          <div class="db-item" data-id="${db.id}">
            <div class="db-item-row">
              <div class="db-item-left">
                <span class="db-item-icon">${typeInfo.icon}</span>
                <div class="db-item-info">
                  <div class="db-item-name">${esc(db.name)}</div>
                  <div class="db-item-meta">
                    <span class="db-type-badge">${esc(typeInfo.label)}</span>
                    <span class="db-item-url" title="${esc(db.url)}">${esc(maskedUrl)}</span>
                  </div>
                  ${db.description ? `<div class="db-item-desc">${esc(db.description)}</div>` : ''}
                  <div class="db-index-status-row" id="idx-row-${db.id}">
                    <span class="db-index-badge db-index-none" id="idx-badge-${db.id}">
                      ยังไม่ได้ Index
                    </span>
                  </div>
                </div>
              </div>
              <div class="db-item-actions">
                <label class="db-toggle" title="${db.enabled ? 'ปิดใช้งาน' : 'เปิดใช้งาน'}">
                  <input type="checkbox" class="db-toggle-input" data-db-id="${db.id}" ${db.enabled ? 'checked' : ''} />
                  <span class="db-toggle-track"></span>
                </label>
                <button class="db-action-btn db-index-btn" data-index-id="${db.id}" title="Index ข้อมูลลง RAG">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                    <line x1="12" y1="22.08" x2="12" y2="12"/>
                  </svg>
                </button>
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
            </div>
          </div>`;
      }).join('');

      this._dbGroupStates = {};
      dbs.forEach(db => {
        API.getIndexStatus(db.id).then(s => this._applyIndexStatus(db.id, s)).catch(() => {});
        API.listGroupStates(db.id).then(r => {
          const map = {};
          (r.groups || []).forEach(g => { map[g.group_name] = g; });
          this._dbGroupStates[db.id] = map;
          this._renderDbGroupOverview();
        }).catch(() => {});
      });

      this._renderDbGroupOverview();

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

      list.querySelectorAll('[data-index-id]').forEach(btn => {
        btn.addEventListener('click', () => this._handleDbIndex(btn.dataset.indexId));
      });

    } catch (err) {
      this._dbListCache = [];
      this._dbIndexByConn = {};
      this._renderDbGroupOverview();
      list.innerHTML = `<p class="docs-empty">โหลดข้อมูลไม่สำเร็จ: ${esc(err.message)}</p>`;
    }
  }

  /** Rebuild overview list below DB cards — one row per [DB name] + domain group */
  _renderDbGroupOverview() {
    const ul = $('dbGroupIndexList');
    if (!ul) return;

    const cmp = (a, b) => String(a).localeCompare(String(b), 'th');

    const rows = [];
    for (const db of this._dbListCache || []) {
      const st = this._dbIndexByConn[db.id];
      if (!st || st.status !== 'indexed') continue;
      const gm = st.group_map || {};
      const tblRows = st.table_rows || {};
      const groupStates = this._dbGroupStates[db.id] || {};
      const names = Object.keys(gm).sort(cmp);
      for (const gName of names) {
        const tables = gm[gName] || [];
        const line = `[DB] ${db.name} ${gName}`;
        const tablesDetail = tables
          .map(t => {
            const n = tblRows[t];
            return typeof n === 'number' ? `${t} (${n} rows)` : t;
          })
          .join(', ');
        const tipOneLine = tablesDetail ? `${line} — ${tablesDetail}` : line;
        const stateRow = groupStates[gName] || {};
        const enabled = stateRow.enabled !== false;
        rows.push({ line, tipOneLine, tblCount: tables.length, enabled, connId: db.id, gName });
      }
    }
    rows.sort((x, y) => cmp(x.line, y.line));

    if (!rows.length) {
      ul.innerHTML = `
        <li class="db-group-overview-placeholder">
          — ยังไม่มีรายการ —
          <span class="db-group-overview-hint-inline">เมื่อ Index สำเร็จ โดเมนที่ตรวจพบจะแสดงที่นี่</span>
        </li>`;
      return;
    }

    const spinSvg = `<svg class="grp-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>`;
    const rebuildSvg = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/></svg>`;

    ul.innerHTML = rows.map(r => `
      <li class="db-group-overview-item${r.enabled ? '' : ' grp-disabled'}"
          data-conn="${esc(r.connId)}" data-gname="${esc(r.gName)}"
          title="${esc(r.tipOneLine)}">
        <label class="grp-toggle" title="${r.enabled ? 'ปิดกลุ่มนี้' : 'เปิดกลุ่มนี้'}">
          <input type="checkbox" class="grp-toggle-input" ${r.enabled ? 'checked' : ''}
            data-conn="${esc(r.connId)}" data-gname="${esc(r.gName)}" />
          <span class="grp-toggle-track"></span>
        </label>
        <span class="db-group-overview-line">${esc(r.line)}</span>
        ${r.tblCount ? `<span class="db-group-overview-n">${r.tblCount} ตาราง</span>` : ''}
        <button class="grp-rebuild-btn" data-conn="${esc(r.connId)}" data-gname="${esc(r.gName)}"
          title="Rebuild Index กลุ่มนี้">
          <span class="grp-rebuild-label">${rebuildSvg} Rebuild</span>
          <span class="grp-rebuild-running hidden">${spinSvg} กำลัง Index...</span>
        </button>
      </li>`).join('');

    // Bind toggle
    ul.querySelectorAll('.grp-toggle-input').forEach(chk => {
      chk.addEventListener('change', async () => {
        const connId = chk.dataset.conn;
        const gName  = chk.dataset.gname;
        const li = chk.closest('li');
        try {
          await API.setGroupEnabled(connId, gName, chk.checked);
          li.classList.toggle('grp-disabled', !chk.checked);
          if (!this._dbGroupStates[connId]) this._dbGroupStates[connId] = {};
          this._dbGroupStates[connId][gName] = { enabled: chk.checked };
          toast(chk.checked ? `เปิดกลุ่ม "${gName}" แล้ว` : `ปิดกลุ่ม "${gName}" แล้ว`, 'success', 2000);
        } catch (err) {
          toast(`ไม่สำเร็จ: ${err.message}`, 'error');
          chk.checked = !chk.checked;
          li.classList.toggle('grp-disabled', !chk.checked);
        }
      });
    });

    // Bind rebuild
    ul.querySelectorAll('.grp-rebuild-btn').forEach(btn => {
      btn.addEventListener('click', () => this._handleGroupRebuild(btn));
    });
  }

  async _handleGroupRebuild(btn) {
    const connId = btn.dataset.conn;
    const gName  = btn.dataset.gname;
    const label   = btn.querySelector('.grp-rebuild-label');
    const running = btn.querySelector('.grp-rebuild-running');

    btn.disabled = true;
    label.classList.add('hidden');
    running.classList.remove('hidden');

    try {
      await API.indexGroup(connId, gName);
      toast(`เริ่ม Rebuild กลุ่ม "${gName}"`, 'success', 3000);
      this._pollGroupRebuild(connId, gName, btn, label, running);
    } catch (err) {
      toast(`Rebuild ไม่สำเร็จ: ${err.message}`, 'error');
      btn.disabled = false;
      label.classList.remove('hidden');
      running.classList.add('hidden');
    }
  }

  _pollGroupRebuild(connId, gName, btn, label, running) {
    const poll = () => {
      API.getGroupIndexStatus(connId, gName).then(s => {
        if (s.status === 'indexing') {
          if (running) running.textContent = `กำลัง Index... ${s.progress ? s.progress + '%' : ''}`;
          setTimeout(poll, 2500);
        } else {
          if (btn) { btn.disabled = false; }
          if (label) label.classList.remove('hidden');
          if (running) running.classList.add('hidden');
          if (s.status === 'done') {
            toast(`Rebuild กลุ่ม "${gName}" สำเร็จ`, 'success', 3000);
            // Refresh index status to update table_rows in overview
            API.getIndexStatus(connId).then(ns => this._applyIndexStatus(connId, ns)).catch(() => {});
          } else if (s.status === 'error') {
            toast(`Rebuild "${gName}" ล้มเหลว: ${s.message}`, 'error');
          }
        }
      }).catch(() => setTimeout(poll, 4000));
    };
    setTimeout(poll, 2500);
  }

  /** Apply an index status object to badge + overview list */
  _applyIndexStatus(id, s) {
    this._dbIndexByConn[id] = s;
    this._renderDbGroupOverview();

    const badge = document.getElementById(`idx-badge-${id}`);
    if (!badge) return;
    badge.className = 'db-index-badge';
    switch (s.status) {
      case 'indexed': {
        badge.classList.add('db-index-ok');
        const groupCount = s.groups?.length || 0;
        const tableCount = s.tables ?? Object.keys(s.table_rows || {}).length;
        badge.innerHTML = `
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
          Indexed${tableCount ? ` · ${tableCount} ตาราง` : ''}${s.chunks ? ` · ${s.chunks} chunks` : ''}${groupCount ? ` · ${groupCount} กลุ่ม` : ''}
          <button class="idx-del-btn" data-del-idx="${id}" title="ลบ Index">×</button>`;
        badge.querySelector('[data-del-idx]')?.addEventListener('click', async e => {
          e.stopPropagation();
          if (!confirm('ลบ Index ข้อมูลของ DB นี้?')) return;
          try {
            await API.deleteDbIndex(id);
            toast('ลบ Index แล้ว', 'success', 2000);
            this._applyIndexStatus(id, { status: 'none', message: 'ยังไม่ได้ Index' });
          } catch (err) { toast(`ลบไม่สำเร็จ: ${err.message}`, 'error'); }
        });
        break;
      }
      case 'indexing':
        badge.classList.add('db-index-running');
        badge.innerHTML = `<span class="idx-spinner"></span> ${esc(s.message || 'กำลัง Index...')}${s.progress ? ` (${s.progress}%)` : ''}`;
        setTimeout(() => API.getIndexStatus(id).then(ns => this._applyIndexStatus(id, ns)).catch(() => {}), 3000);
        break;
      case 'error':
        badge.classList.add('db-index-error');
        badge.textContent = `⚠ ${s.message || 'เกิดข้อผิดพลาด'}`;
        break;
      default:
        badge.classList.add('db-index-none');
        badge.textContent = 'ยังไม่ได้ Index';
    }
  }
  async _handleDbIndex(id) {
    const s = await API.getIndexStatus(id).catch(() => ({ status: 'none' }));
    if (s.status === 'indexing') {
      toast('กำลัง Index อยู่แล้ว รอสักครู่...', 'info', 3000);
      return;
    }
    if (s.status === 'indexed') {
      if (!confirm('Re-index ข้อมูลใหม่ทั้งหมด? Index เดิมจะถูกลบและสร้างใหม่')) return;
    }
    try {
      await API.indexDatabase(id);
      toast('เริ่ม Index ข้อมูล — อาจใช้เวลาสักครู่', 'success', 4000);
      this._applyIndexStatus(id, { status: 'indexing', message: 'เริ่มต้น...', progress: 0 });
    } catch (err) {
      toast(`Index ไม่สำเร็จ: ${err.message}`, 'error');
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

  /* ── URL Source Management ───────────────────────────────────── */
  async _loadUrlSources() {
    const list = $('urlList');
    try {
      const data = await API.listUrlSources();
      const sources = data.urls || [];

      const badge = $('urlCountBadge');
      if (sources.length > 0) {
        badge.textContent = sources.length;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }

      if (!sources.length) {
        list.innerHTML = `
          <div class="db-list-empty">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"
              stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="2" y1="12" x2="22" y2="12" opacity=".65"/>
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" opacity=".8"/>
            </svg>
            <p><strong>ยังไม่มี URL แหล่งข้อมูล</strong><br>กด "เพิ่ม URL" เพื่อเริ่มต้น</p>
          </div>`;
        return;
      }

      list.innerHTML = sources.map(src => {
        const depthLabel = src.crawl_depth >= 1 ? 'ดึงลิงค์ภายใน' : 'หน้าเดียว';
        const lastIndexed = src.last_indexed_at
          ? `Index แล้วเมื่อ ${new Date(src.last_indexed_at).toLocaleString('th-TH')}`
          : 'ยังไม่ได้ Index';
        return `
          <div class="db-item" data-id="${src.id}">
            <div class="db-item-row">
              <div class="db-item-left">
                <span class="db-item-icon">🌐</span>
                <div class="db-item-info">
                  <div class="db-item-name">${esc(src.name)}</div>
                  <div class="db-item-meta">
                    <span class="db-type-badge">depth ${src.crawl_depth} · ${depthLabel}</span>
                    <span class="db-item-url" title="${esc(src.url)}">${esc(src.url)}</span>
                  </div>
                  ${src.description ? `<div class="db-item-desc">${esc(src.description)}</div>` : ''}
                  <div class="db-item-desc" style="opacity:.55;font-size:.75rem">${esc(lastIndexed)}</div>
                </div>
              </div>
              <div class="db-item-actions">
                <label class="db-toggle" title="${src.enabled ? 'ปิดใช้งาน' : 'เปิดใช้งาน'}">
                  <input type="checkbox" class="db-toggle-input url-toggle-input"
                    data-url-id="${src.id}" ${src.enabled ? 'checked' : ''} />
                  <span class="db-toggle-track"></span>
                </label>
                <button class="db-action-btn db-edit-btn" data-edit="${src.id}" title="แก้ไข">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                  </svg>
                </button>
                <button class="db-action-btn db-delete-btn" data-delete="${src.id}" title="ลบ">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                    <path d="M10 11v6M14 11v6"/>
                    <path d="M9 6V4h6v2"/>
                  </svg>
                </button>
              </div>
            </div>
          </div>`;
      }).join('');

      list.querySelectorAll('.url-toggle-input').forEach(chk => {
        chk.addEventListener('change', async () => {
          try {
            await API.updateUrlSource(chk.dataset.urlId, { enabled: chk.checked });
            toast(chk.checked ? 'เปิดใช้งานแล้ว' : 'ปิดใช้งานแล้ว', 'success', 2000);
          } catch (err) {
            toast(`ไม่สำเร็จ: ${err.message}`, 'error');
            chk.checked = !chk.checked;
          }
        });
      });

      list.querySelectorAll('[data-edit]').forEach(btn => {
        btn.addEventListener('click', () => {
          const src = sources.find(s => s.id === btn.dataset.edit);
          if (src) this._showUrlForm(src);
        });
      });

      list.querySelectorAll('[data-delete]').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (!confirm('ต้องการลบ URL นี้ใช่หรือไม่?')) return;
          try {
            await API.deleteUrlSource(btn.dataset.delete);
            toast('ลบ URL แล้ว', 'success');
            await this._loadUrlSources();
          } catch (err) {
            toast(`ลบไม่สำเร็จ: ${err.message}`, 'error');
          }
        });
      });

    } catch (err) {
      list.innerHTML = `<p class="docs-empty">โหลดข้อมูลไม่สำเร็จ: ${esc(err.message)}</p>`;
    }
  }

  _showUrlForm(src = null) {
    $('urlFormTitle').textContent = src ? 'แก้ไข URL' : 'เพิ่ม URL ใหม่';
    $('urlFormId').value    = src ? src.id : '';
    $('urlFormName').value  = src ? src.name : '';
    $('urlFormUrl').value   = src ? src.url : '';
    $('urlFormDesc').value  = src ? src.description : '';
    $('urlFormDepth').value = src ? String(src.crawl_depth) : '0';
    this._urlEditingId = src ? src.id : null;
    $('urlForm').classList.remove('hidden');
    $('urlFormName').focus();
  }

  _hideUrlForm() {
    $('urlForm').classList.add('hidden');
    this._urlEditingId = null;
  }

  async _saveUrlSource() {
    const name        = $('urlFormName').value.trim();
    const url         = $('urlFormUrl').value.trim();
    const description = $('urlFormDesc').value.trim();
    const crawl_depth = parseInt($('urlFormDepth').value, 10);

    if (!name || !url) {
      toast('กรุณากรอกชื่อและ URL ให้ครบ', 'warn');
      return;
    }
    try { new URL(url); } catch {
      toast('URL ไม่ถูกต้อง กรุณาตรวจสอบ', 'warn');
      return;
    }

    const saveBtn = $('urlFormSaveBtn');
    saveBtn.disabled = true;

    try {
      if (this._urlEditingId) {
        await API.updateUrlSource(this._urlEditingId, { name, url, description, crawl_depth });
        toast('อัปเดต URL เรียบร้อย', 'success');
      } else {
        await API.addUrlSource({ name, url, description, crawl_depth, enabled: true });
        toast('เพิ่ม URL เรียบร้อย — กด Rebuild Index เพื่อนำเข้าเนื้อหา', 'success', 5000);
      }
      this._hideUrlForm();
      await this._loadUrlSources();
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
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87" opacity=".6"/>
            <path d="M16 3.13a4 4 0 0 1 0 7.75" opacity=".6"/>
          </svg>
          <p><strong>ยังไม่มีผู้ใช้งาน</strong></p></div>`;
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
            <div class="db-item-row">
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
