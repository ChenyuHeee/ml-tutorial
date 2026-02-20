async function fetchJson(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return await res.json();
}

async function fetchText(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return await res.text();
}

function isDividerItem(item) {
  return typeof item?.title === 'string' && item.title.trim().startsWith('—');
}

function isMarkdownPath(path) {
  return typeof path === 'string' && (path.endsWith('.md') || path.includes('.md#'));
}

function getExtension(path) {
  const clean = (path || '').split('#')[0].split('?')[0];
  const idx = clean.lastIndexOf('.');
  if (idx < 0) return '';
  return clean.slice(idx + 1).toLowerCase();
}

function guessLanguageFromPath(path) {
  const ext = getExtension(path);
  if (ext === 'py') return 'python';
  if (ext === 'sh') return 'bash';
  if (ext === 'yml' || ext === 'yaml') return 'yaml';
  if (ext === 'json') return 'json';
  if (ext === 'js') return 'javascript';
  if (ext === 'css') return 'css';
  if (ext === 'html') return 'xml';
  if (ext === 'txt') return 'plaintext';
  return '';
}

function getRoute() {
  const hash = location.hash || '';
  if (hash.startsWith('#/')) return decodeURIComponent(hash.slice(2));
  return null;
}

function setRoute(path) {
  location.hash = '#/' + encodeURIComponent(path);
}

function isRelativeDocLink(href) {
  if (!href) return false;
  if (href.startsWith('http://') || href.startsWith('https://')) return false;
  if (href.startsWith('#')) return false;

  const lower = href.toLowerCase();
  const exts = ['.md', '.py', '.sh', '.yml', '.yaml', '.json', '.txt'];
  return exts.some((e) => lower.endsWith(e) || lower.includes(e + '#'));
}

function resolveRelative(basePath, href) {
  // basePath: content/ml-tutorial/chapters/00-setup.md
  const baseDir = basePath.split('/').slice(0, -1).join('/') + '/';
  const url = new URL(href, location.origin + '/' + baseDir);
  return url.pathname.replace(/^\//, '');
}

function slugify(text) {
  return String(text)
    .trim()
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'section';
}

function buildToc(container) {
  const headings = Array.from(container.querySelectorAll('h2, h3'));
  if (headings.length === 0) return;

  const used = new Set();
  for (const h of headings) {
    const base = slugify(h.textContent || '');
    let id = base;
    let i = 2;
    while (used.has(id) || document.getElementById(id)) {
      id = `${base}-${i++}`;
    }
    used.add(id);
    h.id = id;
  }

  const toc = document.createElement('div');
  toc.className = 'toc';

  const title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = '目录';
  toc.appendChild(title);

  const list = document.createElement('div');
  list.className = 'toc-list';

  for (const h of headings) {
    const a = document.createElement('a');
    a.href = `#${h.id}`;
    a.textContent = h.textContent || '';
    a.className = h.tagName.toLowerCase() === 'h3' ? 'toc-item toc-h3' : 'toc-item toc-h2';
    a.addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    list.appendChild(a);
  }

  toc.appendChild(list);
  container.prepend(toc);
}

function addCopyButtons(container) {
  for (const pre of container.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    if (!code) continue;
    if (pre.dataset.hasCopy === '1') continue;
    pre.dataset.hasCopy = '1';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = '复制';

    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || '');
        const old = btn.textContent;
        btn.textContent = '已复制';
        setTimeout(() => (btn.textContent = old), 900);
      } catch {
        // Fallback: do nothing
      }
    });

    pre.appendChild(btn);
  }
}

function renderMarkdown(md, basePath) {
  marked.setOptions({
    gfm: true,
    breaks: false,
    highlight: function (code, lang) {
      try {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      } catch {
        return code;
      }
    },
  });

  const html = marked.parse(md);
  const container = document.getElementById('content');
  container.innerHTML = html;

  buildToc(container);
  addCopyButtons(container);

  // Intercept relative doc links so they open inside the viewer.
  for (const a of container.querySelectorAll('a')) {
    const href = a.getAttribute('href');
    if (isRelativeDocLink(href)) {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const resolved = resolveRelative(basePath, href);
        setRoute(resolved);
      });
    }
  }
}

function renderCodeFile(text, path) {
  const container = document.getElementById('content');
  const lang = guessLanguageFromPath(path);
  const escaped = text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');

  const codeHtml = `<pre><code class="language-${lang}">${escaped}</code></pre>`;
  container.innerHTML = codeHtml;

  try {
    const code = container.querySelector('code');
    if (code) hljs.highlightElement(code);
  } catch {
    // ignore
  }

  addCopyButtons(container);
}

async function loadDoc(nav, path) {
  const titleEl = document.getElementById('currentTitle');
  const rawEl = document.getElementById('openRaw');

  const item = nav.items.find((x) => x.path === path);
  titleEl.textContent = item ? item.title : path;
  rawEl.href = path;

  const text = await fetchText(path);
  if (isMarkdownPath(path)) {
    renderMarkdown(text, path);
  } else {
    renderCodeFile(text, path);
  }
}

function setActiveLink(path) {
  const navEl = document.getElementById('nav');
  for (const a of navEl.querySelectorAll('a')) {
    a.classList.toggle('active', a.dataset.path === path);
  }
}

function setPager(nav, path) {
  const pagerItems = nav.items.filter((x) => !isDividerItem(x));
  const idx = pagerItems.findIndex((x) => x.path === path);
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  prevBtn.disabled = idx <= 0;
  nextBtn.disabled = idx < 0 || idx >= pagerItems.length - 1;

  prevBtn.onclick = () => {
    if (idx > 0) setRoute(pagerItems[idx - 1].path);
  };
  nextBtn.onclick = () => {
    if (idx >= 0 && idx < pagerItems.length - 1) setRoute(pagerItems[idx + 1].path);
  };
}

async function main() {
  const nav = await fetchJson('nav.json');

  const navEl = document.getElementById('nav');
  navEl.innerHTML = '';

  for (const item of nav.items) {
    if (isDividerItem(item)) {
      const d = document.createElement('div');
      d.className = 'nav-divider';
      d.textContent = item.title.replaceAll('—', '').trim();
      navEl.appendChild(d);
      continue;
    }

    const a = document.createElement('a');
    a.textContent = item.title;
    a.href = '#/' + encodeURIComponent(item.path);
    a.dataset.path = item.path;
    a.addEventListener('click', (e) => {
      e.preventDefault();
      setRoute(item.path);
    });
    navEl.appendChild(a);
  }

  async function onRouteChange() {
    const first = nav.items.find((x) => !isDividerItem(x));
    const path = getRoute() || first?.path;
    if (!path) return;
    setActiveLink(path);
    setPager(nav, path);
    try {
      await loadDoc(nav, path);
    } catch (err) {
      const titleEl = document.getElementById('currentTitle');
      titleEl.textContent = '加载失败';
      document.getElementById('content').innerHTML = `<pre>${String(err)}</pre>`;
    }
  }

  window.addEventListener('hashchange', onRouteChange);
  await onRouteChange();
}

main();
