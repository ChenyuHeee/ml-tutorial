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

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function extractAcceptanceTasks(container) {
  const keywords = ['验收任务', '验收标准', '验收清单'];
  const headings = Array.from(container.querySelectorAll('h2, h3, h4'));
  const targetHeading = headings.find((h) =>
    keywords.some((k) => (h.textContent || '').includes(k))
  );
  if (!targetHeading) return [];

  const headingLevel = Number(targetHeading.tagName.slice(1));
  const tasks = [];

  // Walk siblings until next heading with level <= current
  let node = targetHeading.nextElementSibling;
  while (node) {
    const tag = node.tagName?.toLowerCase?.() || '';
    if (tag.startsWith('h')) {
      const lvl = Number(tag.slice(1));
      if (!Number.isNaN(lvl) && lvl <= headingLevel) break;
    }

    if (tag === 'ul' || tag === 'ol') {
      for (const li of node.querySelectorAll(':scope > li')) {
        const t = (li.textContent || '').trim();
        if (t) tasks.push(t);
      }
      // Prefer the first list as the checklist.
      if (tasks.length > 0) break;
    }

    node = node.nextElementSibling;
  }

  return tasks;
}

function makeChecklistKey(path, idx) {
  return `ml_tutorial_check:${path}::${idx}`;
}

function buildChecklist(container, path) {
  // Remove previous checklist if any
  const old = container.querySelector('.chapter-checklist');
  if (old) old.remove();

  const tasks = extractAcceptanceTasks(container);
  if (!tasks || tasks.length === 0) return;

  const wrap = document.createElement('div');
  wrap.className = 'chapter-checklist';

  const header = document.createElement('div');
  header.className = 'checklist-header';

  const title = document.createElement('div');
  title.className = 'checklist-title';
  title.textContent = '本章验收清单';

  const progress = document.createElement('div');
  progress.className = 'checklist-progress';

  header.appendChild(title);
  header.appendChild(progress);

  const items = document.createElement('div');
  items.className = 'checklist-items';

  function updateProgress() {
    const checked = items.querySelectorAll('input[type="checkbox"]:checked').length;
    progress.textContent = `${checked}/${tasks.length} 已完成`;
  }

  tasks.forEach((text, idx) => {
    const row = document.createElement('label');
    row.className = 'checklist-item';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const key = makeChecklistKey(path, idx);
    cb.checked = localStorage.getItem(key) === '1';
    cb.addEventListener('change', () => {
      localStorage.setItem(key, cb.checked ? '1' : '0');
      updateProgress();
    });

    const span = document.createElement('div');
    span.className = 'checklist-item-text';
    span.textContent = text;

    row.appendChild(cb);
    row.appendChild(span);
    items.appendChild(row);
  });

  wrap.appendChild(header);
  wrap.appendChild(items);
  container.prepend(wrap);
  updateProgress();
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

  buildChecklist(container, basePath);
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
  const escaped = escapeHtml(text);

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

function setupSearch(nav) {
  const input = document.getElementById('searchInput');
  const resultsEl = document.getElementById('searchResults');
  const box = document.getElementById('searchBox');
  if (!input || !resultsEl || !box) return;

  const searchable = nav.items.filter((x) => !isDividerItem(x));
  let indexPromise = null;

  function hideResults() {
    resultsEl.hidden = true;
    resultsEl.innerHTML = '';
  }

  function showResults() {
    resultsEl.hidden = false;
  }

  function buildIndexOnce() {
    if (indexPromise) return indexPromise;
    indexPromise = (async () => {
      const docs = [];
      for (const item of searchable) {
        try {
          const text = await fetchText(item.path);
          docs.push({
            title: item.title,
            path: item.path,
            text: text,
            textLower: text.toLowerCase(),
            titleLower: item.title.toLowerCase(),
          });
        } catch {
          // ignore missing
        }
      }
      return docs;
    })();
    return indexPromise;
  }

  function makeSnippet(fullText, queryLower) {
    const lower = fullText.toLowerCase();
    const at = lower.indexOf(queryLower);
    if (at < 0) return '';
    const start = Math.max(0, at - 40);
    const end = Math.min(fullText.length, at + queryLower.length + 80);
    return fullText.slice(start, end).replaceAll('\n', ' ').trim();
  }

  function renderResults(items, queryLower) {
    if (items.length === 0) {
      resultsEl.innerHTML = `<div class="search-item"><div class="search-item-title">无结果</div></div>`;
      showResults();
      return;
    }

    const html = items
      .map((x) => {
        const snip = escapeHtml(makeSnippet(x.text, queryLower));
        return (
          `<a class="search-item" href="#/${encodeURIComponent(x.path)}" data-path="${escapeHtml(
            x.path
          )}">` +
          `<div class="search-item-title">${escapeHtml(x.title)}</div>` +
          (snip ? `<div class="search-item-snippet">${snip}</div>` : '') +
          `</a>`
        );
      })
      .join('');
    resultsEl.innerHTML = html;
    for (const a of resultsEl.querySelectorAll('a.search-item')) {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const p = a.dataset.path;
        if (p) setRoute(p);
        hideResults();
        input.blur();
      });
    }
    showResults();
  }

  let timer = null;
  input.addEventListener('input', () => {
    const q = (input.value || '').trim();
    if (q.length < 2) {
      hideResults();
      return;
    }

    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      const queryLower = q.toLowerCase();
      const docs = await buildIndexOnce();

      const hits = [];
      for (const d of docs) {
        const inTitle = d.titleLower.includes(queryLower);
        const inBody = d.textLower.includes(queryLower);
        if (!inTitle && !inBody) continue;
        hits.push(d);
        if (hits.length >= 20) break;
      }
      renderResults(hits, queryLower);
    }, 180);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      hideResults();
      input.blur();
    }
  });

  document.addEventListener('click', (e) => {
    if (!box.contains(e.target)) hideResults();
  });
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

  setupSearch(nav);

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
