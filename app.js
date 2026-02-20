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

function getRoute() {
  const hash = location.hash || '';
  if (hash.startsWith('#/')) return decodeURIComponent(hash.slice(2));
  return null;
}

function setRoute(path) {
  location.hash = '#/' + encodeURIComponent(path);
}

function isRelativeMarkdownLink(href) {
  if (!href) return false;
  if (href.startsWith('http://') || href.startsWith('https://')) return false;
  if (href.startsWith('#')) return false;
  return href.endsWith('.md') || href.includes('.md#');
}

function resolveRelative(basePath, href) {
  // basePath: content/ml-tutorial/chapters/00-setup.md
  const baseDir = basePath.split('/').slice(0, -1).join('/') + '/';
  const url = new URL(href, location.origin + '/' + baseDir);
  return url.pathname.replace(/^\//, '');
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

  // Intercept relative markdown links so they open inside the viewer.
  for (const a of container.querySelectorAll('a')) {
    const href = a.getAttribute('href');
    if (isRelativeMarkdownLink(href)) {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const resolved = resolveRelative(basePath, href);
        setRoute(resolved);
      });
    }
  }
}

async function loadDoc(nav, path) {
  const titleEl = document.getElementById('currentTitle');
  const rawEl = document.getElementById('openRaw');

  const item = nav.items.find((x) => x.path === path);
  titleEl.textContent = item ? item.title : path;
  rawEl.href = path;

  const md = await fetchText(path);
  renderMarkdown(md, path);
}

function setActiveLink(path) {
  const navEl = document.getElementById('nav');
  for (const a of navEl.querySelectorAll('a')) {
    a.classList.toggle('active', a.dataset.path === path);
  }
}

function setPager(nav, path) {
  const idx = nav.items.findIndex((x) => x.path === path);
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  prevBtn.disabled = idx <= 0;
  nextBtn.disabled = idx < 0 || idx >= nav.items.length - 1;

  prevBtn.onclick = () => {
    if (idx > 0) setRoute(nav.items[idx - 1].path);
  };
  nextBtn.onclick = () => {
    if (idx >= 0 && idx < nav.items.length - 1) setRoute(nav.items[idx + 1].path);
  };
}

async function main() {
  const nav = await fetchJson('nav.json');

  const navEl = document.getElementById('nav');
  navEl.innerHTML = '';

  for (const item of nav.items) {
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
    const path = getRoute() || nav.items[0].path;
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
