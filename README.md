# GitHub Pages 静态阅读界面

- `index.html`：页面入口
- `nav.json`：左侧导航（指向 `content/` 下的 markdown）
- `app.js`：加载 markdown 并渲染

发布到 `gh-pages` 分支时，会自动把 `ml-tutorial/` 复制到 `content/ml-tutorial/`。
