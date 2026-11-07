# SignalML UI Migration Guide

This guide is the binding implementation reference for migrating all 12 SignalML page templates (`backend/app/templates/*.html`) and page scripts (`backend/app/static/js/*.js`) to the SignalML Design System (`docs/DESIGN.md`, `backend/app/static/css/style.css`, `backend/app/static/css/nodes.css`, `backend/app/templates/_macros.html`, and `/styleguide`).

---

## 1. Golden Rules (Non-Negotiable)

1. **Preserve all behavior & hooks**:
   - Keep **every** `id="..."`, `name="..."`, `data-*="..."`, `onclick`/`onchange`/`oninput` attribute, form field, table body ID, modal ID, and global JS function/variable name intact.
   - Before removing or renaming any class or element, `grep` for it across `backend/app/templates/`, `backend/app/static/js/`, and `backend/tests/`.
2. **Zero emojis anywhere**:
   - Remove every emoji and decorative Unicode symbol from HTML templates, JS-generated markup, context menus, `SignalML.toast()` calls, `confirm()`/`prompt()` dialogs, and chart labels.
   - Replace with Lucide icons via `{{ icon('name') }}` (Jinja) or `SignalML.icon('name')` (JS), or plain text inside `<option>` elements (since `<option>` cannot contain `<svg>`).
3. **English-only technical copy**:
   - Translate all Turkish or bilingual Turkish/English labels, hints, comments in visible UI, placeholders, toasts, and modal copy into concise, technical English.
   - Use sentence case for buttons, labels, and card headers (`"Train model"`, `"Export CSV"`, `"Architecture wizard"`).
4. **No inline `style="..."` in templates**:
   - Replace static inline styles with classes from `style.css` and `nodes.css`.
   - The **only** permitted inline styles are dynamic runtime values set by JS (e.g., `modal.style.display = "flex"`, `progressBar.style.width = pct + "%"`, dynamic cluster swatch color `swatch.style.background = color`).
   - Delete page-level `<style>` blocks in templates when their selectors (`wizard-card`, `idec-cluster-card`, `custom-layer-item`, `toolbar-btn`, etc.) are already covered by `style.css` / `nodes.css`.
5. **No gradients, glow shadows, or purple-blue slop**:
   - Never use `linear-gradient(...)`, `backdrop-filter: blur(...)`, or `box-shadow: 0 0 ...` on buttons, badges, or cards.
   - ~90% of the UI is neutral (`--surface-1`..`--surface-5`, `--border-subtle`..`--border-strong`, `--text-primary`..`--text-muted`). Use `--accent` (`btn-primary`, `badge-accent`) only for primary actions and active selection.
6. **Tabular monospace numbers**:
   - Every metric, tensor shape, sample count, epoch, loss/accuracy value, parameter count, and ID must use `.mono` or `.num` (`--font-mono` with `tabular-nums`).

---

## 2. Icons (`_macros.html` & `SignalML.icon`)

Sprite location: `/static/vendor/lucide/sprite.svg` (lucide-static 1.69.0).

### In Jinja2 templates
At the top of every template (right after `{% extends "base.html" %}`):
```jinja2
{% extends "base.html" %}
{% from "_macros.html" import icon %}
```
Usage:
```jinja2
{{ icon('play') }}                        {# 16x16 decorative icon (aria-hidden="true") #}
{{ icon('play', 'icon-sm') }}             {# 14x14 icon #}
{{ icon('play', 'icon-xs') }}             {# 12x12 icon #}
{{ icon('folder-open', 'icon-lg') }}      {# 20x20 icon #}
{{ icon('database', 'icon-xl') }}         {# 24x24 empty-state icon #}
{{ icon('loader', 'icon-spin') }}         {# spinning loader icon #}
{{ icon('trash', '', 'Delete dataset') }} {# accessible icon with role="img" aria-label="..." #}
```

### In JavaScript (`app.js` / page scripts)
```js
SignalML.icon("play")                     // '<svg class="icon" aria-hidden="true"><use href="/static/vendor/lucide/sprite.svg#play"/></svg>'
SignalML.icon("circle-check", "icon-sm")  // with extra class
SignalML.icon("trash", "", "Delete row")  // with aria-label
```

> **Note on `<option>` elements**: HTML `<select><option>` elements only render plain text. When removing emojis from `<option>` tags (e.g., `<option value="">📂 Kayıtlı Model Yükle...</option>`), replace with clean plain English text (`<option value="">Load saved model...</option>`), **not** `<svg>`.

---

## 3. Complete Emoji-to-Lucide Mapping Table

Every emoji/symbol currently in `backend/app/templates/*.html` and `backend/app/static/js/*.js` is listed below with its verified symbol ID in `/static/vendor/lucide/sprite.svg`:

| Current Emoji / Symbol | Hex | Verified Lucide Icon Name | Typical Context / Replacement |
|---|---|---|---|
| `🪄` | `U+1FA84` | `wand-sparkles` | Architecture wizard (`openWizardModal`) |
| `📥` | `U+1F4E5` | `download` or `file-down` | Download CSV / Import model (`upload` when importing) |
| `📤` | `U+1F4E4` | `upload` or `file-up` | Export / Upload |
| `🗑️` / `🗑` | `U+1F5D1` | `trash` | Delete button (`.btn-danger` or `.btn-ghost`) |
| `⚡` | `U+26A1` | `zap` | Fast action / Auto-detect / Extract features |
| `✕` / `✗` / `&times;` | `U+2715` / `U+2717` | `x` | Modal close button, clear/remove action |
| `✓` | `U+2713` | `check` | Valid / Selected / Apply |
| `▶` | `U+25B6` | `play` | Run / Execute / Start training / Next frame |
| `◀` | `U+25C0` | `chevron-left` | Previous page / Previous sample |
| `⏮` | `U+23EE` | `skip-back` | First sample |
| `⏭` | `U+23ED` | `skip-forward` | Last sample |
| `⏯` | `U+23EF` | `rotate-ccw` or `play` | Resume training |
| `📊` | `U+1F4CA` | `chart-column` | Distribution / Summary / Metrics |
| `📈` | `U+1F4C8` | `chart-line` or `trending-up` | Time series / Convergence / Min-Max |
| `📋` | `U+1F4CB` | `copy` or `clipboard` | Copy code / Duplicate project / Summary table |
| `💾` | `U+1F4BE` | `save` | Save / Update model |
| `📁` | `U+1F4C1` | `folder` | Directory / Upload archive / Quick upload |
| `📂` | `U+1F4C2` | `folder-open` | Load saved model (plain text inside `<option>`) |
| `📄` | `U+1F4C4` | `file-text` or `file-spreadsheet` | Data file / CSV file |
| `✏️` / `✏` | `U+270F` | `pencil` | Edit / Rename |
| `🔍` | `U+1F50D` | `search` | Check & infer shapes / Search / Inspect |
| `🔄` | `U+1F504` | `refresh-cw` | Refresh list / Reload |
| `🚀` | `U+1F680` | `rocket` or `play` | Build & save model / Apply template |
| `💡` | `U+1F4A1` | `lightbulb` or `info` | Callout hint / Info alert |
| `⚠️` / `⚠` | `U+26A0` | `triangle-exclamation-point` | Warning alert / Low confidence (plain text in `<option>`) |
| `🐍` | `U+1F40D` | `file-code` or `code` | Python / Keras code |
| `⏳` | `U+23F3` | `loader` (with `icon-spin`) or `clock` | Loading / Pending |
| `🔒` | `U+1F512` | `lock` | Frozen layer (`lock-open` for trainable/unfrozen) |
| `❄️` / `❄` | `U+2744` | `snowflake` | Layer freezing card header |
| `🧩` | `U+1F9E9` | `puzzle` or `blocks` | Custom layers |
| `🧠` | `U+1F9E0` | `brain` or `brain-circuit` | Neural network / Deep learning |
| `⚙️` / `⚙` | `U+2699` | `settings` or `cog` | Configure custom layer |
| `⛶` | `U+26F6` | `maximize-2` | Expand / Collapse canvas (`minimize-2` when expanded) |
| `✨` | `U+2728` | `sparkles` | Auto-layout nodes |
| `🧹` | `U+1F9F9` | `eraser` | Clear canvas |
| `📐` | `U+1F4D0` | `ruler` or `layers` | Output shape & parameters |
| `🔬` | `U+1F52C` | `microscope` | Inspect extracted features |
| `🧪` | `U+1F9EA` | `flask-conical` | Batch prediction / Evaluation |
| `🌌` | `U+1F30C` | `orbit` | IDEC deep clustering |
| `🌐` | `U+1F310` | `orbit` or `compass` | Open in Clustering & 3D PCA |
| `🎯` | `U+1F3AF` | `target` | Z-Score standardization |
| `🏷️` / `🏷` | `U+1F3F7` | `tag` | Classification / Label |
| `👥` | `U+1F465` | `users` or `layers` | Multi-signal overlay |
| `🚫` | `U+1F6AB` | `ban` | Clear selection / None |
| `🔥` | `U+1F525` | `flame` | Highest uncertainty (plain text in `<option>`) |
| `⭐` | `U+2B50` | `star` | Highest confidence (plain text in `<option>`) |
| `⇄` | `U+21C4` | `arrow-left-right` | Invert selection |
| `→` / `&rarr;` | `U+2192` | `arrow-right` | Next / Flow arrow |
| `←` / `&larr;` | `U+2190` | `arrow-left` | Prev / Back arrow |

---

## 4. CSS Component & Utility Vocabulary (`style.css` + `nodes.css`)

### App Shell & Page Structure
- `.page-header`: Flex header row (`justify-content: space-between`, `align-items: flex-end`, `margin-bottom: 20px`).
  - `.page-title`: `18px/24px`, `600`, `-0.02em`.
  - `.page-desc`: `13px/18px`, `--text-secondary`.
  - `.page-actions`: Right-aligned button cluster (`display: flex; gap: 8px; flex-wrap: wrap`).
- `.section`: `margin-bottom: 24px`.
- `.section-title`: `11px/14px` uppercase `500` `--text-muted` label (`margin-bottom: 8px`).

### Layout & Grid Utilities
- **Flex**: `.flex`, `.inline-flex`, `.flex-col`, `.flex-row`, `.flex-wrap`, `.flex-1`, `.flex-none`, `.shrink-0`, `.grow`, `.min-w-0`, `.items-start`, `.items-center`, `.items-end`, `.items-stretch`, `.items-baseline`, `.justify-start`, `.justify-center`, `.justify-end`, `.justify-between`, `.ml-auto`, `.mr-auto`.
- **Shorthand flex rows/stacks**:
  - `.row`: `display: flex; align-items: center; gap: 8px`
  - `.row-between`: `display: flex; align-items: center; justify-content: space-between; gap: 8px`
  - `.cluster`: `display: flex; flex-wrap: wrap; align-items: center; gap: 8px`
  - `.stack-xs` (`gap: 4px`), `.stack-sm` (`gap: 8px`), `.stack` (`gap: 12px`), `.stack-lg` (`gap: 16px`)
  - `.spacer`: `flex: 1 1 auto`
- **Grid**:
  - `.grid-2`, `.grid-3`, `.grid-4`, `.grid-5`, `.grid-6` (16px gap; combine with `.gap-2` or `.gap-3` for tighter grids)
  - `.grid-auto` (`minmax(240px, 1fr)`), `.grid-auto-sm` (`minmax(160px, 1fr)`)
  - `.grid-2-1` (`2fr 1fr`), `.grid-1-2` (`1fr 2fr`), `.grid-3-1` (`3fr 1fr`)
  - `.grid-sidebar` (`300px 1fr`), `.grid-sidebar-right` (`1fr 320px`)
  - `.split` (`300px 1fr`), `.split-right` (`1fr 320px`), `.split-3` (`260px 1fr 320px`), `.split-fill`
  - `.col-span-2`, `.col-span-3`, `.col-span-full`
- **Spacing (4px base: `1`=4px, `2`=8px, `3`=12px, `4`=16px, `5`=20px, `6`=24px, `8`=32px)**:
  - Margin: `.m-0`, `.mt-0..8`, `.mb-0..8`, `.ml-1..3`, `.mr-1..3`, `.my-2..4`
  - Padding: `.p-0..6`, `.px-2..4`, `.py-1..4`, `.pt-2..3`, `.pb-2..3`
- **Sizing & Scroll**:
  - `.w-full`, `.h-full`, `.w-auto`, `.min-h-0`, `.max-w-sm` (360px), `.max-w-md` (560px), `.max-w-lg` (800px)
  - `.scroll-y`, `.scroll-x`, `.overflow-hidden`, `.max-h-xs` (160px), `.max-h-sm` (240px), `.max-h-md` (320px), `.max-h-lg` (480px)
  - `.is-hidden` (`display: none`), `.relative`, `.sticky-top`

### Typography & Text Utilities
- Scale classes: `.text-micro` (11/14 uppercase), `.text-caption` (12/16), `.text-body` (13/18), `.text-subhead` (14/20 600), `.text-title` (18/24 600).
- Size/weight: `.text-xs` (11px), `.text-sm` (12px), `.text-md` (13px), `.text-lg` (14px), `.text-xl` (18px), `.fw-400`, `.fw-500`, `.fw-600`.
- Color: `.text-primary`, `.text-secondary`, `.text-muted`, `.text-accent`, `.text-success`, `.text-warning`, `.text-danger`, `.text-info`.
- Alignment & formatting: `.text-left`, `.text-center`, `.text-right`, `.uppercase`, `.mono` / `.font-mono`, `.tabular`, `.num` (mono + tabular + right-aligned), `.truncate`, `.nowrap`, `.break-all`, `.pre-wrap`.

### Buttons (`.btn`)
- Variants: `.btn-primary` (accent), `.btn-secondary` (surface-3 + border), `.btn-ghost` (transparent), `.btn-danger` (danger outline), `.btn-success` (success outline), `.btn-link`.
- Sizes: default (32px), `.btn-sm` (28px), `.btn-xs` (22px), `.btn-block` (`width: 100%`), `.btn-icon` (square icon-only button — **must** have `aria-label="..."`).
- Group: `<div class="btn-group"><button class="btn btn-secondary btn-sm">...</button>...</div>`.

### Forms
- `.form-group`, `.form-label` (with optional `<span class="form-label-meta">`), `.form-hint`, `.form-error`.
- `.form-row` (2-col grid), `.form-row-3` (3-col grid), `.form-inline`.
- `.form-control` (30px height, works on `<input>`, `<select>`, `<textarea>`), `.form-control-sm` (24px height), `.form-control.mono`.
- `.input-group` + `.input-addon`, `.input-icon` (leading `.icon` + `.form-control`).
- `.check` (`<label class="check"><input type="checkbox"> <span>Label</span></label>`).
- `.switch` (`<label class="switch"><input type="checkbox"><span class="switch-track"></span><span>Label</span></label>`).
- `.dropzone` (dashed upload drop area; toggle `.is-dragover` on drag).

### Cards, Panels, Tiles & Stats
- `.card` / `.panel`: `--surface-2`, `1px solid var(--border-subtle)`, `radius-lg` (6px), `padding: 16px`.
  - `.card-title`: Flex header row (`14px` `600`).
  - `.card-subtitle`: `12px` `--text-muted`.
- Structured card (`.card.card-flush`):
  - `.card-header` -> `.card-header-title` + `.card-actions`
  - `.card-body` (16px padding) or `.card-body-tight` (12px padding)
  - `.card-footer`
- Nested inset blocks: `.tile` / `.inset` (`--surface-1` bg, `1px solid var(--border-subtle)`, `12px` padding).
- Interactive selectable cards: `.tile-link` / `.option-card` (supports `.active` / `aria-checked="true"`), with `.tile-title` and `.tile-desc`.
- Telemetry stats:
  - `.stat-grid` -> `.stat` (or `.stat.stat-inset`) -> `.stat-label`, `.stat-value` (`.stat-value-sm`, `.stat-unit`), `.stat-meta` (`.stat-delta-up`, `.stat-delta-down`).
  - Legacy `.grid-cards > .card > .card-title + .stat-val + .stat-desc` is also styled automatically.
- Key-value list: `<dl class="kv"><dt>Label</dt><dd>Value</dd></dl>`.
- Chart containers: `.chart-box` (`min-height: 280px`), `.chart-box-sm` (`200px`), `.chart-box-lg` (`400px`).

### Tables
- Wrapper: `.table-wrap` or `.table-responsive`.
- Table: `<table class="data-table">` (or `.table`, `.table-compact`, `.table-mono`).
- Right-aligned numeric columns: `<th class="num">` and `<td class="num">`.
- Selected row: `<tr class="is-selected">`.

### Tabs, Badges, Alerts, Modals, Empty States
- **Tabs**: `<div class="tabs"><button type="button" class="tab active">Tab 1</button><button type="button" class="tab">Tab 2</button></div>`.
- **Badges**: `.badge`, `.badge-mono`, `.badge-accent`, `.badge-success`, `.badge-warning`, `.badge-danger`, `.badge-info`, `.shape-badge`, `.status-badge` (`.completed`/`.success`, `.running`/`.info`, `.warning`, `.failed`/`.error`, `.pending`), `.status-dot` (`.is-muted`, `.is-accent`, `.is-warning`, `.is-danger`, `.is-live`).
- **Alerts**: `<div class="alert alert-info|alert-success|alert-warning|alert-danger">{{ icon('info') }}<div class="alert-body"><span class="alert-title">...</span>...</div></div>`.
- **Callouts**: `<div class="callout callout-accent|callout-warning|callout-danger|callout-success">...</div>`.
- **Modals**:
  ```html
  <div id="my-modal" class="modal-overlay" role="dialog" aria-modal="true">
      <div class="modal modal-lg"> <!-- or modal-sm, modal-xl, or default 580px -->
          <div class="modal-header">
              <h3 class="modal-title">{{ icon('layers') }}<span>Modal Title</span></h3>
              <button type="button" class="btn btn-ghost btn-sm btn-icon modal-close" onclick="closeMyModal()" aria-label="Close">{{ icon('x') }}</button>
          </div>
          <div class="modal-body">...</div>
          <div class="modal-footer">...</div>
      </div>
  </div>
  ```
  - Note: `.modal-overlay` has `display: none` by default in CSS, so you do **not** need `style="display: none;"` on `.modal-overlay`. Opening via `el.style.display = "flex"` and closing via `el.style.display = "none"` works seamlessly.
- **Empty State**:
  ```html
  <div class="empty-state">
      {{ icon('database', 'icon-xl') }}
      <div class="empty-state-title">No datasets registered</div>
      <div class="empty-state-hint">Upload a CSV, NPY, or ZIP archive to get started.</div>
  </div>
  ```

---

## 5. Common Inline-Style Replacement Patterns

| Legacy Inline Style Pattern | Replace With Class / Component |
|---|---|
| `style="display: flex; gap: 8px; align-items: center;"` | `class="row"` or `class="cluster"` |
| `style="display: flex; justify-content: space-between; align-items: center;"` | `class="row-between"` |
| `style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;"` | `class="grid-2"` |
| `style="display: grid; grid-template-columns: 360px 1fr; gap: 1.5rem;"` | `class="grid-sidebar"` or `class="split"` |
| `style="margin-bottom: 1.5rem;"` | `class="mb-6"` (or `class="section"`) |
| `style="margin-bottom: 1rem;"` | `class="mb-4"` |
| `style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px;"` | `class="tile"` |
| `style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--accent-cyan); padding: ..."` | `class="callout callout-accent"` or `class="alert alert-info"` |
| `style="background: linear-gradient(135deg, #2563eb, #7c3aed); ..."` | `class="btn btn-primary"` (no gradients!) |
| `style="font-family: var(--font-mono); text-align: right;"` | `class="num"` |
| `style="font-family: var(--font-mono); font-size: 12px;"` | `class="mono text-sm"` |
| `style="color: var(--text-muted); font-size: 12px;"` | `class="text-muted text-sm"` |
| `style="display: none;"` on hidden file inputs / panels | `class="is-hidden"` or `hidden` attribute (keep JS `.style.display` toggles working: `.is-hidden` uses `display: none`, or if JS sets `el.style.display = "block"`, ensure removing `.is-hidden` or using HTML structure compatible with JS) |

> **Important compatibility note on `style="display: none;"`**:
> If existing page JS shows an element by setting `el.style.display = "block"` or `el.style.display = "grid"` (without removing a class), putting `class="is-hidden"` (which sets `display: none` in CSS) is overridden by inline `el.style.display = "block"` because inline styles have higher specificity than class selectors (`.is-hidden` does **not** use `!important`). However, if JS hides/shows by toggling `el.style.display = ""`, test that state carefully! For `.modal-overlay`, `display: none` is already the CSS default, and `el.style.display = "flex"` / `"none"` works directly.

---

## 6. Turkish-to-English Copy Checklist by File

Every Turkish or mixed Turkish/English string below must be converted to clean English while preserving all element IDs and JS hooks:

### `backend/app/templates/dashboard.html`
- `🪄 Sihirbaz` badge on NN Builder card -> remove gradient/emoji, use `<span class="badge badge-accent">Templates</span>` or `{{ icon('wand-sparkles') }}`.

### `backend/app/templates/projects.html`
- `⚠️ Silme Onayı` / confirmation modal copy -> `Confirm project deletion`
- `✏️ Düzenle`, `📋 Çoğalt`, `🗑️ Sil` -> `{{ icon('pencil') }} Edit`, `{{ icon('copy') }} Duplicate`, `{{ icon('trash') }} Delete`

### `backend/app/templates/data_manager.html`
- `📁`, `📄`, `🔄`, `⚡`, `🎯`, `📈`, `💡`, `🧠`, `✏️`, `🗑️`, `🚫`, `✓`, `✕`, `⇄`, `⏮`, `◀`, `▶`, `⏭` -> replace with Lucide icons per Section 3 table.
- Any Turkish labels/toasts in inline scripts (e.g., normalization scope hints, dataset rename/delete prompts) -> clean English (`"Per-waveform (normalize each signal independently)"`, `"Global dataset (normalize across entire dataset)"`, `"Rename dataset"`, `"Delete dataset"`).

### `backend/app/templates/signal_tagger.html`
- `📁 Quick Upload` -> `{{ icon('upload') }} <span>Quick upload</span>`
- `👥 Multi-Signal Overlay` -> `{{ icon('layers') }} <span>Multi-signal overlay</span>`

### `backend/app/templates/signal_processing.html`
- Convert inline layout/form styles to `.grid-sidebar`, `.card`, `.tile`, `.callout`, `.data-table`, `.num`.
- Ensure all normalization and pipeline descriptions are English-only.

### `backend/app/templates/feature_extraction.html`
- `⚡ Extract Layer Features` -> `{{ icon('zap') }} <span>Extract features</span>`
- `📥 CSV İndir` -> `{{ icon('download') }} <span>CSV</span>`
- `🔬 İncele` -> `{{ icon('microscope') }} <span>Inspect</span>`
- `✏️` / `🗑️` action buttons -> `{{ icon('pencil') }}` / `{{ icon('trash') }}` with `aria-label`
- JS prompts/confirms/toasts (`"Yeni Özellik / Katman Adını Girin:"`, `"Özellik adı başarıyla güncellendi!"`, `"Bu kayıtlı özellik çıkarımını silmek istediğinize emin misiniz?"`, `"Özellik çıkarımı başarıyla silindi!"`) -> English (`"Enter new feature extraction name:"`, `"Feature extraction renamed"`, `"Delete this saved feature extraction?"`, `"Feature extraction deleted"`).

### `backend/app/templates/workflow_builder.html` & `backend/app/static/js/workflow_editor.js`
- `🪄 Model Sihirbazı (NN Builder)` -> `{{ icon('wand-sparkles') }} <span>NN Builder</span>`
- `🪄 Model Sihirbazı (Autoencoder / U-Net) →` -> `{{ icon('wand-sparkles') }} <span>Open Architecture Wizard</span> {{ icon('arrow-right', 'icon-xs') }}`
- In `workflow_editor.js`:
  - `Scope / Normalizasyon Kapsamı` -> `Normalization scope`
  - `Per-Waveform / Kendi İçinde (Her sinyal bağımsız ölçeklenir)` -> `Per-waveform (scale each signal independently)`
  - `Global Dataset (Tüm veri setine göre)` -> `Global dataset (scale across all samples)`
  - `💡 Kendi İçinde Normalizasyon (per_signal): ...` -> `.callout.callout-accent` with English explanation: `Per-waveform normalization scales each signal by its own mean and range to remove amplitude offsets.`
  - `True (Aktif)` / `False (Devre Dışı)` -> `True (Enabled)` / `False (Disabled)`

### `backend/app/templates/nn_builder.html` & `backend/app/static/js/nn_editor.js`
- Delete the large inline `<style>` block in `nn_builder.html` (all toolbar, wizard, IDEC, and layout rules live in `nodes.css`).
- Toolbar & header buttons:
  - `🪄 Model Sihirbazı` / `🪄 Sihirbaz` -> `{{ icon('wand-sparkles') }} <span>Architecture wizard</span>` / `<span>Wizard</span>`
  - `🧩 Özel Katmanlar` -> `{{ icon('blocks') }} <span>Custom layers</span>`
  - `🔍 Check & Infer Shapes` / `🔍 Boyut Doğrula` -> `{{ icon('search') }} <span>Infer shapes</span>` / `<span>Validate</span>`
  - `🚀 Build & Save Model` / `💾 Modeli Derle & Kaydet` -> `{{ icon('play') }} <span>Build &amp; save</span>`
  - `📂 Kayıtlı Model Yükle...` (inside `<select>`) -> `Load saved model...` (plain text, no emoji)
  - `💾 Güncelle` -> `{{ icon('save') }} <span>Update</span>`
  - `🗑️ Sil` -> `{{ icon('trash') }} <span>Delete</span>`
  - `📥 İçe Aktar` -> `{{ icon('upload') }} <span>Import</span>`
  - `📤 JSON İndir` -> `{{ icon('download') }} <span>Export JSON</span>`
  - `✨ Otomatik Hizala` -> `{{ icon('sparkles') }} <span>Auto-layout</span>`
  - `🧹 Temizle` -> `{{ icon('eraser') }} <span>Clear</span>`
  - `⛶ Genişlet` / `Daralt` -> `{{ icon('maximize-2') }} <span>Expand</span>` / `{{ icon('minimize-2') }} <span>Collapse</span>`
  - `🐍 Python Kodu` -> `{{ icon('file-code') }} <span>Python code</span>`
  - `📊 Model Özeti` -> `{{ icon('table') }} <span>Summary</span>`
- Palette category titles (`Core & IO (Giriş / Çıkış)`, `1D CNN & Deconv (Evrişim)`, `Recurrent (Zaman Serisi)`, `Attention & Transformer`, `Reshape & Regularization`, `Merge & Arithmetic`, `Clustering & Custom (Özel)`): English only (`Core & IO`, `1D Convolution`, `Recurrent`, `Attention & Transformer`, `Reshape & Regularization`, `Merge & Arithmetic`, `Clustering & Custom`).
- Modals (`wizard-modal`, `custom-layer-modal`, `build-model-modal`, `python-code-modal`, `model-summary-modal`):
  - Convert all Turkish headings, descriptions, template cards, IDEC preview labels, and toast messages to concise English.
- In `nn_editor.js`:
  - Context menu `"🪄 Model Mimarisi Sihirbazı..."` -> `"Architecture wizard..."`
  - Inspector `"📐 Çıktı Boyutu & Parametre"` -> `"Output shape & parameters"`, `"Parametreler:"` -> `"Parameters:"`
  - `"Özel Düğüm / Katman Adı (Custom Node Name)"` -> `"Layer name"`, placeholder `"e.g. encoder_bottleneck"`
  - `"Katman Kodu (Code)"` -> `"Layer code"`
  - Toasts (`"Hizalanacak düğüm bulunamadı!"`, `"Mimari düğümleri otomatik hizalandı!"`, `"Geçersiz JSON formatı: "`) -> English (`"No nodes to align"`, `"Nodes aligned"`, `"Invalid JSON format: "`).

### `backend/app/templates/training.html` & `backend/app/static/js/training_monitor.js`
- `⏯ Eğitime Devam Et (Resume)` -> `{{ icon('rotate-ccw') }} <span>Resume training</span>`
- `▶ Launch Training Job` -> `{{ icon('play') }} <span>Start training</span>`
- `🪄 Model Sihirbazı` -> `{{ icon('wand-sparkles') }} <span>Architecture wizard</span>`
- IDEC card: `IDEC Kümeleme Eğitimi Aktif (Autoencoder / IDEC)` -> `Enable IDEC deep clustering training`, `Küme Sayısı (K)` -> `Clusters (k)`, `Kümeleme Ağırlığı (γ)` -> `Clustering weight (γ)`, `AE Ön-Eğitim Epoch` -> `Pretrain epochs`, `Hedef Güncelleme (Epoch)` -> `Update interval (epochs)`.
- Layer Freezing card: `❄️ Katman Dondurma ve Seçimli Eğitim (Layer Freezing)` -> `{{ icon('snowflake') }} <span>Layer Freezing &amp; Selective Training</span>`, `Eğitilebilir:` -> `Trainable:`, `Dondurulmuş:` -> `Frozen:`, `🔒 Frozen` / `🔓 Trainable` -> `{{ icon('lock') }} Frozen` / `{{ icon('lock-open') }} Trainable`.
- In `training_monitor.js`:
  - Use `SignalML.chartTheme()` (remove hardcoded dark colors `#1e293b`, `#0f172a`, `#f8fafc` in Highcharts init so light/dark theme switching works automatically).
  - Axis titles `"Loss (Kayıp)"` -> `"Loss"`, `"Accuracy (Doğruluk)"` -> `"Accuracy"`.
  - IDEC summary cards (`Küme Sayısı (K)`, `Ortalama Küme Güveni`, `Rekonstrüksiyon MSE`, `Rekonstrüksiyon MAE`, `📊 Küme Dağılımı`, `🌐 Clustering & 3D PCA Sayfasında İncele`, `🪄 NN Model Sihirbazı`) -> English with Lucide icons (`SignalML.icon(...)`).

### `backend/app/templates/clustering.html` & `backend/app/static/js/pca_visualizer.js`
- Page description & buttons:
  - `📥 Kümeleme & Olasılıkları CSV İndir (Input ID ile)` -> `{{ icon('download') }} <span>Export cluster CSV</span>`
  - `▶ Run Clustering & PCA` -> `{{ icon('play') }} <span>Run clustering &amp; PCA</span>`
- Cluster probability & uncertainty inspector:
  - `📊 Sinyal Küme Olasılıkları ve Belirsizlik İnceleyici` -> `Cluster Probability & Uncertainty Inspector`
  - `Küme Filtresi:` -> `Cluster:`, `Tüm Kümeler (All)` -> `All clusters`
  - `Sıralama:` -> `Sort by:`
  - `<option>` items (plain text, NO emojis!):
    - `Lowest confidence (boundary samples first)`
    - `Highest uncertainty (entropy first)`
    - `Highest confidence (core samples first)`
    - `Sample index (ascending)`
  - Table headers: `Atanan Küme` -> `Cluster`, `Güven (Confidence)` -> `Confidence`, `Küme Olasılık Dağılımı ($q_{ij}$)` -> `Cluster probability distribution (q_ij)`
  - Pagination: `◀ Önceki` -> `{{ icon('chevron-left') }} <span>Prev</span>`, `Sonraki ▶` -> `<span>Next</span> {{ icon('chevron-right') }}`
  - `Kayıtlı Kümeleme Sonuçları (Saved Clustering Runs)` -> `Saved Clustering Runs`
- In `pca_visualizer.js`:
  - Read theme palette via `SignalML.getThemePalette()` so both the 2D Highcharts scatter plot and the 3D HTML5 `<canvas>` projection adapt cleanly to light and dark themes (re-render on `signalml:themechange`).

### `backend/app/templates/model_evaluation.html`
- `🧪 Batch Tahmin CSV Dışa Aktarma` -> `{{ icon('flask-conical') }} <span>Batch Prediction CSV Export</span>`
- `Değerlendirme Veri Kümesi (Evaluation Dataset)` -> `Evaluation dataset`
- `📥 CSV Olarak İndir` -> `{{ icon('download') }} <span>Download predictions CSV</span>`
- Toasts and error messages -> English.

### `backend/app/templates/exports.html`
- `📥 Model İçe Aktar (Import Model)` -> `{{ icon('upload') }} <span>Import Model (.keras / .json)</span>`
- `📂 Dosya Seç & Yükle` -> `{{ icon('folder-open') }} <span>Select file &amp; upload</span>`
- `🐍 .keras İndir` -> `{{ icon('download') }} <span>.keras</span>`
- `📥 .json Mimari` -> `{{ icon('file-code') }} <span>.json</span>`
- `🗑️` -> `{{ icon('trash') }}`
- Status messages (`✓ Başarıyla yüklendi`, `✗ Hata`) -> use `SignalML.icon('circle-check')` / `SignalML.icon('circle-x')` + English text.

---

## 7. Verification Checklist for Each Migrated Page

Before committing your page group:
1. **Grep check**:
   - `grep -n 'style="' backend/app/templates/YOUR_PAGE.html` -> 0 static inline styles.
   - Check for any remaining emojis or Turkish characters (`[çğıöşüÇĞİÖŞÜ]`) in your assigned templates and JS files.
2. **Visual check (both themes)**:
   - Render your pages in headless Chromium at `1440x900` with `?theme=dark` and `?theme=light`:
     ```bash
     CHROME="/workspace/Academic Paper App - Advanced/chrome/linux-154.0.8037.57/chrome-linux64/chrome"
     "$CHROME" --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=1440,900 --virtual-time-budget=4000 --screenshot=FILE URL
     ```
   - Inspect screenshots with `read_file` to verify alignment, contrast, borders, tables, charts, and icons.
3. **Test suite**:
   - Run `cd /workspace/SignalML/backend && PYTHONPATH=. python3 -m pytest -q` and ensure all 83 tests pass.
