# SignalML Design System

Aesthetic: technical instrument / pro tool (Linear + Grafana + Blender). Dark and light themes. English only.
Sources: Anthropic `frontend-design` skill, pbakaus/impeccable, ui-ux-pro-max skill, Radix Colors, Linear, Grafana Saga, Vercel Geist.

## 1. Anti-slop rules
1. No emoji anywhere in UI (templates, JS-generated HTML, toasts, alerts). Use Lucide icons or plain text.
2. No gradients on UI chrome, no glow shadows (`box-shadow: 0 0 Npx color`), no blue-to-purple anything.
3. No card soup. Depth comes from neutral surface steps + 1px hairline borders, not drop shadows.
4. One chromatic accent (`--accent`) used for primary actions, focus and selection only. ~90% of the UI is neutral.
5. No entrance animations or bouncy hovers. Transitions 100-150ms ease-out, state feedback only.
6. Numbers (metrics, shapes, sample counts, epochs, IDs) use `--font-mono` with `tabular-nums`.
7. No inline `style="..."` in templates except truly dynamic values set by JS. Use utility/component classes.
8. Copy is concise and technical, active verbs ("Train model", "Export CSV"). Sentence case for buttons and headings.

## 2. Tokens
Theme set on `<html data-theme="dark|light">`, persisted in `localStorage("signalml_theme")`, default follows `prefers-color-scheme`.

### Dark (default)
| Token | Value | Use |
|---|---|---|
| --surface-1 | #111113 | app background |
| --surface-2 | #18191b | header, panels, cards, table head |
| --surface-3 | #212225 | popovers, inputs on panels, row hover |
| --surface-4 | #272a2d | hover fill |
| --surface-5 | #2e3135 | pressed / selected |
| --border-subtle | #272a2d | dividers, row lines |
| --border-default | #363a3f | inputs, cards |
| --border-strong | #43484e | hover borders, popovers |
| --text-primary | #edeef0 | |
| --text-secondary | #b0b4ba | |
| --text-muted | #777b84 | |
| --accent / -hover / -subtle / -ring | #0090ff / #3b9eff / #0d2847 / rgba(0,144,255,.35) | |
| --success / -subtle / -border | #30a46c / #132d21 / #28684a | |
| --warning / -subtle / -border | #ffc53d / #302008 / #714f19 | |
| --danger / -subtle / -border | #e5484d / #3b1219 / #8c333a | |
| --info / -subtle / -border | #3b9eff / #0d2847 / #205d9e | |

### Light
| Token | Value |
|---|---|
| --surface-1..5 | #fcfcfd, #f9f9fb, #f0f0f3, #e8e8ec, #e0e1e6 |
| --border-subtle/default/strong | #e8e8ec, #d9d9e0, #cdced6 |
| --text-primary/secondary/muted | #1c2024, #60646c, #80838d |
| --accent / -hover / -subtle / -ring | #0090ff / #0588f0 / #e6f4fe / rgba(0,144,255,.25) |
| --success | #218358 / #e6f6eb / #8eceaa |
| --warning | #ab6400 / #fff7c2 / #e9c162 |
| --danger | #ce2c31 / #feebec / #f4a9aa |
| --info | #0d74ce / #e6f4fe / #8ec8f6 |

Chart series palette: #0090ff, #30a46c, #ffc53d, #e5484d, #8e4ec6, #00b8d9, #f76b15, #a0a7b4.

## 3. Type, spacing, geometry
- Fonts (local woff2 in `static/vendor/fonts`): `Inter` (UI), `JetBrains Mono` (code, metrics).
- Scale: micro 11/14 500 (table headers, badges, uppercase +0.04em) · caption 12/16 · body 13/18 · subhead 14/20 600 · title 18/24 600 -0.02em.
- Spacing (4px base): 4, 8, 12, 16, 20, 24, 32.
- Radius: sm 3px (badges) · md 4px (buttons, inputs) · lg 6px (cards, menus) · xl 8px (modals). Never pill buttons or 16px+.
- Elevation: structural elements have no shadow. Overlays only: `0 4px 16px -2px rgba(0,0,0,.4)` dark / `rgba(0,0,0,.08)` light.

## 4. Components
- **Header**: 44px, `--surface-2`, bottom hairline. Brand mark (flat, no gradient) + grouped dropdown nav + right tools (project switcher, engine status CPU/GPU, theme toggle).
  - Workspace: Dashboard, Projects, Exports
  - Data: Data Manager, Signal Tagger, Signal Processing, Feature Extraction
  - Modeling: Visual Workflow, NN Builder, Training
  - Analysis: Clustering & PCA, Evaluation
  - Group trigger shows active state when a child page is active. Menu: `--surface-3`, `--border-strong`, radius 6, padding 4, items 28px tall with 14px icon. Keyboard accessible (Enter/Space/Escape, arrow keys), closes on outside click.
- **Buttons**: 28px (sm) / 32px, padding 0 10px, radius 4, 13px 500. Variants: primary (accent), secondary (surface-3 + border), ghost, danger (outline danger). Icon-only buttons are square with `aria-label`.
- **Inputs/selects/textarea**: 30px, 13px, `--surface-1` bg on panels, `--border-default`, radius 4. Focus: border accent + `0 0 0 2px var(--accent-ring)`.
- **Tables**: rows 28-30px, cells 4px 10px, sticky header `--surface-2` 11px uppercase 600, numbers mono right-aligned, hover `--surface-3`.
- **Cards/panels**: `--surface-2`, `1px --border-subtle`, radius 6, no shadow; header row with bottom hairline.
- **Tabs**: segmented control, `--surface-1` track, active `--surface-3` + hairline.
- **Modals**: backdrop rgba(0,0,0,.6); surface-2, border-strong, radius 8, max-width 580.
- **Badges**: 18px, 0 6px, radius 3, 11px 500; neutral or status-subtle.
- **Toasts**: bottom-right, surface-3, border-strong, status shown by a left icon in status color.
- **Empty states**: 24px muted icon, 13px 600 title, 12px muted hint, one compact action.

## 5. Icons (Lucide, local)
- Sprite: `/static/vendor/lucide/sprite.svg` (lucide-static 1.69.0, ISC). Symbol ids are plain kebab-case names (`#play`, `#settings`).
- Jinja: `{% from "_macros.html" import icon %}` then `{{ icon('play') }}` / `{{ icon('play', 'icon-lg') }}`.
- JS: `SignalML.icon('play')` returns the same SVG markup string.
- CSS `.icon { width:16px; height:16px; stroke:currentColor; stroke-width:1.75; fill:none; stroke-linecap:round; stroke-linejoin:round; flex-shrink:0 }` — required, sprite symbols have no stroke attributes.
- Decorative icons get `aria-hidden="true"`; icon-only controls need `aria-label`.

## 6. Charts and node canvas
- Highcharts: `SignalML.chartTheme()` applies transparent bg, Inter labels, mono axis labels 11px, grid `--border-subtle`, tooltip surface-3, no shadow, series palette above. Re-applied on theme change.
- LiteGraph: node bg surface-2, title bar surface-3, border border-default, text primary/secondary, links border-strong, connecting link accent, shadows off, canvas bg surface-1 with subtle dot/grid.
