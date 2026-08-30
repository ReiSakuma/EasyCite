# EasyCite v15 Product MVP Release Candidate

v15 is the first version organized as a product MVP rather than an experiment snapshot. Earlier versions focused on capability depth: Zotero retrieval, evidence verification, LangGraph-style workflow, feedback memory, BibTeX quality, and eval support. v15 keeps those capabilities and improves the installation, first-run, and packaging experience.

## Product Goal

The target user should be able to install a VSIX, open a LaTeX project, complete a guided setup, and use EasyCite without reading internal development notes or manually editing JSON for the common path.

The product MVP is not a marketplace-ready commercial release yet. It is a release candidate suitable for demos, interviews, and controlled external testing.

## User-Facing Scope

- Run a setup wizard from the command palette.
- Detect and install missing Python dependencies into extension private storage.
- Configure Zotero, Better BibTeX, model API, embedding, and PDF reading through VS Code UI.
- Restrict Zotero retrieval to a selected collection.
- Scan a TeX project for likely citation-needed claims.
- Open a stable candidate review webview.
- Review recommendation evidence directly in the webview.
- Insert one or more citations and update the selected or discovered BibTeX file.
- See clear Tool Health status when something is misconfigured.

## Developer Scope

The following tools are retained but hidden unless `citation-agent.developerMode` is true:

- Run Trace
- Agent State
- Eval Bad Case export
- Restart Server

This keeps the normal command palette focused while preserving observability for debugging and regression work.

## Packaging Changes

`package.json` now emits v15-specific artifacts:

- `npm run vsce-package` -> `easycite-v20.vsix`
- `npm run vsce-package-windows` -> `easycite-v20-Windows.vsix`

The VSIX ignore files now exclude local package artifacts, eval reports, cache folders, and temporary Python caches. The Windows ignore file still excludes Linux-native Python wheels so Windows installs can populate private dependencies on the user machine.

## First-Run Behavior

v15 uses versioned setup global-state keys:

- `citation-agent.setupCompleted.v15`
- `citation-agent.setupPromptDismissed.v15`

This means users who tested earlier versions will still receive the v15 setup prompt once. The prompt is intentionally non-blocking: users can postpone it, but the command remains available as `EasyCite: 设置向导`.

## MVP Acceptance Checklist

- Fresh VS Code profile can install the VSIX.
- Opening a LaTeX workspace prompts for setup once.
- Setup wizard can check Python runtime and install private dependencies.
- Tool Health clearly reports Zotero, Better BibTeX, LLM, embedding, and PDF capability status.
- Candidate review works from selected text and from lightbulb diagnostics.
- Multi-citation insertion writes valid `\cite{...}` before English punctuation.
- BibTeX write plan selects an existing project BibTeX file when available.
- Developer tools remain accessible only after enabling developer mode.

## Known Product Gaps After v15

- The extension is not yet published to the VS Code Marketplace.
- Offline model downloads still depend on the selected provider and network availability.
- Zotero launching behavior is platform-dependent and may still bring Zotero to the foreground on some Windows setups.
- Full TeX project parsing is pragmatic rather than a complete LaTeX AST implementation.
- Retrieval quality still depends on Zotero metadata/PDF quality and user collection scoping.
