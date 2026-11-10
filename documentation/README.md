# pyNIRS_Toolbox documentation workspace

This folder contains two complementary draft formats:

- `pyNIRS_Toolbox_User_Manual.docx` — editable Word manuscript. It is
  intentionally excluded from Git until the author approves it.
- `wiki/` — Markdown pages suitable for a GitHub Wiki. These files are part of
  the main repository as an editable source copy; publishing to GitHub Wiki is
  a separate step (see [Publishing the wiki](wiki/Publishing.md)).

## Editing workflow

Edit the Word manuscript as the authoring copy. Keep paragraphs beginning with
`[SYNC HOOK — DO NOT EDIT]` intact; these stable IDs are synchronization
anchors for a later pass that will update the corresponding Python
documentation and wiki pages. Edit the surrounding user-facing text and the
`[AUTHOR: ...]` prompts freely. The hooks can be removed for final PDF
production.

The wiki is an initial searchable scaffold, not yet a verbatim copy of the
manual. Once the manual is edited, the approved changes can be reflected in the
relevant Python docstrings, README, and wiki pages.

## GitHub Wiki publishing

GitHub Wiki pages are Markdown files stored in a separate repository from the
main project repository. The `wiki/` folder here is the source to review and
copy; it does not automatically publish when the main repository is pushed.
See [Publishing.md](wiki/Publishing.md) for the upload workflow.
