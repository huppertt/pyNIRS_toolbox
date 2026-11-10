# Publishing this wiki

The Markdown pages in this `wiki/` directory are stored in the main
`pyNIRS_toolbox` repository as a reviewable source copy. A GitHub Wiki is a
separate Git repository and does not automatically update when the main
repository is pushed.

## One-time setup

1. Open the main repository on GitHub and select **Wiki**.
2. If prompted, create the initial wiki page. GitHub then provides the wiki
   clone URL.
3. Clone the wiki repository separately. Its URL typically ends in
   `.wiki.git`, for example:

   ```bash
   git clone https://github.com/OWNER/REPOSITORY.wiki.git
   ```

4. Copy the Markdown pages from this directory into that clone, keeping the
   GitHub Wiki page filenames (for example `Home.md`, `Getting-Started.md`,
   and `_Sidebar.md`).
5. Review the rendered pages, then commit and push to the wiki repository's
   default branch.

## Later updates

Keep the `documentation/wiki/` copy in the main project as the canonical
reviewable source, then copy approved changes into the wiki clone and push
them separately. Alternatively, the team can choose to edit the wiki clone
directly and periodically sync those changes back into this source folder.

GitHub's documentation:
[Adding or editing wiki pages](https://docs.github.com/articles/adding-or-editing-wiki-pages).
