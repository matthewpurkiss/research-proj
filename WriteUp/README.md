# Final write-up

The LaTeX template for the final research report. It uses the same wording as the proposal, but the format is a compact article, because the target is **8–12 pages of content**: Introduction through Conclusion, excluding the title page, abstract, declaration, contents, references and appendix.

## Files and page budget

| File | Section | Pages |
|---|---|---|
| `sections/00_frontmatter.tex` | Abstract, declaration, acknowledgements, contents | not counted |
| `sections/01_introduction.tex` | Motivation, research question, H1–H3, contributions | ~1 |
| `sections/02_background.tex` | Condensed from the proposal: one paragraph per topic | ~1.5–2 |
| `sections/03_data_method.tex` | Data, splits, framework, Risk Engine, backtest, statistics | ~3 |
| `sections/04_results.tex` | **Tables and figures already filled in with the real numbers.** You write the text. | ~3–4 |
| `sections/05_discussion.tex` | Why, deviations from the proposal, limitations | ~1.5 |
| `sections/06_conclusion.tex` | Answer, takeaway, future work | ~0.5 |
| `sections/appendix.tex` | Feature list, hyperparameters, extra figure, reproducibility | not counted |
| `references.bib` | Paste the proposal's `.bib` entries here | not counted |

The fixed content (5 tables and 3 figures) already takes about 3.5 pages, which leaves roughly 6–8 pages of your own writing. Detail that doesn't fit belongs in the appendix.

## Markers

- Green **WRITE** boxes say what each section needs.
- Red **[TBC: …]** marks a gap still to fill.
- Before submitting:
  - set `\guidefalse` in `main.tex` so the boxes disappear;
  - check the page count with the boxes off;
  - search the PDF for "TBC" to make sure none are left.

## Compiling

You can compile it any of these ways:
- **Overleaf:** upload `WriteUp/` and the `figures/` folder, keeping `figures/` next to `main.tex`.
- **Tectonic:** run `tectonic main.tex` in this folder.
- **latexmk:** run `latexmk -pdf main.tex` in this folder.

## Suggested writing order (due 2 Nov)

1. **Data and Methodology** (23–24 Oct)
2. **Results** (25–26 Oct)
3. **Discussion and Conclusion** (27–28 Oct)
4. **Introduction and Background** (29–30 Oct)
5. **Abstract, proofread, send to Clint** (31 Oct). Then use 1–2 Nov for his comments.
