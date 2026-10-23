# Final write-up

The LaTeX template for the final research report. It uses the same report style as the proposal.

## Files

| File | What goes in it |
|---|---|
| `main.tex` | Title page, settings and chapter order. Only edit it to switch the guidance boxes off. |
| `chapters/00_frontmatter.tex` | Abstract, declaration, acknowledgements |
| `chapters/01_introduction.tex` | Motivation, research question, H1–H3, contributions |
| `chapters/02_background.tex` | Mostly copied from the proposal, plus two new sections |
| `chapters/03_data.tex` | Sources, features, targets, splits, integrity checks |
| `chapters/04_methodology.tex` | Alpha Agent, Beta Agent, Risk Engine, backtest, statistical tests |
| `chapters/05_results.tex` | **Tables already filled in with the real numbers.** You write the text. |
| `chapters/06_discussion.tex` | Why the results came out this way, deviations from the proposal, limitations |
| `chapters/07_conclusion.tex` | Conclusion and future work |
| `chapters/appendix.tex` | Hyperparameters, reproducibility, extra results |
| `references.bib` | Paste the proposal's `.bib` entries here |

- Green **WRITE** boxes tell you what each section needs.
- Red **[TBC: …]** marks a gap still to fill.
- Before submitting, set `\guidefalse` in `main.tex` so the boxes disappear. Search the PDF for "TBC" to make sure none are left.

## Compiling

You can compile it any of these ways:
- **Overleaf:** upload `WriteUp/` and the `figures/` folder, keeping `figures/` next to `main.tex`.
- **Locally with Tectonic:** run `tectonic main.tex` in this folder.
- **Locally with latexmk:** run `latexmk -pdf main.tex` in this folder.

Figures are picked up straight from `../figures/`, so if a chart is regenerated the report updates automatically.

## Suggested writing order (due 2 Nov)

Write the easiest parts first and the abstract last.

1. **Methodology + Data** (23–25 Oct). You know this best, and the proposal gives you a head start.
2. **Results** (26–27 Oct). The tables are done; write 1–2 paragraphs per table.
3. **Discussion + Conclusion** (28–29 Oct).
4. **Introduction + Background** (30 Oct). Adapt from the proposal and literature review.
5. **Abstract, proofread, send to Clint** (31 Oct). Then use 1–2 Nov for his comments.

If Clint's extension is needed, the slip goes into step 5.
