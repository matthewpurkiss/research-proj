# Final write-up

The final research report in **IEEE conference format** (`IEEEtran`, two columns, numbered references). It's written as a full draft, ready for proofreading. `Final_Report_Draft.pdf` is the compiled copy.

**Length:** about 10 pages from the title to the end of the Conclusion, against a target of 8–12. References, the appendix and the declarations don't count.

## Files

| File | Section |
|---|---|
| `sections/00_abstract.tex` | Abstract and index terms |
| `sections/01_introduction.tex` | I. Introduction, hypotheses, contributions |
| `sections/02_related_work.tex` | II. Related Work |
| `sections/03_data.tex` | III. Data |
| `sections/04_method.tex` | IV. Methodology |
| `sections/05_setup.tex` | V. Experimental Setup, including Use of AI tools |
| `sections/06_results.tex` | VI. Results |
| `sections/07_discussion.tex` | VII. Discussion |
| `sections/08_conclusion.tex` | VIII. Conclusion |
| `sections/appendix.tex` | Features, models and settings, reproducibility |
| `sections/99_declaration.tex` | Wits declaration and AI declaration |
| `references.bib` | References |

## Before submitting

- Fill in the red **[TBC]** items in the declaration: the dates, the signature, and Quillbot if you used it.
- Check the ticks on the AI declaration.
- Check the references added for the final report (second group in `references.bib`) against the originals.
- After your own rewrite, recompile and check the page count.

## Compiling

You can compile it any of these ways:
- **Overleaf:** upload `WriteUp/` and the `figures/` folder, keeping `figures/` next to `main.tex`.
- **Tectonic:** run `tectonic main.tex` in this folder.
- **latexmk:** run `latexmk -pdf main.tex` in this folder.
