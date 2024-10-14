## Report

We are currently evaluating how we report data-science work - both at the project-level and the feature-level.

You should write reports in markdown putting them in `reports` and referencing plots in `reports/figures`.

We are [experimenting](../roadmap#Reporting) with a toolchain using [pandoc](https://pandoc.org/) to generate HTML and PDF (LaTeX) outputs
from a single ([pandoc flavoured](https://pandoc.org/MANUAL.html#pandocs-markdown)) markdown file, including facilitating the trivial
inclusion of interactive [altair](https://altair-viz.github.io/index.html) plots within HTML outputs.
