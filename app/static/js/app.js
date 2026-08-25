/**
 * Portfolio Analytics — results page charts.
 *
 * Each chart pairs a <canvas> with a <script type="application/json"> data
 * island rendered by the template. Sections the analysis skipped simply have
 * neither, so nothing here needs to know why a chart is missing.
 */

(function () {
  "use strict";

  const BAR = { maxBarThickness: 24, borderRadius: 4, borderSkipped: "start" };
  const ROW_HEIGHT = 28;
  const AXIS_CHROME = 88;

  const decimal = (value) => Number(value).toFixed(4);
  const percent = (value) => (Number(value) * 100).toFixed(1) + "%";
  const ranked = (source) => Object.entries(source || {}).sort((a, b) => b[1] - a[1]);
  const bar = (label, data, backgroundColor) => ({ label, data, backgroundColor, ...BAR });

  function readTheme() {
    const style = getComputedStyle(document.documentElement);
    const token = (name) => style.getPropertyValue(name).trim();
    return {
      series1: token("--chart-series-1"), series2: token("--chart-series-2"),
      negative: token("--chart-negative"), grid: token("--color-border"),
      ink: token("--color-muted"), text: token("--color-text"), surface: token("--color-surface"),
    };
  }

  function axis(theme, stacked, isValueAxis, format) {
    const ticks = { color: theme.ink, autoSkip: false };
    if (isValueAxis) {
      ticks.callback = format;
    }
    const grid = isValueAxis ? { color: theme.grid, drawTicks: false } : { display: false };
    return { stacked, grid, border: { display: false }, ticks };
  }

  function baseOptions(theme, { horizontal, stacked = false, legend = false, format }) {
    const valueAxis = horizontal ? "x" : "y";
    return {
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 400 },
      interaction: { mode: "index", intersect: false },
      scales: {
        [valueAxis]: axis(theme, stacked, true, format),
        [horizontal ? "y" : "x"]: axis(theme, stacked, false, format),
      },
      plugins: {
        legend: { display: legend, position: "top", align: "start",
          labels: { color: theme.text, boxWidth: 12, boxHeight: 12, padding: 16 } },
        // Value leads, series name follows — the reader has the series already.
        tooltip: {
          callbacks: { label: (ctx) => `${format(ctx.parsed[valueAxis])}  ·  ${ctx.dataset.label}` },
        },
      },
    };
  }

  function bars(labels, datasets, theme, config) {
    return { rows: labels.length, labels, datasets, options: baseOptions(theme, config) };
  }

  function rankedBars(entries, label, color, theme, format) {
    const labels = entries.map(([ticker]) => ticker);
    const values = entries.map(([, value]) => value);
    return bars(labels, [bar(label, values, color)], theme, { horizontal: true, format });
  }

  function weights(data, theme) {
    const entries = ranked(data).filter(([, weight]) => weight > 0);
    return rankedBars(entries, "Weight", theme.series1, theme, percent);
  }

  function beta(data, theme) {
    return rankedBars(ranked(data.betas), "Beta", theme.series1, theme, decimal);
  }

  function alpha(data, theme) {
    const entries = ranked(data.alphas);
    // Color carries the sign, not the rank — this is a diverging scale.
    const colors = entries.map(([, value]) => (value >= 0 ? theme.series1 : theme.negative));
    return rankedBars(entries, "Alpha", colors, theme, decimal);
  }

  function risk(data, theme) {
    const datasets = [
      bar("VaR", [data.var_90, data.var_95, data.var_99], theme.series1),
      bar("CVaR", [data.cvar_90, data.cvar_95, data.cvar_99], theme.series2),
    ];
    return bars(["90%", "95%", "99%"], datasets, theme, { horizontal: false, legend: true, format: decimal });
  }

  function utility(data, theme) {
    const entries = ranked(data.utility);
    const tickers = entries.map(([ticker]) => ticker);
    const best = tickers.map((t) => (t in data.max_utility ? data.max_utility[t] : null));
    const datasets = [
      bar("Utility (U)", entries.map(([, value]) => value), theme.series1),
      bar("Max utility (U*)", best, theme.series2),
    ];
    return bars(tickers, datasets, theme, { horizontal: true, legend: true, format: decimal });
  }

  function riskSplit(data, theme) {
    const rows = Object.keys(data.systematic_risks || {}).map((ticker) => {
      const systematic = data.systematic_risks[ticker];
      const specific = data.firm_specific_risks[ticker] || 0;
      return { ticker, systematic, specific, total: systematic + specific };
    });
    rows.sort((a, b) => b.total - a.total);
    const share = (row, part) => (row.total > 0 ? row[part] / row.total : 0);
    // A 2px border in the surface color reads as a gap between the segments.
    const gap = { borderColor: theme.surface, borderWidth: 2, borderSkipped: false };
    const datasets = [
      { ...bar("Systematic", rows.map((r) => share(r, "systematic")), theme.series1), ...gap },
      { ...bar("Firm-specific", rows.map((r) => share(r, "specific")), theme.series2), ...gap },
    ];
    const labels = rows.map((row) => row.ticker);
    return bars(labels, datasets, theme, { horizontal: true, stacked: true, legend: true, format: percent });
  }

  const CHARTS = [
    { canvas: "weights-chart", island: "weights-data", spec: weights },
    { canvas: "risk-chart", island: "risk-data", spec: risk },
    { canvas: "utility-chart", island: "utility-data", spec: utility },
    { canvas: "beta-chart", island: "sim-data", spec: beta },
    { canvas: "alpha-chart", island: "sim-data", spec: alpha },
    { canvas: "risk-split-chart", island: "sim-data", spec: riskSplit },
  ];

  function parseIsland(id) {
    const node = document.getElementById(id);
    if (!node) {
      return null;
    }
    try {
      return JSON.parse(node.textContent);
    } catch (err) {
      console.error(`Failed to parse JSON in #${id}`, err);
      return null;
    }
  }

  function showEmpty(canvas, message) {
    const note = document.createElement("p");
    note.className = "chart-note";
    note.textContent = message;
    canvas.parentElement.replaceWith(note);
  }

  /**
   * Sizes the canvas frame before construction so a tall chart overflows its
   * scroll container instead of being squashed to fit it.
   */
  function draw(canvas, spec) {
    const horizontal = spec.options.indexAxis === "y";
    const height = horizontal ? Math.max(180, spec.rows * ROW_HEIGHT + AXIS_CHROME) : 300;
    canvas.parentElement.style.height = height + "px";
    new Chart(canvas, {
      type: "bar",
      data: { labels: spec.labels, datasets: spec.datasets },
      options: spec.options,
    });
  }

  function renderCharts() {
    const noLibrary = typeof Chart === "undefined";
    const theme = noLibrary ? null : readTheme();
    const islands = new Map();

    for (const entry of CHARTS) {
      const canvas = document.getElementById(entry.canvas);
      if (!canvas) {
        continue;
      }
      if (noLibrary) {
        showEmpty(canvas, "Chart library failed to load — the figures are in the table below.");
        continue;
      }
      if (!islands.has(entry.island)) {
        islands.set(entry.island, parseIsland(entry.island));
      }
      const data = islands.get(entry.island);
      if (data === null) {
        continue;
      }
      const spec = entry.spec(data, theme);
      if (spec.rows === 0) {
        showEmpty(canvas, "Nothing to plot for this section — see the table below.");
        continue;
      }
      draw(canvas, spec);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderCharts);
  } else {
    renderCharts();
  }
})();
