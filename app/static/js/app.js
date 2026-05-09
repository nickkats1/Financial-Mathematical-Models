/**
 * Portfolio Analytics — client-side helpers.
 *
 * Renders the optimal-weights bar chart on the results page using Chart.js.
 * The weights are embedded as JSON inside <script id="weights-data">.
 */

(function () {
  "use strict";

  function renderWeightsChart() {
    const canvas = document.getElementById("weights-chart");
    const dataNode = document.getElementById("weights-data");
    if (!canvas || !dataNode || typeof Chart === "undefined") {
      return;
    }

    let weights;
    try {
      weights = JSON.parse(dataNode.textContent);
    } catch (err) {
      console.error("Failed to parse weights JSON", err);
      return;
    }

    const entries = Object.entries(weights)
      .filter(([, weight]) => weight > 0)
      .sort(([, a], [, b]) => b - a);

    if (entries.length === 0) {
      canvas.replaceWith(document.createTextNode("No non-zero weights to plot."));
      return;
    }

    const labels = entries.map(([ticker]) => ticker);
    const values = entries.map(([, weight]) => weight);

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Weight",
          data: values,
          backgroundColor: "rgba(37, 99, 235, 0.85)",
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: (value) => (value * 100).toFixed(1) + "%",
            },
          },
        },
      },
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderWeightsChart);
  } else {
    renderWeightsChart();
  }
})();
