/* CodePulse — Frontend Application Logic
   State machine: input screen ↔ results dashboard
   ─────────────────────────────────────────────────────────────── */

'use strict';

// ─── DOM References ────────────────────────────────────────────── //

const inputScreen   = document.getElementById('input-screen');
const resultsScreen = document.getElementById('results-screen');
const urlInput      = document.getElementById('repo-url');
const analyzeBtn    = document.getElementById('analyze-btn');
const loadingState  = document.getElementById('loading-state');
const errorMsg      = document.getElementById('error-msg');
const errorText     = document.getElementById('error-text');

const repoHeading  = document.getElementById('repo-heading');
const statTotal    = document.getElementById('stat-total');
const statHighRisk = document.getElementById('stat-high-risk');
const statTime     = document.getElementById('stat-time');
const riskTableBody = document.getElementById('risk-table-body');

// ─── State ─────────────────────────────────────────────────────── //

let pollTimer = null;

// ─── Risk Helpers ──────────────────────────────────────────────── //

function getRiskColor(score) {
  if (score < 40) return '#3fb950';
  if (score < 70) return '#d29922';
  return '#f85149';
}

function getRiskLabel(score) {
  if (score < 40) return 'LOW';
  if (score < 70) return 'MED';
  return 'HIGH';
}

function getRiskClass(score) {
  if (score < 40) return 'low';
  if (score < 70) return 'mid';
  return 'high';
}

// ─── Repo Name Formatting ──────────────────────────────────────── //

function formatRepoName(url) {
  try {
    const parts = url.replace(/\/$/, '').split('/');
    const owner = parts[parts.length - 2] || '';
    const repo  = parts[parts.length - 1] || '';
    if (owner && repo) return `${owner} / ${repo}`;
    return url;
  } catch {
    return url;
  }
}

// ─── Error Display ─────────────────────────────────────────────── //

function showError(message) {
  errorText.textContent = message;
  errorMsg.classList.add('visible');
}

function clearError() {
  errorMsg.classList.remove('visible');
  errorText.textContent = '';
}

// ─── State Transitions ─────────────────────────────────────────── //

function showInputState() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  resultsScreen.classList.remove('visible');
  inputScreen.style.display = '';

  analyzeBtn.disabled = false;
  analyzeBtn.textContent = 'Analyze →';
  loadingState.classList.remove('visible');
  urlInput.value = '';
  clearError();
  urlInput.focus();
}

function showResultsState(data) {
  inputScreen.style.display = 'none';
  resultsScreen.classList.add('visible');

  repoHeading.innerHTML = formatRepoName(data.repo_url)
    .replace(' / ', ' <span>/</span> ');

  const total    = data.files.length;
  const highRisk = data.files.filter(f => f.risk_score >= 70).length;

  statTotal.textContent    = total;
  statHighRisk.textContent = highRisk;
  statTime.textContent     = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  renderChart(data.files);
  renderTable(data.files);
}

// ─── Plotly Chart ──────────────────────────────────────────────── //

function renderChart(files) {
  const top10 = [...files]
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 10);

  const labels = top10.map(f => f.file_path).reverse();
  const scores = top10.map(f => f.risk_score).reverse();
  const colors = top10.map(f => getRiskColor(f.risk_score)).reverse();

  const trace = {
    type:        'bar',
    orientation: 'h',
    x:           scores,
    y:           labels,
    marker: {
      color:   colors,
      opacity: 0.9,
      line:    { width: 0 },
    },
    hovertemplate: '<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>',
    hoverlabel: {
      bgcolor:   '#161b22',
      font:      { family: 'JetBrains Mono', size: 12, color: '#e6edf3' },
      bordercolor: '#30363d',
    },
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    margin:        { l: 200, r: 40, t: 20, b: 40 },
    xaxis: {
      range:      [0, 100],
      tickfont:   { family: 'Inter', size: 11, color: '#8b949e' },
      gridcolor:  '#21262d',
      linecolor:  '#30363d',
      title:      { text: 'Risk Score', font: { family: 'Inter', size: 11, color: '#8b949e' } },
      fixedrange: true,
    },
    yaxis: {
      tickfont:   { family: 'JetBrains Mono', size: 11, color: '#e6edf3' },
      linecolor:  '#30363d',
      fixedrange: true,
      automargin: true,
    },
    bargap: 0.35,
    height: Math.max(300, top10.length * 44 + 60),
    font:   { family: 'Inter', color: '#e6edf3' },
    shapes: [
      { type: 'line', x0: 40, x1: 40, y0: -0.5, y1: top10.length - 0.5,
        line: { color: '#d29922', width: 1, dash: 'dot' } },
      { type: 'line', x0: 70, x1: 70, y0: -0.5, y1: top10.length - 0.5,
        line: { color: '#f85149', width: 1, dash: 'dot' } },
    ],
  };

  const config = {
    displayModeBar: false,
    responsive:     true,
  };

  Plotly.newPlot('risk-chart', [trace], layout, config);
}

// ─── Risk Table ────────────────────────────────────────────────── //

function renderTable(files) {
  const sorted = [...files].sort((a, b) => b.risk_score - a.risk_score);
  riskTableBody.innerHTML = '';

  sorted.forEach((file, index) => {
    const cls   = getRiskClass(file.risk_score);
    const label = getRiskLabel(file.risk_score);

    const driversHtml = file.top_drivers
      .slice(0, 3)
      .map(d => `<span class="driver-tag">${escapeHtml(d)}</span>`)
      .join('');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="row-rank">${index + 1}</td>
      <td class="file-path">${escapeHtml(file.file_path)}</td>
      <td style="text-align:center">
        <span class="risk-badge ${cls}">${label} ${file.risk_score.toFixed(0)}</span>
      </td>
      <td class="drivers-cell">${driversHtml}</td>
    `;
    riskTableBody.appendChild(tr);
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── API Calls ─────────────────────────────────────────────────── //

async function submitAnalysis() {
  const repoUrl = urlInput.value.trim();
  if (!repoUrl) {
    showError('Please enter a GitHub repository URL.');
    urlInput.focus();
    return;
  }

  clearError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  loadingState.classList.add('visible');

  let jobId;
  try {
    const res = await fetch('/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ repo_url: repoUrl }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    jobId = data.job_id;
  } catch (err) {
    showError(err.message || 'Failed to reach the server. Is the API running?');
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze →';
    loadingState.classList.remove('visible');
    return;
  }

  pollResults(jobId);
}

function pollResults(jobId) {
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/results/${jobId}`);

      if (!res.ok) {
        clearInterval(pollTimer);
        pollTimer = null;
        showError(`Could not retrieve results (job: ${jobId}).`);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze →';
        loadingState.classList.remove('visible');
        return;
      }

      const data = await res.json();

      if (data.status === 'complete') {
        clearInterval(pollTimer);
        pollTimer = null;
        showResultsState(data);
      } else if (data.status && data.status.startsWith('error')) {
        clearInterval(pollTimer);
        pollTimer = null;
        const detail = data.status.replace(/^error:\s*/i, '') || 'Pipeline failed.';
        showError(`Analysis failed: ${detail}`);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze →';
        loadingState.classList.remove('visible');
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      showError('Connection lost while polling for results.');
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze →';
      loadingState.classList.remove('visible');
    }
  }, 2000);
}

// ─── Event Listeners ───────────────────────────────────────────── //

analyzeBtn.addEventListener('click', submitAnalysis);

urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submitAnalysis();
});

urlInput.addEventListener('input', clearError);

document.getElementById('reset-btn').addEventListener('click', showInputState);

// ─── Init ──────────────────────────────────────────────────────── //

urlInput.focus();
