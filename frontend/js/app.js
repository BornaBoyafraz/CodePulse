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

const repoHeading   = document.getElementById('repo-heading');
const statTotal     = document.getElementById('stat-total');
const statHighRisk  = document.getElementById('stat-high-risk');
const statTime      = document.getElementById('stat-time');
const riskTableBody = document.getElementById('risk-table-body');
const metricsChips   = document.getElementById('metrics-chips');
const prCurveSection  = document.getElementById('pr-curve-section');
const treemapSection  = document.getElementById('treemap-section');
const detailPanel   = document.getElementById('detail-panel');
const detailContent = document.getElementById('detail-content');
const detailCloseBtn = document.getElementById('detail-close-btn');

// ─── State ─────────────────────────────────────────────────────── //

let pollTimer   = null;
let activeJobId = null;

// ─── Pipeline Stage Labels ─────────────────────────────────────── //

const STAGE_LABELS = {
  pending:    'Queued…',
  running:    'Starting pipeline…',
  mining:     'Mining commit history — this may take 60–90 seconds…',
  labeling:   'Labeling bug-fix commits…',
  features:   'Computing file features…',
  training:   'Training XGBoost model…',
  explaining: 'Generating SHAP explanations…',
};

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

function showSkeletonState(repoUrl) {
  inputScreen.style.display = 'none';
  loadingState.classList.remove('visible');
  resultsScreen.classList.add('visible');

  repoHeading.innerHTML    = formatRepoName(repoUrl).replace(' / ', ' <span>/</span> ');
  statTotal.textContent    = '—';
  statHighRisk.textContent = '—';
  statTime.textContent     = '—';
  metricsChips.innerHTML   = '';

  riskTableBody.innerHTML = Array.from({ length: 8 }, () => `
    <tr>
      <td class="row-rank"><span class="skeleton skeleton-inline"></span></td>
      <td class="file-path"><span class="skeleton skeleton-path"></span></td>
      <td style="text-align:center"><span class="skeleton skeleton-badge-sm"></span></td>
      <td class="drivers-cell">
        <span class="skeleton skeleton-tag"></span>
        <span class="skeleton skeleton-tag"></span>
      </td>
    </tr>
  `).join('') + `
    <tr id="skeleton-status-row">
      <td colspan="4" class="skeleton-status-text" id="skeleton-status-text">Queued…</td>
    </tr>
  `;
}

function updateSkeletonStatus(status) {
  const el = document.getElementById('skeleton-status-text');
  if (el) el.textContent = STAGE_LABELS[status] || 'Analyzing…';
}

function showInputState() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  closeDetailPanel();
  activeJobId = null;

  resultsScreen.classList.remove('visible');
  inputScreen.style.display = '';

  analyzeBtn.disabled = false;
  analyzeBtn.textContent = 'Analyze →';
  loadingState.classList.remove('visible');
  urlInput.value = '';
  clearError();
  urlInput.focus();

  // Reset table
  riskTableBody.innerHTML = '';

  // Purge charts so stale data never bleeds into a fresh analysis
  if (document.getElementById('risk-chart'))     Plotly.purge('risk-chart');
  if (document.getElementById('treemap-chart'))  Plotly.purge('treemap-chart');
  if (document.getElementById('pr-curve-chart')) Plotly.purge('pr-curve-chart');

  // Hide conditional sections (they start hidden and are un-hidden after load)
  treemapSection.setAttribute('hidden', '');
  prCurveSection.setAttribute('hidden', '');

  // Reset stat strips
  statTotal.textContent    = '—';
  statHighRisk.textContent = '—';
  statTime.textContent     = '—';
  metricsChips.innerHTML   = '';
}

function showResultsState(data, jobId) {
  inputScreen.style.display = 'none';
  resultsScreen.classList.add('visible');
  activeJobId = jobId;

  repoHeading.innerHTML = formatRepoName(data.repo_url)
    .replace(' / ', ' <span>/</span> ');

  const total    = data.files.length;
  const highRisk = data.files.filter(f => f.risk_score >= 70).length;

  statTotal.textContent    = total;
  statHighRisk.textContent = highRisk;
  statTime.textContent     = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (total === 0) {
    riskTableBody.innerHTML = '<tr><td colspan="4" class="empty-state">Not enough commit history to generate predictions.</td></tr>';
    metricsChips.innerHTML = '';
    return;
  }

  renderChart(data.files);
  renderTable(data.files, jobId);
  loadMetrics(jobId);
  loadTreemap(jobId);
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

// ─── Metrics Chips + PR Curve ──────────────────────────────────── //

async function loadTreemap(jobId) {
  try {
    const res = await fetch(`/results/${jobId}/treemap`);
    if (!res.ok) return;
    const data = await res.json();
    if (!data.treemap_json || data.treemap_json === '{}') return;
    const fig = JSON.parse(data.treemap_json);
    const layout = Object.assign({}, fig.layout || {}, { paper_bgcolor: 'transparent' });
    Plotly.newPlot('treemap-chart', fig.data || [], layout, { displayModeBar: false, responsive: true });
    treemapSection.removeAttribute('hidden');
  } catch { /* silent — treemap is supplementary */ }
}

async function loadMetrics(jobId) {
  try {
    const res = await fetch(`/results/${jobId}/metrics`);
    if (!res.ok) return;
    const m = await res.json();
    renderMetricsChips(m);
    if (m.pr_curve_json && m.pr_curve_json !== '{}') {
      renderPRCurve(m.pr_curve_json);
    }
  } catch { /* metrics are supplementary — silent fail */ }
}

function renderMetricsChips(m) {
  const aucVal  = m.auc_roc   > 0 ? m.auc_roc.toFixed(3)   : 'N/A';
  const apVal   = m.avg_precision > 0 ? m.avg_precision.toFixed(3) : 'N/A';

  metricsChips.innerHTML = `
    <div class="metric-chip">
      <span class="chip-label">AUC-ROC</span>
      <span class="chip-value">${aucVal}</span>
    </div>
    <div class="metric-chip">
      <span class="chip-label">Avg Precision</span>
      <span class="chip-value">${apVal}</span>
    </div>
  `;
}

function renderPRCurve(prJson) {
  try {
    const fig = JSON.parse(prJson);
    const data    = fig.data    || [];
    const layout  = Object.assign({}, fig.layout || {}, {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
    });
    const config = { displayModeBar: false, responsive: true };
    Plotly.newPlot('pr-curve-chart', data, layout, config);
    prCurveSection.removeAttribute('hidden');
  } catch { /* invalid JSON — skip */ }
}

// ─── File Detail Panel ─────────────────────────────────────────── //

function openDetailPanel() {
  detailPanel.classList.add('is-open');
  detailPanel.setAttribute('aria-hidden', 'false');
}

function closeDetailPanel() {
  detailPanel.classList.remove('is-open');
  detailPanel.setAttribute('aria-hidden', 'true');
  detailContent.innerHTML = '';
}

async function loadFileDetail(jobId, filePath) {
  const encoded = encodeURIComponent(filePath);
  try {
    const res = await fetch(`/results/${jobId}/file?path=${encoded}`);
    if (!res.ok) return;
    const d = await res.json();
    renderDetailPanel(d);
    openDetailPanel();
  } catch { /* silent */ }
}

function renderDetailPanel(d) {
  const cls   = getRiskClass(d.risk_score);
  const label = getRiskLabel(d.risk_score);
  const isCritical = d.risk_score >= 80 ? 'is-critical' : '';

  const driversHtml = d.top_drivers.length
    ? d.top_drivers.map(dr => `<li class="detail-driver-item">${escapeHtml(dr)}</li>`).join('')
    : '<li class="detail-driver-item muted">No significant drivers identified.</li>';

  detailContent.innerHTML = `
    <p class="detail-file-path">${escapeHtml(d.file_path)}</p>
    <span class="risk-badge detail-risk-badge ${cls} ${isCritical}">${label} ${d.risk_score.toFixed(0)}</span>

    <h3 class="detail-section-label">Risk Drivers</h3>
    <ul class="detail-drivers">${driversHtml}</ul>

    <h3 class="detail-section-label">File Statistics</h3>
    <div class="detail-stats-grid">
      <div class="detail-stat">
        <span class="detail-stat-label">Total Commits</span>
        <span class="detail-stat-value">${d.total_commits}</span>
      </div>
      <div class="detail-stat">
        <span class="detail-stat-label">30-day Churn</span>
        <span class="detail-stat-value">${d.churn_30d}</span>
      </div>
      <div class="detail-stat">
        <span class="detail-stat-label">Bus Factor</span>
        <span class="detail-stat-value">${d.bus_factor_score.toFixed(2)}</span>
      </div>
      <div class="detail-stat">
        <span class="detail-stat-label">Max Coupling</span>
        <span class="detail-stat-value">${d.coupling_score_max.toFixed(2)}</span>
      </div>
    </div>
  `;
}

// ─── Risk Table ────────────────────────────────────────────────── //

function renderTable(files, jobId) {
  const sorted = [...files].sort((a, b) => b.risk_score - a.risk_score);
  riskTableBody.innerHTML = '';

  sorted.forEach((file, index) => {
    const cls       = getRiskClass(file.risk_score);
    const label     = getRiskLabel(file.risk_score);
    const isCritical = file.risk_score >= 80 ? 'is-critical' : '';

    const driversHtml = file.top_drivers
      .slice(0, 3)
      .map(d => `<span class="driver-tag">${escapeHtml(d)}</span>`)
      .join('');

    const tr = document.createElement('tr');
    tr.className = 'clickable-row';
    tr.setAttribute('title', 'Click for full breakdown');
    tr.innerHTML = `
      <td class="row-rank">${index + 1}</td>
      <td class="file-path">${escapeHtml(file.file_path)}</td>
      <td style="text-align:center">
        <span class="risk-badge ${cls} ${isCritical}">${label} ${file.risk_score.toFixed(0)}</span>
      </td>
      <td class="drivers-cell">${driversHtml}</td>
    `;
    tr.addEventListener('click', () => loadFileDetail(jobId, file.file_path));
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

  // Switch to results screen immediately with skeleton rows
  showSkeletonState(repoUrl);
  activeJobId = jobId;
  pollResults(jobId);
}

function pollResults(jobId) {
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/results/${jobId}`);

      if (!res.ok) {
        clearInterval(pollTimer);
        pollTimer = null;
        showInputState();
        showError(`Could not retrieve results (job: ${jobId}).`);
        return;
      }

      const data = await res.json();

      if (data.status === 'complete') {
        clearInterval(pollTimer);
        pollTimer = null;
        showResultsState(data, jobId);
      } else if (data.status && data.status.startsWith('error')) {
        clearInterval(pollTimer);
        pollTimer = null;
        const raw = data.status.replace(/^error:\s*/i, '') || 'Pipeline failed.';
        const lower = raw.toLowerCase();
        const friendly = lower.includes('fewer than 100 commits') || lower.includes('not enough')
          ? "This repo doesn't have enough commit history (need at least 100 commits) to generate meaningful predictions."
          : lower.includes('not found') || raw.includes('404')
          ? 'Repository not found. Make sure the URL is correct and the repo is public.'
          : `Analysis failed: ${raw}`;
        showInputState();
        showError(friendly);
      } else {
        // Still running — update skeleton status text
        updateSkeletonStatus(data.status);
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      showInputState();
      showError('Connection lost while polling for results.');
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

detailCloseBtn.addEventListener('click', closeDetailPanel);

// Close detail panel on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && detailPanel.classList.contains('is-open')) {
    closeDetailPanel();
  }
});

// ─── Init ──────────────────────────────────────────────────────── //

urlInput.focus();
