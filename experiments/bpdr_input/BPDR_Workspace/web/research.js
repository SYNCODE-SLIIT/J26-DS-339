const COLORS = {
  eig: "#8ee5a1",
  lowest_confidence: "#6cc9b8",
  random: "#edc37d",
  none: "#8a9992",
};

function renderCurveSVG(containerId, title, seriesByStrat, metricKey, yLabel, yMin, yMax) {
  const box = document.getElementById(containerId);
  const W = 540, H = 260, padL = 48, padR = 16, padT = 28, padB = 34;
  const xMap = (k) => padL + (k / 5) * (W - padL - padR);
  const yMap = (v) => H - padB - ((v - yMin) / Math.max(1e-6, yMax - yMin)) * (H - padT - padB);

  let paths = "";
  let legend = "";
  Object.entries(seriesByStrat).forEach(([strat, rows], idx) => {
    const col = COLORS[strat] || "#ccc";
    const pts = rows.map((r) => {
      const val = Array.isArray(r[metricKey]) ? r[metricKey][0] : r[metricKey];
      return `${xMap(r.k).toFixed(1)},${yMap(val).toFixed(1)}`;
    });
    paths += `<polyline fill="none" stroke="${col}" stroke-width="2.4" points="${pts.join(" ")}"/>`;
    rows.forEach((r) => {
      const val = Array.isArray(r[metricKey]) ? r[metricKey][0] : r[metricKey];
      paths += `<circle cx="${xMap(r.k).toFixed(1)}" cy="${yMap(val).toFixed(1)}" r="3.5" fill="${col}"/>`;
    });
    legend += `<span style="color:${col}; margin-right:12px; font-size:12px; font-family:var(--mono);">● ${strat}</span>`;
  });

  const yTicks = [yMin, (yMin + yMax) / 2, yMax].map((v) => {
    const y = yMap(v).toFixed(1);
    return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#253431"/><text x="6" y="${Number(y) + 4}" fill="#9eb0a6" font-size="11">${v.toFixed(2)}</text>`;
  }).join("");

  const xTicks = [0, 1, 2, 3, 4, 5].map((k) => (
    `<text x="${xMap(k) - 4}" y="${H - 10}" fill="#9eb0a6" font-size="11">K=${k}</text>`
  )).join("");

  box.innerHTML = `
    <div style="margin-bottom:6px;"><strong>${title}</strong> <span class="small muted">(${yLabel})</span></div>
    <div style="margin-bottom:6px;">${legend}</div>
    <svg viewBox="0 0 ${W} ${H}">${yTicks}${xTicks}${paths}</svg>
  `;
}

function fmtCI(a, d) {
  if (!Array.isArray(a)) return "n/a";
  return `${a[0].toFixed(d)} [${a[1].toFixed(d)}, ${a[2].toFixed(d)}]`;
}

async function init() {
  const res = await fetch("/api/research/summary");
  const data = await res.json();

  // Section 0: Core 4-View BPDR Disentangled Benchmark (vs BERT-APC split/merge limitation)
  if (data.bpdr_4view_benchmark && data.bpdr_4view_benchmark.summary_by_arm) {
    const sum = data.bpdr_4view_benchmark.summary_by_arm;
    const rows = Object.entries(sum).map(([arm, m]) => `
      <tr>
        <td class="mono"><strong>${arm}</strong></td>
        <td class="mono">${m.mean_clean_mae} c</td>
        <td class="mono">${m.mean_split_mae ?? m.mean_corrupt_mae} c</td>
        <td class="mono">${m.mean_merge_mae ?? m.mean_corrupt_mae} c</td>
        <td class="mono">${m.mean_jitter_mae ?? m.mean_corrupt_mae} c</td>
        <td class="mono"><strong>${m.mean_corrupt_mae} c</strong> (${m.mean_corruption_drop >= 0 ? "+" : ""}${m.mean_corruption_drop} c)</td>
        <td class="mono">${m.mean_boundary_sensitivity_lb ?? "—"} c</td>
        <td class="mono">${m.mean_counter_detuning_ld ?? "—"} c</td>
        <td class="mono">${m.mean_vibrato_preservation_mae} c</td>
      </tr>
    `).join("");
    document.getElementById("bpdr-4view-table").innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Arm (3 seeds: 339, 340, 341)</th>
            <th>Clean MAE</th>
            <th>False Split MAE</th>
            <th>False Merge MAE</th>
            <th>Jitter MAE</th>
            <th>Corrupt Macro (Drop)</th>
            <th>Boundary Sens. (L_B)</th>
            <th>Counter-Detuning (L_D)</th>
            <th>Vibrato Pres.</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // Section A: H-I1a & H-I1b
  if (data.aic_dev) {
    const { vocadito, v64_pitch } = data.aic_dev;
    renderCurveSVG("chart-hi1a-oracle", "H-I1a Boundary (Oracle A1 user)", vocadito.curves.oracle_A1, "note_f1", "Note F1", 0.55, 0.78);
    renderCurveSVG("chart-hi1a-noisy", "H-I1a Boundary (Noisy A2 user)", vocadito.curves.noisy_A2, "note_f1", "Note F1", 0.55, 0.78);
    renderCurveSVG("chart-hi1b-pitch", "H-I1b Pitch Correction Error (498 v64 records)", v64_pitch.curves, "pos_mae_cents", "Positive MAE cents (lower is better)", 0, 30);

    // Section B: Calibration & Do-No-Harm
    const bc = vocadito.boundary_calibration;
    const pc = v64_pitch.pitch_calibration;
    document.getElementById("calibration-summary").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Factor</th><th>Candidates</th><th>Raw ECE</th><th>LOO Calibrated ECE</th><th>Raw Brier</th><th>LOO Calibrated Brier</th></tr></thead>
        <tbody>
          <tr><td>Boundary (Vocadito, 14 singers)</td><td>${bc.n_candidates}</td><td>${bc.raw.ece}</td><td><strong>${bc.calibrated_loo.ece}</strong></td><td>${bc.raw.brier}</td><td><strong>${bc.calibrated_loo.brier}</strong></td></tr>
          <tr><td>Pitch (v64, 20 groups)</td><td>${pc.n_candidates}</td><td>${pc.raw.ece}</td><td><strong>${pc.calibrated_loo.ece}</strong></td><td>${pc.raw.brier}</td><td><strong>${pc.calibrated_loo.brier}</strong></td></tr>
        </tbody>
      </table>
      <div class="notice warn" style="margin-top:12px;">
        <strong>H-I3 Do-No-Harm (measured, simulated listener):</strong>
        untouched originals edited under AIC K=3 with a noisy listener:
        <strong>${fmtCI(v64_pitch.h_i3_do_no_harm.aic_noisy_user_pct, 1)}%</strong>
        (oracle listener: ${fmtCI(v64_pitch.h_i3_do_no_harm.aic_oracle_user_pct, 1)}%, zero by construction);
        automatic repair of every candidate would edit
        <strong>${fmtCI(v64_pitch.h_i3_do_no_harm.auto_repair_every_candidate_pct, 1)}%</strong>.
        ${v64_pitch.h_i3_do_no_harm.n_untouched_originals} originals; human listeners not yet tested.
      </div>
      <div class="notice" style="margin-top:12px;">
        <strong>Ceilings:</strong> boundary note F1 if every candidate were answered by the oracle
        ${fmtCI(vocadito.recall_ceiling.note_f1_all_candidates_answered_by_oracle, 3)};
        pitch positive MAE if every candidate were answered correctly
        ${fmtCI(v64_pitch.recall_ceiling.pos_mae_cents_if_every_candidate_answered_correctly, 2)} c
        (uncorrected ${v64_pitch.recall_ceiling.uncorrected_pos_mae_cents} c).
      </div>
    `;

    // Section C: Ablation
    const methods = vocadito.ablation.methods;
    const rowsHtml = Object.entries(methods).map(([name, m]) => `
      <tr>
        <td class="mono">${name}</td>
        <td class="mono">${m.note_f1[0].toFixed(3)} [${m.note_f1[1].toFixed(3)}, ${m.note_f1[2].toFixed(3)}]</td>
        <td class="mono">${m.onset_f1[0].toFixed(3)} [${m.onset_f1[1].toFixed(3)}, ${m.onset_f1[2].toFixed(3)}]</td>
      </tr>
    `).join("");
    document.getElementById("ablation-tables").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Method (39 Vocadito dev windows, 14 singers)</th><th>Note F1 [95% CI]</th><th>Onset F1 [95% CI]</th></tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  }

  // Section D: R2 Honest Negative
  if (data.r2_control_summary) {
    const r2 = data.r2_control_summary;
    const armRows = Object.entries(r2.arms).map(([arm, v]) => `
      <tr>
        <td class="mono">${arm}</td>
        <td class="mono">${v.raw_positive_mae_cents} c</td>
        <td class="mono">${v.raw_original_false_corr_cents} c</td>
      </tr>
    `).join("");
    document.getElementById("r2-table").innerHTML = `
      <div class="notice warn">
        <strong>Fit health:</strong> ${r2.fit_health.healthy_fits}/${r2.fit_health.total_fits} healthy fits ·
        <strong>advance_to_delivery:</strong> ${String(r2.advance_to_delivery)} ·
        <strong>production_eligible:</strong> ${String(r2.production_eligible)}
      </div>
      <table class="data-table">
        <thead><tr><th>Arm (1.5855x weight-matched controls)</th><th>Raw Positive MAE</th><th>Raw Original False Correction</th></tr></thead>
        <tbody>${armRows}</tbody>
      </table>
    `;
  }

  // Section E: Claims Ledger & Provenance
  const cl = data.claims_ledger;
  document.getElementById("claims-ledger").innerHTML = `
    <div class="ab-grid">
      <div class="notice ok">
        <strong>May Claim (with stated limits)</strong>
        <ul>${cl.may_claim.map((c) => `<li><strong>${c.claim}:</strong> ${c.evidence} <span class="small muted">(${c.limit})</span></li>`).join("")}</ul>
      </div>
      <div class="notice err">
        <strong>Do Not Claim</strong>
        <ul>${cl.do_not_claim.map((c) => `<li>${c}</li>`).join("")}</ul>
      </div>
    </div>
  `;

  document.getElementById("provenance-table").innerHTML = `
    <table class="data-table">
      <thead><tr><th>Artifact</th><th>Path</th><th>SHA-256</th></tr></thead>
      <tbody>
        ${data.provenance.map((p) => `<tr><td class="mono">${p.key}</td><td class="mono">${p.path}</td><td class="mono small">${p.sha256}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

init();
