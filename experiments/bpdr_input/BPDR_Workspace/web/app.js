const state = {
  sessionId: null,
  currentStep: 1,
  verdict: null,
  questionsPayload: null,
  qIndex: 0,
  answers: [],
  repairResult: null,
  approvedSpanIds: new Set(),
  trimConfirmed: true,
  isRecording: false,
  recCtx: null,
  recStream: null,
  recChunks: [],
  recSampleRate: 22050,
  recTimerId: null,
  recStartMs: 0,
};

const $ = (id) => document.getElementById(id);

function showStep(stepNum) {
  state.currentStep = stepNum;
  for (let i = 1; i <= 8; i++) {
    const sec = $(`screen-${i}`);
    if (sec) sec.hidden = i !== stepNum;
  }
  document.querySelectorAll(".step-pill").forEach((el) => {
    const s = Number(el.dataset.step);
    el.classList.toggle("active", s === stepNum);
    el.classList.toggle("done", s < stepNum);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function captureMicWav(maxSeconds, onTick, onDone) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/audio-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-recorder-processor");
  const chunks = [];
  let maxPeak = 0;

  node.port.onmessage = (ev) => {
    const { samples, peak } = ev.data;
    chunks.push(samples);
    if (peak > maxPeak) maxPeak = peak;
    if (onTick) onTick(peak, (performance.now() - t0) / 1000, maxPeak);
  };

  source.connect(node);
  node.connect(ctx.destination);
  const t0 = performance.now();

  const stop = async () => {
    clearTimeout(autoStopId);
    clearInterval(timerId);
    node.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const total = chunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(total);
    let pos = 0;
    for (const c of chunks) {
      merged.set(c, pos);
      pos += c.length;
    }
    const sr = ctx.sampleRate;
    await ctx.close();
    const wavBlob = encodeWAV(merged, sr);
    if (onDone) onDone(wavBlob);
    return wavBlob;
  };

  const timerId = setInterval(() => {
    const elapsed = (performance.now() - t0) / 1000;
    if (onTick) onTick(0, elapsed, maxPeak);
  }, 100);
  const autoStopId = setTimeout(() => stop(), maxSeconds * 1000);
  return stop;
}

async function apiJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const raw = await res.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch (_) {
    throw new Error(raw || `HTTP ${res.status}`);
  }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// Screen 1: Welcome & Privacy
$("consent-check").addEventListener("change", (e) => {
  $("btn-start-session").disabled = !e.target.checked;
});

$("btn-start-session").addEventListener("click", async () => {
  try {
    const res = await apiJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ consent: true }),
    });
    state.sessionId = res.session_id;
    $("session-pill").textContent = `Session ${res.session_id.slice(0, 8)}… (24h retention)`;
    await loadExamples();
    showStep(2);
  } catch (err) {
    alert(`Could not create session: ${err.message}`);
  }
});

// Screen 2: Room Check
$("btn-room-record").addEventListener("click", async () => {
  const btn = $("btn-room-record");
  btn.disabled = true;
  btn.textContent = "Listening (3 s silence)...";
  try {
    await captureMicWav(3.0, null, async (blob) => {
      const fd = new FormData();
      fd.append("file", blob, "room.wav");
      const res = await apiJson(`/api/sessions/${state.sessionId}/room`, { method: "POST", body: fd });
      renderRoomResult(res);
      btn.disabled = false;
      btn.textContent = "Record 3 s silence again";
    });
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Record 3 s silence";
    renderRoomResult({ verdict: "error", tip: `Microphone unavailable (${err.message}). You can skip room check.` });
  }
});

$("btn-room-skip").addEventListener("click", async () => {
  await apiJson(`/api/sessions/${state.sessionId}/room`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preset: "skip" }),
  });
  showStep(3);
});

$("btn-room-next").addEventListener("click", () => showStep(3));

function renderRoomResult(res) {
  const box = $("room-result");
  box.hidden = false;
  box.className = `notice ${res.verdict === "good" ? "ok" : res.verdict === "too_loud" ? "err" : "warn"}`;
  box.innerHTML = `<strong>Room check: ${String(res.verdict).toUpperCase()}</strong><br>${res.tip}`;
  $("room-next-row").hidden = false;
}

// Screen 3: Record or Upload
async function loadExamples() {
  try {
    const res = await apiJson("/api/examples");
    const sel = $("example-select");
    sel.innerHTML = "";
    res.examples.forEach((ex) => {
      const opt = document.createElement("option");
      opt.value = ex.name;
      opt.textContent = ex.label;
      sel.appendChild(opt);
    });
  } catch (_) {}
}

let stopMicFn = null;
$("btn-mic-toggle").addEventListener("click", async () => {
  const btn = $("btn-mic-toggle");
  if (!state.isRecording) {
    try {
      state.isRecording = true;
      btn.classList.add("recording");
      btn.setAttribute("aria-pressed", "true");
      btn.textContent = "■ Stop Recording";
      stopMicFn = await captureMicWav(
        20.0,
        (peak, elapsed, maxPeak) => {
          $("rec-timer").textContent = `${Math.min(20, elapsed).toFixed(1)} s / 20.0 s`;
          const pct = Math.min(100, Math.round(peak * 100));
          $("meter-fill").style.width = `${pct}%`;
          const isClip = maxPeak >= 0.98;
          $("meter-fill").classList.toggle("clip", isClip);
          $("meter-label").textContent = peak < 0.02 ? "Quiet" : peak > 0.9 ? "Hot / Loud" : "Good level";
          $("clip-flag").textContent = isClip ? "CLIPPING DETECTED" : "No clipping";
        },
        async (wavBlob) => {
          state.isRecording = false;
          btn.classList.remove("recording");
          btn.setAttribute("aria-pressed", "false");
          btn.textContent = "● Start Recording (max 20 s)";
          const fd = new FormData();
          fd.append("file", wavBlob, "mic_take.wav");
          await handleAudioUploaded(fd, true);
        }
      );
    } catch (err) {
      state.isRecording = false;
      btn.classList.remove("recording");
      alert(`Microphone error: ${err.message}. You can upload a WAV file or load an example instead.`);
    }
  } else if (stopMicFn) {
    await stopMicFn();
    stopMicFn = null;
  }
});

$("file-upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file, file.name);
  await handleAudioUploaded(fd, true);
});

$("btn-load-example").addEventListener("click", async () => {
  const exName = $("example-select").value;
  if (!exName) return;
  await handleAudioUploaded(JSON.stringify({ example_name: exName }), false);
});

async function handleAudioUploaded(payload, isMultipart) {
  const opts = isMultipart
    ? { method: "POST", body: payload }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: payload };
  const res = await apiJson(`/api/sessions/${state.sessionId}/audio`, opts);
  $("pre-send-box").hidden = false;
  $("pre-send-audio").src = res.input_wav_url;
  if (res.technical_failure) {
    $("pre-send-meta").textContent = `${res.source_label} — Warning: ${res.error}`;
    $("trim-confirm-box").hidden = true;
    return;
  }
  $("pre-send-meta").textContent = `${res.source_label} · Duration: ${res.duration_s} s · Clipped fraction: ${(res.clipped_fraction * 100).toFixed(2)}%`;
  const ps = res.phrase_selection || {};
  if (res.needs_trim_confirmation) {
    $("trim-confirm-box").hidden = false;
    const msg = ps.phrase_selected
      ? `Recording exceeds 20 s (${ps.original_duration_s.toFixed(1)} s). Proposed best <=20 s phrase: ${ps.start_s.toFixed(1)}–${ps.end_s.toFixed(1)} s.`
      : `Detected leading/trailing silence (${(ps.trimmed_leading_s || 0).toFixed(1)} s lead, ${(ps.trimmed_trailing_s || 0).toFixed(1)} s trail).`;
    $("trim-confirm-box").innerHTML = `
      <strong>Phrase selection / silence trim proposed</strong><br>${msg}
      <label style="display:block; margin-top:8px;">
        <input type="checkbox" id="chk-confirm-trim" checked> Confirm trimming to the active <=20 s phrase
      </label>
    `;
  } else {
    $("trim-confirm-box").hidden = true;
  }
}

$("btn-analyze").addEventListener("click", async () => {
  const chk = $("chk-confirm-trim");
  const autoTrim = chk ? chk.checked : true;
  showStep(4);
  $("verdict-body").hidden = true;
  $("analyze-progress-wrap").hidden = false;
  $("analyze-progress-fill").style.width = "35%";
  try {
    const res = await apiJson(`/api/sessions/${state.sessionId}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intent: "unsure", strategy: "eig", k: 3, auto_trim: autoTrim }),
    });
    $("analyze-progress-fill").style.width = "100%";
    state.verdict = res.verdict;
    renderVerdict(res.verdict);
  } catch (err) {
    $("analyze-progress-wrap").hidden = true;
    $("verdict-body").hidden = false;
    $("verdict-badge").className = "status-badge err";
    $("verdict-badge").textContent = "TECHNICAL_FAILURE";
    $("verdict-reasons").textContent = err.message;
  }
});

// Screen 4: Verdict
function renderVerdict(v) {
  $("analyze-progress-wrap").hidden = true;
  $("verdict-body").hidden = false;
  $("s4-title").textContent = `Verdict (${v.processing_elapsed_s ?? "—"} s)`;
  const ok = v.status === "ACCEPT_UNCHANGED" || v.status === "ACCEPT_PREPARED";
  const badge = $("verdict-badge");
  badge.className = `status-badge ${ok ? "ok" : v.status === "RERECORD_REQUIRED" ? "warn" : "err"}`;
  badge.textContent = `${ok ? "✓" : "!"} ${v.status}`;

  $("verdict-reasons").className = `notice ${ok ? "ok" : "err"}`;
  $("verdict-reasons").innerHTML = (v.reasons || []).map((r) => `<div>${r}</div>`).join("")
    + (v.advisories && v.advisories.length ? `<div class="small muted" style="margin-top:6px;">${v.advisories.join(" · ")}</div>` : "");

  if (v.guidance && v.guidance.length) {
    $("verdict-guidance").hidden = false;
    $("verdict-guidance").innerHTML = `<strong>Actionable guidance:</strong><ul>${v.guidance.map((g) => `<li>${g}</li>`).join("")}</ul>`;
  } else {
    $("verdict-guidance").hidden = true;
  }

  $("speech-fallback-box").hidden = !v.speech_detected;
  $("btn-verdict-next").hidden = !ok;
}

$("btn-speech-preview").addEventListener("click", async () => {
  const res = await apiJson(`/api/sessions/${state.sessionId}/speech-preview`, { method: "POST" });
  const aud = $("speech-preview-audio");
  aud.src = res.wav_url;
  aud.hidden = false;
});

$("btn-verdict-rerecord").addEventListener("click", () => showStep(3));
$("btn-verdict-next").addEventListener("click", () => {
  renderHearBack(state.verdict);
  showStep(5);
});

// Screen 5: Hear-back
function renderHearBack(v) {
  $("hb-orig-audio").src = v.audio_urls.original;
  $("hb-synth-audio").src = v.audio_urls.heard_after;
  renderPitchAndBoundariesSVG(v);
  const mv = v.bpdr_multiview;
  const mvBox = $("bpdr-multiview-summary");
  if (mv && mvBox) {
    mvBox.hidden = false;
    mvBox.innerHTML = `
      <strong>BPDR 3-View Boundary-Invariance Controller (Proposal Fig. 2 Steps 5–6):</strong>
      Mean boundary sensitivity across nominal, false-split, and false-merge views:
      <span class="mono"><strong>${mv.mean_boundary_sensitivity_cents ?? 0} cents</strong></span>
      (threshold &le; ${mv.sensitivity_threshold_cents ?? 6.0} c) ·
      View-invariant spans: <span class="mono">${(mv.view_invariant_span_ids || []).length}</span> ·
      Ambiguous spans routed to FR5 review: <span class="mono">${(mv.ambiguous_span_ids || []).length}</span>
    `;
  }
}

function renderPitchAndBoundariesSVG(v) {
  const box = $("hearback-svg-box");
  const W = 860, H = 200;
  const dur = Math.max(1.0, v.duration_s || 1.0);
  const pt = v.pitch_track || { t_s: [], f0_hz: [], voiced: [] };
  const peaks = (v.boundary_hints && v.boundary_hints.peaks_s) || [];
  const spans = (v.span_diagnostics && v.span_diagnostics.spans) || [];

  const midiVals = pt.f0_hz
    .map((f, i) => (pt.voiced[i] && f > 0 ? 69 + 12 * Math.log2(f / 440) : null))
    .filter((m) => m !== null);
  const minM = midiVals.length ? Math.min(...midiVals) - 2 : 55;
  const maxM = midiVals.length ? Math.max(...midiVals) + 2 : 75;
  const xMap = (t) => 36 + (t / dur) * (W - 52);
  const yMap = (m) => H - 24 - ((m - minM) / Math.max(1, maxM - minM)) * (H - 42);

  let candRects = "";
  spans.filter((s) => s.review_repair_candidate).forEach((s) => {
    const x0 = xMap(s.start_s);
    const w = Math.max(4, xMap(s.end_s) - x0);
    candRects += `<rect x="${x0.toFixed(1)}" y="14" width="${w.toFixed(1)}" height="${H - 36}" fill="rgba(237,195,125,0.16)" stroke="#edc37d" stroke-width="1"/>`;
  });

  let peakLines = "";
  peaks.forEach((p) => {
    const xp = xMap(p);
    peakLines += `<line x1="${xp.toFixed(1)}" y1="14" x2="${xp.toFixed(1)}" y2="${H - 22}" stroke="#6cc9b8" stroke-dasharray="4,3" stroke-width="1.2"/>`;
  });

  let pathD = "";
  let penUp = true;
  for (let i = 0; i < pt.t_s.length; i++) {
    const f = pt.f0_hz[i];
    if (pt.voiced[i] && f > 0) {
      const m = 69 + 12 * Math.log2(f / 440);
      const px = xMap(pt.t_s[i]).toFixed(1);
      const py = yMap(m).toFixed(1);
      pathD += penUp ? `M ${px} ${py} ` : `L ${px} ${py} `;
      penUp = false;
    } else {
      penUp = true;
    }
  }

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" aria-label="Pitch contour and FastTNet boundary hints">
      ${candRects}
      ${peakLines}
      <path d="${pathD}" fill="none" stroke="#8ee5a1" stroke-width="2.2"/>
      <line x1="36" y1="${H - 22}" x2="${W - 16}" y2="${H - 22}" stroke="#2a3a37"/>
      <text x="36" y="${H - 6}" fill="#9eb0a6" font-size="11">0.0 s</text>
      <text x="${W - 60}" y="${H - 6}" fill="#9eb0a6" font-size="11">${dur.toFixed(1)} s</text>
      <text x="38" y="14" fill="#6cc9b8" font-size="11">Dashed: FastTNet boundaries · Shaded: pitch clarification candidates</text>
    </svg>
  `;
}

$("btn-hb-yes").addEventListener("click", async () => {
  await apiJson(`/api/sessions/${state.sessionId}/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed_as_heard: true, answers: [] }),
  });
  await loadHandoffAndShow();
});

$("btn-hb-clarify").addEventListener("click", async () => {
  const qRes = await apiJson(`/api/sessions/${state.sessionId}/questions?strategy=eig&k=3`);
  state.questionsPayload = qRes;
  state.qIndex = 0;
  state.answers = [];
  if (!qRes.questions || qRes.questions.length === 0) {
    await apiJson(`/api/sessions/${state.sessionId}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed_as_heard: true, answers: [] }),
    });
    await loadHandoffAndShow();
    return;
  }
  renderQuestionCard();
  showStep(6);
});

// Screen 6: Clarify cards + Keyboard shortcuts 1, 2, S
function renderQuestionCard() {
  const qs = state.questionsPayload.questions;
  const q = qs[state.qIndex];
  const total = qs.length;
  $("clarify-progress").textContent = `Card ${state.qIndex + 1} of ${total}`;

  const trust = state.questionsPayload.trust_curve?.mean_trust ?? 0.65;
  const answeredBoost = state.answers.filter((a) => a.choice !== "skip").length * 0.08;
  const liveTrust = Math.min(0.95, trust + answeredBoost);
  $("trust-meter-text").textContent = `${Math.round(liveTrust * 100)}%`;
  $("trust-meter-fill").style.width = `${Math.round(liveTrust * 100)}%`;

  const factorBadge = q.factor === "boundary" ? "Boundary ambiguity" : "Pitch ambiguity";
  $("question-card-box").innerHTML = `
    <div class="notice">
      <span class="proposal-tag">${factorBadge}</span>
      <span class="mono small muted" style="margin-left:8px;">EIG ${q.eig_bits} bits · Score ${q.eig_value_score}</span>
      <p style="margin-top:8px; font-size:16px;"><strong>${q.question_text}</strong></p>
    </div>
    <div class="ab-grid">
      <div class="ab-box">
        <strong>Original recording excerpt</strong>
        <audio controls src="${q.clips.orig_url}"></audio>
      </div>
      <div class="ab-box">
        <strong>Option 1: ${q.option_a.label}</strong>
        <audio controls src="${q.clips.opt_a_url}"></audio>
        <div class="btn-row"><button class="btn btn-primary" data-choice="optA">Choose Option 1 [Key 1]</button></div>
      </div>
      <div class="ab-box">
        <strong>Option 2: ${q.option_b.label}</strong>
        <audio controls src="${q.clips.opt_b_url}"></audio>
        <div class="btn-row"><button class="btn btn-primary" data-choice="optB">Choose Option 2 [Key 2]</button></div>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn" data-choice="skip">I can't tell / Skip [Key S]</button>
    </div>
  `;

  $("question-card-box").querySelectorAll("button[data-choice]").forEach((btn) => {
    btn.addEventListener("click", () => submitSingleCardChoice(btn.dataset.choice));
  });
}

async function submitSingleCardChoice(choice) {
  const qs = state.questionsPayload.questions;
  const q = qs[state.qIndex];
  state.answers.push({ qid: q.qid, choice, timestamp_utc: new Date().toISOString() });
  state.qIndex += 1;
  if (state.qIndex < qs.length) {
    renderQuestionCard();
    return;
  }

  const ansRes = await apiJson(`/api/sessions/${state.sessionId}/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed_as_heard: false, answers: state.answers }),
  });

  if (ansRes.needs_repair_review) {
    state.approvedSpanIds = new Set(ansRes.centred_span_ids);
    const repRes = await apiJson(`/api/sessions/${state.sessionId}/repair`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved_span_ids: Array.from(state.approvedSpanIds) }),
    });
    state.repairResult = repRes;
    renderRepairReview(repRes);
    showStep(7);
  } else {
    await loadHandoffAndShow();
  }
}

document.addEventListener("keydown", (e) => {
  if (state.currentStep !== 6) return;
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.key === "1") submitSingleCardChoice("optA");
  else if (e.key === "2") submitSingleCardChoice("optB");
  else if (e.key.toLowerCase() === "s") submitSingleCardChoice("skip");
});

// Screen 7: Review Changes
function renderRepairReview(rep) {
  const box = $("repair-review-list");
  if (!rep.edits || rep.edits.length === 0) {
    box.innerHTML = `<div class="notice">No spans required adjustment.</div>`;
    return;
  }
  box.innerHTML = rep.edits.map((ed) => {
    const ok = ed.verified;
    return `
      <div class="card" style="padding:16px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <strong>Span #${ed.span_id} · ${ed.note_name} (${ed.start_s.toFixed(2)}–${ed.end_s.toFixed(2)} s)</strong>
          <span class="status-badge ${ok ? "ok" : "warn"}">${ed.action}</span>
        </div>
        <p class="small muted" style="margin-top:6px;">
          Before deviation: ${ed.before_deviation_cents} c · Shift: ${ed.applied_shift_cents} c ·
          After deviation: ${ed.after_deviation_cents ?? "rolled back"} c
        </p>
        <div class="ab-grid">
          <div class="ab-box"><strong>Original</strong><audio controls src="${ed.before_url}"></audio></div>
          <div class="ab-box"><strong>Prepared</strong><audio controls src="${ed.after_url}"></audio></div>
        </div>
        ${ok ? `<label style="cursor:pointer;"><input type="checkbox" class="span-keep-chk" data-span-id="${ed.span_id}" checked> Keep this centring edit</label>` : `<span class="small muted">Automatically rolled back because post-render verification did not confirm improvement.</span>`}
      </div>
    `;
  }).join("");
}

$("btn-apply-recheck").addEventListener("click", async () => {
  const keepIds = Array.from(document.querySelectorAll(".span-keep-chk"))
    .filter((el) => el.checked)
    .map((el) => Number(el.dataset.spanId));
  await apiJson(`/api/sessions/${state.sessionId}/repair`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved_span_ids: keepIds }),
  });
  await loadHandoffAndShow();
});

// Screen 8: Hand-off Receipt
async function loadHandoffAndShow() {
  const hf = await apiJson(`/api/sessions/${state.sessionId}/handoff`);
  $("handoff-summary").innerHTML = `
    <div class="notice ok">
      <strong>Status: ${hf.status}</strong> · Processing time: ${hf.processing_elapsed_s ?? "—"} s<br>
      <span class="mono small">accepted.wav SHA-256: ${hf.accepted_sha256 || "None (not accepted)"}</span>
    </div>
    ${hf.accepted_audio_url ? `<audio controls src="${hf.accepted_audio_url}" style="width:100%; margin:10px 0;"></audio>` : ""}
  `;

  // Trust curve SVG
  const tc = hf.trust_curve || { times_s: [], trust: [] };
  if (tc.times_s && tc.times_s.length) {
    const W = 800, H = 110;
    const bars = tc.trust.map((v, i) => {
      const bw = (W - 40) / tc.trust.length;
      const bh = v * (H - 30);
      return `<rect x="${(20 + i * bw).toFixed(1)}" y="${(H - 18 - bh).toFixed(1)}" width="${Math.max(2, bw - 2).toFixed(1)}" height="${bh.toFixed(1)}" fill="#8ee5a1" opacity="0.8"/>`;
    }).join("");
    $("trust-curve-box").innerHTML = `
      <div class="small muted" style="margin-bottom:6px;">Proposed Calibrated Melody Trust Curve (mean ${(tc.mean_trust * 100).toFixed(0)}%) <span class="proposal-tag">PROPOSAL</span></div>
      <svg viewBox="0 0 ${W} ${H}">${bars}</svg>
    `;
  }

  const spans = (hf.intent_map && hf.intent_map.spans) || [];
  $("intent-map-table").innerHTML = `
    <thead><tr><th>Span</th><th>Time (s)</th><th>Note</th><th>State</th><th>p_split</th><th>p_inflection</th><th>Trust</th></tr></thead>
    <tbody>
      ${spans.map((s) => `
        <tr>
          <td>#${s.span_id}</td>
          <td class="mono">${s.t0}–${s.t1}</td>
          <td>${s.note_name}</td>
          <td><span class="mono">${s.state}</span></td>
          <td class="mono">${s.p_split}</td>
          <td class="mono">${s.p_inflection}</td>
          <td class="mono">${s.span_trust}</td>
        </tr>
      `).join("")}
    </tbody>
  `;
  $("handoff-json").textContent = JSON.stringify(hf, null, 2);
  showStep(8);
}

$("btn-another-take").addEventListener("click", () => {
  showStep(3);
});

$("btn-delete-now").addEventListener("click", async () => {
  const res = await apiJson(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
  $("delete-status").hidden = false;
  $("delete-status").textContent = res.deleted
    ? `Session ${res.session_id} and all audio files have been permanently deleted.`
    : "Session folder was already removed.";
  $("session-pill").textContent = "Session deleted";
});
