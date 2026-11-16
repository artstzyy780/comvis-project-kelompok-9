(function () {
  const cfg = window.APP_CONFIG || {};
  const models = cfg.models || {};

  const $ = (id) => document.getElementById(id);
  const primaryModel = $('primary_model');
  const secondaryModel = $('secondary_model');
  const compareEl = $('compare');
  const secondaryGroup = $('secondary_group');
  const conf = $('conf');
  const iou = $('iou');
  const confVal = $('conf_val');
  const iouVal = $('iou_val');
  const sourceType = $('source_type');
  const btnDetect = $('btn_detect');
  const statusEl = $('status');
  const outputArea = $('output_area');
  const primaryCard = $('primary_card');
  const secondaryCard = $('secondary_card');
  const illumBox = $('illumination_box');
  const compSummary = $('comparison_summary');
  const streamWrap = $('stream_wrap');
  const streamImg = $('stream_img');
  const camPreviewWrap = $('camera_preview_wrap');

  let cameraStream = null;
  let snapshotBlob = null;

  function setStatus(msg, type) {
    statusEl.textContent = msg || '';
    statusEl.className = 'status' + (type ? ' ' + type : '');
  }

  function updateDesc() {
    const m = models[primaryModel.value];
    $('primary_desc').textContent = m ? m.description : '';
  }

  function syncSourceUI() {
    const v = sourceType.value;
    document.querySelectorAll('.source-block').forEach(el => el.style.display = 'none');
    const active = $('source_' + v);
    if (active) active.style.display = 'block';
    $('static_actions').style.display = (v === 'camera_live') ? 'none' : 'block';
    streamWrap.style.display = (v === 'camera_live') ? 'block' : 'none';
    camPreviewWrap.style.display = (v === 'camera_snapshot') ? 'block' : 'none';
    if (v !== 'camera_live') stopLiveStream();
    if (v !== 'camera_snapshot') stopCameraPreview();
  }

  function renderMetrics(containerId, diag) {
    const el = $(containerId);
    if (!el || !diag) return;
    el.innerHTML = `
      <div class="metric"><div class="label">Pejalan Kaki</div><div class="value">${diag.person_count}</div></div>
      <div class="metric"><div class="label">Kendaraan</div><div class="value">${diag.vehicle_count}</div></div>
      <div class="metric"><div class="label">Latensi</div><div class="value">${Number(diag.latency_ms).toFixed(1)} ms</div></div>
      <div class="metric"><div class="label">FPS Inferensi</div><div class="value">${Number(diag.fps).toFixed(2)}</div></div>
    `;
  }

  function renderResult(data) {
    const p = data.primary;
    $('primary_title').textContent = p.model_name;
    $('primary_badge').textContent = (models[p.model_key] && models[p.model_key].badge) || '';
    $('primary_img').src = 'data:image/jpeg;base64,' + p.image;
    renderMetrics('primary_metrics', p.diagnostics);

    if (data.secondary) {
      const s = data.secondary;
      secondaryCard.style.display = 'block';
      outputArea.classList.remove('single');
      $('secondary_title').textContent = s.model_name;
      $('secondary_badge').textContent = (models[s.model_key] && models[s.model_key].badge) || '';
      $('secondary_img').src = 'data:image/jpeg;base64,' + s.image;
      renderMetrics('secondary_metrics', s.diagnostics);

      const diff = p.diagnostics.person_count - s.diagnostics.person_count;
      compSummary.style.display = 'block';
      if (diff > 0) {
        compSummary.innerHTML = `<strong>Perbandingan:</strong> Model utama mendeteksi ${diff} pejalan kaki lebih banyak dibanding model pembanding.`;
      } else if (diff === 0) {
        compSummary.innerHTML = `<strong>Perbandingan:</strong> Kedua model mendeteksi jumlah pejalan kaki yang sama (${p.diagnostics.person_count}).`;
      } else {
        compSummary.innerHTML = `<strong>Perbandingan:</strong> Model pembanding mendeteksi ${Math.abs(diff)} pejalan kaki lebih banyak pada ambang saat ini.`;
      }
    } else {
      secondaryCard.style.display = 'none';
      outputArea.classList.add('single');
      compSummary.style.display = 'none';
    }

    if (p.diagnostics.illumination) {
      const il = p.diagnostics.illumination;
      illumBox.style.display = 'block';
      illumBox.innerHTML = `<strong>Profil pencahayaan adaptif:</strong> ${il.mode_label}. ` +
        `Luminansi rata-rata: ${il.luminance} L*. Rasio glare: ${(il.glare_ratio * 100).toFixed(2)}%. ` +
        `Ambang efektif: ${il.effective_conf_threshold}.`;
    } else {
      illumBox.style.display = 'none';
    }
  }

  async function runDetect() {
    const stype = sourceType.value;
    if (stype === 'camera_live') return;
    setStatus('Memproses inferensi...', '');
    btnDetect.disabled = true;
    try {
      const fd = new FormData();
      fd.append('primary_model', primaryModel.value);
      fd.append('compare', compareEl.checked ? 'true' : 'false');
      fd.append('secondary_model', secondaryModel.value);
      fd.append('conf', conf.value);
      fd.append('iou', iou.value);
      fd.append('imgsz', $('imgsz').value);
      fd.append('adaptive', $('adaptive').checked ? 'true' : 'false');
      fd.append('source_type', stype);

      if (stype === 'preset') {
        fd.append('preset_filename', $('preset_select').value);
      } else if (stype === 'upload') {
        const f = $('image_file').files[0];
        if (!f) { setStatus('Pilih berkas citra terlebih dahulu.', 'error'); return; }
        fd.append('image', f);
      } else if (stype === 'camera_snapshot') {
        if (!snapshotBlob) { setStatus('Ambil snapshot kamera terlebih dahulu.', 'error'); return; }
        fd.append('image', snapshotBlob, 'snapshot.jpg');
      }

      const resp = await fetch(cfg.detectUrl, { method: 'POST', body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Gagal melakukan inferensi.');
      renderResult(data);
      setStatus('Inferensi selesai.', 'success');
    } catch (e) {
      setStatus(e.message || String(e), 'error');
    } finally {
      btnDetect.disabled = false;
    }
  }

  async function startCameraPreview() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      $('camera_video').srcObject = cameraStream;
      $('btn_capture').disabled = false;
      setStatus('Kamera aktif. Ambil snapshot untuk inferensi.', 'success');
    } catch (e) {
      setStatus('Tidak dapat mengakses kamera browser: ' + e.message, 'error');
    }
  }

  function stopCameraPreview() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    const v = $('camera_video');
    if (v) v.srcObject = null;
    const capBtn = $('btn_capture');
    if (capBtn) capBtn.disabled = true;
    snapshotBlob = null;
  }

  function captureSnapshot() {
    const video = $('camera_video');
    const canvas = $('camera_canvas');
    if (!video || !cameraStream) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      snapshotBlob = blob;
      setStatus('Snapshot diambil. Klik "Jalankan Deteksi" untuk memproses.', 'success');
    }, 'image/jpeg', 0.92);
  }

  function startLiveStream() {
    const params = new URLSearchParams({
      source: 'camera',
      cam_index: $('cam_index').value || '0',
      primary_model: primaryModel.value,
      secondary_model: secondaryModel.value,
      compare: compareEl.checked ? 'true' : 'false',
      conf: conf.value,
      iou: iou.value,
      adaptive: $('adaptive').checked ? 'true' : 'false',
      tracking: $('tracking').checked ? 'true' : 'false',
      imgsz: $('imgsz').value || '416'
    });
    streamImg.src = cfg.videoFeedUrl + '?' + params.toString();
    streamWrap.style.display = 'block';
    $('btn_start_stream').disabled = true;
    $('btn_stop_stream').disabled = false;
    setStatus('Stream kamera dimulai. Hentikan stream sebelum mengganti parameter.', 'success');
  }

  function stopLiveStream() {
    if (streamImg) streamImg.src = '';
    const bs = $('btn_start_stream');
    const bstop = $('btn_stop_stream');
    if (bs) bs.disabled = false;
    if (bstop) bstop.disabled = true;
  }

  // Wiring
  if (conf) conf.addEventListener('input', () => confVal.textContent = conf.value);
  if (iou) iou.addEventListener('input', () => iouVal.textContent = iou.value);
  if (compareEl) compareEl.addEventListener('change', () => {
    secondaryGroup.style.display = compareEl.checked ? 'block' : 'none';
  });
  if (primaryModel) primaryModel.addEventListener('change', updateDesc);
  if (sourceType) sourceType.addEventListener('change', syncSourceUI);
  if (btnDetect) btnDetect.addEventListener('click', runDetect);

  const btnStartCam = $('btn_start_cam');
  const btnCapture = $('btn_capture');
  const btnStartStream = $('btn_start_stream');
  const btnStopStream = $('btn_stop_stream');
  if (btnStartCam) btnStartCam.addEventListener('click', startCameraPreview);
  if (btnCapture) btnCapture.addEventListener('click', captureSnapshot);
  if (btnStartStream) btnStartStream.addEventListener('click', startLiveStream);
  if (btnStopStream) btnStopStream.addEventListener('click', () => { stopLiveStream(); setStatus('Stream dihentikan.', ''); });

  // Init
  updateDesc();
  syncSourceUI();
  secondaryGroup.style.display = compareEl && compareEl.checked ? 'block' : 'none';
  outputArea.classList.add('single');
  window.addEventListener('beforeunload', () => { stopCameraPreview(); stopLiveStream(); });
})();
