/**
 * defectQcController.js
 * Controller cho module Defect QC — phát hiện sứt mẻ / vỡ / khuyết tật cấu trúc
 * trên vật thể có hình dạng khép kín (tròn, oval, đa giác...).
 * Hoàn toàn độc lập, KHÔNG dùng pipeline cũ.
 */
const path = require('path');
const fs   = require('fs');
const bridge = require('../services/pythonBridge');

const PYTHON_OUT_DIR = path.resolve(__dirname, '../../Detai_img_processing/solidity_qc/outputs');
const DEFECT_QC_OUT  = path.join(PYTHON_OUT_DIR, 'defect_qc');
const UPLOAD_BASE    = path.resolve(__dirname, '../uploads/defectqc');

const IMG_EXTS = new Set(['.png', '.jpg', '.jpeg']);

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }

function listImages(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => IMG_EXTS.has(path.extname(f).toLowerCase())).sort();
}

function readJsonSafe(p) {
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return null; }
}

// ─────────────────────────────────────────────────────────────
// GET /defectqc — Trang chính (upload + hướng dẫn)
// ─────────────────────────────────────────────────────────────
async function renderMainPage(req, res) {
  try {
    const sessions = [];
    if (fs.existsSync(DEFECT_QC_OUT)) {
      for (const name of fs.readdirSync(DEFECT_QC_OUT).sort().reverse()) {
        const dir = path.join(DEFECT_QC_OUT, name);
        if (!fs.statSync(dir).isDirectory()) continue;
        const metaPath = path.join(dir, 'scan_result.json');
        const meta = readJsonSafe(metaPath);
        sessions.push({ name, meta, hasResult: !!meta });
      }
    }
    res.render('defectqc-main', {
      title: 'Defect QC — Phát Hiện Sứt Mẻ / Vỡ',
      active: 'defectqc',
      sessions
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ─────────────────────────────────────────────────────────────
// POST /defectqc/api/scan — Upload ok/ + defect/ → chạy defect_scan
// ─────────────────────────────────────────────────────────────
async function runScan(req, res) {
  try {
    // sessionId: ưu tiên tên do user đặt (body), fallback timestamp
    const sessionName = (req.body.session_name || ('scan_' + Date.now())).replace(/[^a-zA-Z0-9_-]/g, '_');
    const channel      = req.body.channel || 'gray';
    const objectBright = req.body.object_bright !== 'false';
    const thresholdK   = parseFloat(req.body.threshold_k || '3.0');
    const reps         = parseInt(req.body.reps || '20');

    const okFiles     = (req.files['ok_images']     || []);
    const defectFiles = (req.files['defect_images'] || []);

    if (okFiles.length < 3) {
      return res.status(400).json({ success: false, error: 'Cần ít nhất 3 ảnh OK để hiệu chuẩn.' });
    }
    if (defectFiles.length < 1) {
      return res.status(400).json({ success: false, error: 'Cần ít nhất 1 ảnh Defect.' });
    }

    // Multer lưu file vào uploads/defectqc/<sessionId>/ok và .../defect
    // sessionId trong multer được tạo bằng timestamp (req._defectSessionId).
    // Vì req.body chưa parse tại thời multer destination chạy, ta dùng session ID từ multer.
    const uploadedSessionId = req._defectSessionId || sessionName;
    const dataDir = path.join(UPLOAD_BASE, uploadedSessionId);
    const outDir  = path.join(DEFECT_QC_OUT, sessionName);
    ensureDir(outDir);

    const result = await bridge.defectScan(dataDir, {
      channel,
      object_bright: objectBright,
      threshold_k: thresholdK,
      reps,
      out_dir: outDir,
      session_name: sessionName
    });

    fs.writeFileSync(path.join(outDir, 'scan_result.json'), JSON.stringify(result, null, 2), 'utf-8');
    res.json({ success: true, session_name: sessionName, ...result });
  } catch (err) {
    console.error('[defectScan]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

// ─────────────────────────────────────────────────────────────
// GET /defectqc/session/:name — Xem chi tiết 1 session scan
// ─────────────────────────────────────────────────────────────
async function renderSessionDetail(req, res) {
  try {
    const name    = req.params.name;
    const outDir  = path.join(DEFECT_QC_OUT, name);
    const annDir  = path.join(outDir, 'annotated');
    const meta    = readJsonSafe(path.join(outDir, 'scan_result.json'));

    if (!meta) {
      return res.status(404).render('error', { title: '404', error: 'Không tìm thấy session.' });
    }

    const items = (meta.results || []).map(r => {
      const isCorrect = r.label ? (r.pred === r.label) : null;
      const annFname  = r.label ? `${r.label}__${r.file}` : r.file;
      return {
        ...r,
        imgUrl: `/defectqc-outputs/${name}/annotated/${annFname}`,
        isCorrect
      };
    });

    res.render('defectqc-session', {
      title: `Defect QC — Kết quả: ${name}`,
      active: 'defectqc',
      sessionName: name,
      meta,
      items
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ─────────────────────────────────────────────────────────────
// GET /defectqc/inspect — Trang phân loại ảnh mới
// ─────────────────────────────────────────────────────────────
async function renderInspectPage(req, res) {
  try {
    res.render('defectqc-inspect', {
      title: 'Defect QC — Phân Loại Ảnh Mới',
      active: 'defectqc'
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ─────────────────────────────────────────────────────────────
// POST /defectqc/api/inspect — Upload ok_ref + ảnh mới → defect_inspect
// ─────────────────────────────────────────────────────────────
async function runInspect(req, res) {
  try {
    const channel      = req.body.channel || 'gray';
    const objectBright = req.body.object_bright !== 'false';
    const thresholdK   = parseFloat(req.body.threshold_k || '3.0');

    const okFiles  = (req.files['ok_ref']     || []);
    const newFiles = (req.files['new_images'] || []);

    if (okFiles.length < 3) {
      return res.status(400).json({ success: false, error: 'Cần ít nhất 3 ảnh OK tham chiếu.' });
    }
    if (newFiles.length < 1) {
      return res.status(400).json({ success: false, error: 'Cần ít nhất 1 ảnh cần phân loại.' });
    }

    // Multer lưu vào uploads/defectqc/<req._defectSessionId>/{ok_ref, new_images}
    const sessionId  = req._defectSessionId || ('inspect_' + Date.now());
    const sessionDir = path.join(UPLOAD_BASE, sessionId);
    const okDir      = path.join(sessionDir, 'ok_ref');
    const newDir     = path.join(sessionDir, 'new_images');
    const outDir     = path.join(DEFECT_QC_OUT, sessionId);
    ensureDir(outDir);

    const result = await bridge.defectInspect(newDir, okDir, outDir, {
      channel,
      object_bright: objectBright,
      threshold_k: thresholdK
    });

    const enriched = (result.results || []).map(r => ({
      ...r,
      imgUrl: `/defectqc-outputs/${sessionId}/${r.annotated_fname}`
    }));

    res.json({ success: true, session_id: sessionId, ...result, results: enriched });
  } catch (err) {
    console.error('[defectInspect]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

// ─────────────────────────────────────────────────────────────
// GET /defectqc/api/session/:name/export — Xuất CSV
// ─────────────────────────────────────────────────────────────
const DEFECT_FEATS_CSV = ['solidity', 'convexity', 'extent', 'hole_ratio', 'boundary_roughness'];

async function exportSessionCsv(req, res) {
  try {
    const meta = readJsonSafe(path.join(DEFECT_QC_OUT, req.params.name, 'scan_result.json'));
    if (!meta || !meta.results) return res.status(404).send('Không có dữ liệu.');
    const cols = ['file', 'label', 'pred', 'score', ...DEFECT_FEATS_CSV];
    const header = cols.join(',');
    const rows = meta.results.map(r => cols.map(c => (r[c] !== undefined ? r[c] : '')).join(','));
    const csv = [header, ...rows].join('\n');
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="defect_qc_${req.params.name}.csv"`);
    res.send('\uFEFF' + csv);
  } catch (err) {
    res.status(500).send(err.message);
  }
}

module.exports = {
  renderMainPage,
  runScan,
  renderSessionDetail,
  renderInspectPage,
  runInspect,
  exportSessionCsv
};
