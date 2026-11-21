const path = require('path');
const fs = require('fs');
const db = require('../models/db-queries');
const bridge = require('../services/pythonBridge');

const DATA_ROOT = path.resolve(__dirname, '../../data');
const PYTHON_OUT_DIR = path.resolve(__dirname,
  '../../Detai_img_processing/solidity_qc/outputs');

function getDataDir(dataset) {
  const uploadDir = path.resolve(__dirname, `../uploads/train/${dataset.id}`);
  if (fs.existsSync(path.join(uploadDir, 'ok')) || fs.existsSync(path.join(uploadDir, 'defect'))) {
    return uploadDir;
  }
  return path.join(DATA_ROOT, dataset.name);
}

async function renderInspectionPage(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    // Chỉ hiện dataset đã hoàn thành pipeline (status=ready)
    const readyDatasets = datasets.filter(d => d.pipeline_step === 'ready' || d.status === 'ready');
    const inspections = await db.getAllInspections(10);
    res.render('inspection', {
      title: 'Phân Loại Ảnh Mới',
      datasets: readyDatasets,
      allDatasets: datasets,
      inspections,
      active: 'inspection'
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

async function renderInspectionSessionPage(req, res) {
  try {
    const id = req.params.id;
    const inspection = await db.getInspectionById(id);
    if (!inspection) return res.status(404).render('error', { title: '404', error: 'Không tìm thấy phiên kiểm tra.' });
    const results = await db.getInspectionResults(id);
    res.render('inspection-detail', {
      title: `Kết quả: ${inspection.name}`,
      inspection,
      results,
      active: 'inspection'
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

async function runBatchInspection(req, res) {
  try {
    const { dataset_id, session_name } = req.body;
    if (!dataset_id) return res.status(400).json({ success: false, error: 'Vui lòng chọn bộ dữ liệu.' });

    const files = req.files || [];
    if (!files.length) return res.status(400).json({ success: false, error: 'Vui lòng tải lên ít nhất 1 ảnh.' });

    const dataset = await db.getDatasetById(dataset_id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    // Bắt buộc pipeline phải hoàn thành
    if (dataset.pipeline_step !== 'ready' && dataset.status !== 'ready') {
      return res.status(400).json({
        success: false,
        error: `Dataset "${dataset.name}" chưa hoàn thành pipeline 4 bước. Vui lòng vào trang Pipeline để hoàn thành trước.`
      });
    }

    // Tạo phiên kiểm tra trong DB
    const inspectionName = session_name ||
      `KiemTra_${dataset.name}_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '_')}`;
    const session = await db.createInspection({ dataset_id, name: inspectionName, total_images: files.length });

    const imagesDir = path.dirname(files[0].path);
    const outputResultsDir = path.resolve(__dirname, `../uploads/results/${session.id}`);
    fs.mkdirSync(outputResultsDir, { recursive: true });

    // Gọi inspect_batch với data_dir gốc (để build calibration từ OK images)
    const data_dir = getDataDir(dataset);
    const pythonOutput = await bridge.inspectBatch(
      dataset.name,
      imagesDir,
      outputResultsDir,
      {
        data_dir,
        threshold_k: dataset.threshold_k || 3.0
      }
    );

    const results = pythonOutput.results || [];
    let okCount = 0;
    let defectCount = 0;

    for (const r of results) {
      if (r.prediction === 'defect') defectCount++;
      else if (r.prediction === 'ok') okCount++;

      const feats = r.features || {};
      const relAnnotated = r.annotated_filename
        ? `/uploads/results/${session.id}/${r.annotated_filename}`
        : null;

      await db.insertInspectionResult({
        inspection_id: session.id,
        filename: r.filename,
        original_path: null,
        annotated_path: relAnnotated,
        prediction: r.prediction,
        anomaly_score: r.anomaly_score,
        solidity: feats.solidity,
        circularity: feats.circularity,
        edge_density: feats.edge_density,
        sobel_mean: feats.sobel_mean,
        gray_std: feats.gray_std,
        hist_dist: feats.hist_dist,
        patch_max: feats.patch_max,
        patch_top5: feats.patch_top5,
        patch_p90: feats.patch_p90,
        patch_mean: feats.patch_mean
      });
    }

    await db.updateInspectionStatus(session.id, {
      total_images: results.length,
      ok_count: okCount,
      defect_count: defectCount,
      status: 'completed'
    });

    res.json({
      success: true,
      inspection_id: session.id,
      name: inspectionName,
      total_images: results.length,
      ok_count: okCount,
      defect_count: defectCount,
      threshold_used: pythonOutput.threshold_used,
      results
    });
  } catch (err) {
    console.error('[runBatchInspection]', err);
    res.status(500).json({ success: false, error: err.message });
  }
}

async function exportInspectionCsv(req, res) {
  try {
    const inspection = await db.getInspectionById(req.params.id);
    if (!inspection) return res.status(404).send('Không tìm thấy phiên kiểm tra.');
    const results = await db.getInspectionResults(req.params.id);

    const headers = ['File', 'Du_doan', 'Z_score', 'Solidity', 'Circularity', 'Edge_density', 'Sobel_mean', 'Gray_std', 'Hist_dist'];
    const rows = results.map(r => [
      `"${r.filename}"`, r.prediction,
      r.anomaly_score?.toFixed(3) ?? '',
      r.solidity?.toFixed(4) ?? '',
      r.circularity?.toFixed(4) ?? '',
      r.edge_density?.toFixed(4) ?? '',
      r.sobel_mean?.toFixed(2) ?? '',
      r.gray_std?.toFixed(2) ?? '',
      r.hist_dist?.toFixed(4) ?? ''
    ]);

    const csv = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="ket_qua_${req.params.id}.csv"`);
    res.send('\uFEFF' + csv);
  } catch (err) {
    res.status(500).send(err.message);
  }
}

module.exports = { renderInspectionPage, renderInspectionSessionPage, runBatchInspection, exportInspectionCsv };
