/**
 * pipelineController.js
 * Điều phối 4 bước pipeline: explore → evaluate → benchmark → report
 * Kèm các trang xem kết quả Output của Bước 2 (annotated), Bước 3 (annotated_v2), Bước 4 (report & table)
 */
const path = require('path');
const fs = require('fs');
const db = require('../models/db-queries');
const bridge = require('../services/pythonBridge');

const PYTHON_OUT_DIR = path.resolve(__dirname,
  '../../Detai_img_processing/solidity_qc/outputs');
const DATA_ROOT = path.resolve(__dirname, '../../data');
const BUILTIN_DATASETS = new Set(['biscuits', 'hazelnut', 'pills']);
const IMG_EXTS = new Set(['.png', '.jpg', '.jpeg']);

/** Lấy data_dir cho dataset: custom upload hoặc DATA_ROOT gốc */
function getDataDir(dataset) {
  const uploadDir = path.resolve(__dirname, `../uploads/train/${dataset.id}`);
  if (fs.existsSync(path.join(uploadDir, 'ok')) || fs.existsSync(path.join(uploadDir, 'defect'))) {
    return uploadDir;
  }
  return path.join(DATA_ROOT, dataset.name);
}

/** Lấy cfg từ DB nếu là custom dataset */
function getCustomCfg(dataset) {
  if (BUILTIN_DATASETS.has(dataset.name)) return undefined;
  return {
    channel: dataset.channel || 'gray',
    object_bright: dataset.object_bright !== false,
    crop: dataset.crop || 0,
    interior: dataset.interior || 'filled'
  };
}

/** Đọc CSV đơn giản thành mảng object */
function readCsvFile(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, 'utf-8').trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).filter(Boolean).map(line => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, idx) => {
      const v = (vals[idx] ?? '').trim();
      const num = Number(v);
      obj[h] = (v !== '' && !Number.isNaN(num) && h !== 'file' && h !== 'label' && h !== 'split' && h !== 'pred')
        ? num : v;
    });
    return obj;
  });
}

/** Đọc JSON an toàn */
function readJsonSafe(filePath) {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

/** Đọc danh sách ảnh trong thư mục outputs/annotated hoặc annotated_v2 */
function listDirImages(dirPath) {
  if (!fs.existsSync(dirPath)) return [];
  return fs.readdirSync(dirPath)
    .filter(f => IMG_EXTS.has(path.extname(f).toLowerCase()))
    .sort();
}

/** Tính quantile (giống np.quantile mặc định linear) */
function quantile(arr, q) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return s[base + 1] !== undefined ? s[base] + rest * (s[base + 1] - s[base]) : s[base];
}

// ----------------------------------------------------------------
// GET /pipeline - Danh sách dataset + trạng thái pipeline
// ----------------------------------------------------------------
async function renderPipelinePage(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    res.render('pipeline', {
      title: 'Pipeline Phân Tích',
      datasets,
      active: 'pipeline'
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ----------------------------------------------------------------
// GET /pipeline/:id - Trang pipeline từng dataset
// ----------------------------------------------------------------
async function renderPipelineDetail(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).render('error', { title: '404', error: 'Không tìm thấy dataset.' });

    const counts = await db.countTrainingImages(dataset.id);
    const figures = {
      explore: dataset.explore_fig ? dataset.explore_fig.replace(/\\/g, '/') : null,
      eval:    dataset.eval_fig    ? dataset.eval_fig.replace(/\\/g, '/')    : null,
      bench:   dataset.bench_fig   ? dataset.bench_fig.replace(/\\/g, '/')   : null,
      errors:  dataset.errors_fig  ? dataset.errors_fig.replace(/\\/g, '/')  : null,
    };

    const toWebUrl = (absPath) => {
      if (!absPath) return null;
      const rel = path.relative(PYTHON_OUT_DIR, absPath).replace(/\\/g, '/');
      return `/python-outputs/${rel}`;
    };

    // Đếm nhanh ảnh output bước 2 & bước 3 để hiển thị badge
    const annotatedCount = listDirImages(path.join(PYTHON_OUT_DIR, 'annotated', dataset.name)).length;
    const annotatedV2Count = listDirImages(path.join(PYTHON_OUT_DIR, 'annotated_v2', dataset.name)).length;

    res.render('pipeline-detail', {
      title: `Pipeline: ${dataset.name}`,
      dataset,
      counts,
      active: 'pipeline',
      figureUrls: {
        explore: toWebUrl(figures.explore),
        eval:    toWebUrl(figures.eval),
        bench:   toWebUrl(figures.bench),
        errors:  toWebUrl(figures.errors),
      },
      evalData:  dataset.eval_data  || null,
      benchData: dataset.bench_data || null,
      annotatedCount,
      annotatedV2Count
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ----------------------------------------------------------------
// GET /pipeline/:id/annotated - Page xem Output Bước 2 (outputs/annotated/<name>)
// ----------------------------------------------------------------
async function renderAnnotatedPage(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).render('error', { title: '404', error: 'Không tìm thấy dataset.' });

    const annDir = path.join(PYTHON_OUT_DIR, 'annotated', dataset.name);
    const files = listDirImages(annDir);
    const csvRows = readCsvFile(path.join(PYTHON_OUT_DIR, `results_${dataset.name}.csv`));
    const summaryJson = readJsonSafe(path.join(PYTHON_OUT_DIR, `summary_${dataset.name}.json`));
    const summaryTxtPath = path.join(PYTHON_OUT_DIR, `summary_${dataset.name}.txt`);
    const summaryTxt = fs.existsSync(summaryTxtPath) ? fs.readFileSync(summaryTxtPath, 'utf-8') : null;
    const evalFigExists = fs.existsSync(path.join(PYTHON_OUT_DIR, 'figures', `eval_${dataset.name}.png`));

    // Map CSV row theo `${label}__${file}` để ghép thông số chi tiết với từng ảnh
    const csvMap = new Map();
    for (const row of csvRows) {
      csvMap.set(`${row.label}__${row.file}`, row);
    }

    const items = files.map(fname => {
      const m = fname.match(/^(ok|defect)__(.+)$/i);
      const gtLabel = m ? m[1].toLowerCase() : 'unknown';
      const origName = m ? m[2] : fname;
      const meta = csvMap.get(fname) || {};
      return {
        filename: fname,
        origName,
        gtLabel,
        imgUrl: `/python-outputs/annotated/${dataset.name}/${fname}`,
        split: meta.split || '—',
        pred: meta.pred || null,
        score_all: meta.score_all ?? null,
        solidity: meta.solidity ?? null,
        circularity: meta.circularity ?? null,
        edge_density: meta.edge_density ?? null,
        sobel_mean: meta.sobel_mean ?? null,
        gray_std: meta.gray_std ?? null,
        hist_dist: meta.hist_dist ?? null,
        isCorrect: meta.pred ? meta.pred === gtLabel : null
      };
    });

    const stats = {
      total: items.length,
      gtOk: items.filter(x => x.gtLabel === 'ok').length,
      gtDefect: items.filter(x => x.gtLabel === 'defect').length,
      predOk: items.filter(x => x.pred === 'ok').length,
      predDefect: items.filter(x => x.pred === 'defect').length,
      calCount: items.filter(x => x.split === 'cal').length,
      testCount: items.filter(x => x.split === 'test').length,
    };

    res.render('pipeline-annotated', {
      title: `Output Bước 2 (Annotated): ${dataset.name}`,
      dataset,
      active: 'pipeline',
      items,
      csvRows,
      stats,
      summaryJson,
      summaryTxt,
      evalFigUrl: evalFigExists ? `/python-outputs/figures/eval_${dataset.name}.png` : null
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ----------------------------------------------------------------
// GET /pipeline/:id/annotated-v2 - Page xem Output Bước 3 (outputs/annotated_v2/<name>)
// ----------------------------------------------------------------
async function renderAnnotatedV2Page(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).render('error', { title: '404', error: 'Không tìm thấy dataset.' });

    const annDir = path.join(PYTHON_OUT_DIR, 'annotated_v2', dataset.name);
    const files = listDirImages(annDir);
    const scoreRows = readCsvFile(path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}_scores.csv`));
    const benchJson = readJsonSafe(path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}.json`));
    const benchTxtPath = path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}.txt`);
    const benchTxt = fs.existsSync(benchTxtPath) ? fs.readFileSync(benchTxtPath, 'utf-8') : null;
    const benchFigExists = fs.existsSync(path.join(PYTHON_OUT_DIR, 'figures', `bench_${dataset.name}.png`));

    const bestMethod = benchJson?.annot_method || 'baseline_maxz';
    const okScores = scoreRows.filter(r => r.label === 'ok').map(r => Number(r[bestMethod])).filter(v => !Number.isNaN(v));
    const opThr = okScores.length ? quantile(okScores, 0.90) : null;

    const scoreMap = new Map();
    for (const row of scoreRows) {
      scoreMap.set(row.file, row);
    }

    const items = files.map(fname => {
      const m = fname.match(/^(ok|defect)__(.+)$/i);
      const gtLabel = m ? m[1].toLowerCase() : 'unknown';
      const origName = m ? m[2] : fname;
      const meta = scoreMap.get(fname) || {};
      const bestScore = meta[bestMethod] !== undefined ? Number(meta[bestMethod]) : null;
      const pred = (bestScore !== null && opThr !== null) ? (bestScore > opThr ? 'defect' : 'ok') : null;
      let outcome = null;
      if (pred) {
        if (gtLabel === 'defect' && pred === 'defect') outcome = 'TP';
        else if (gtLabel === 'defect' && pred === 'ok') outcome = 'FN';
        else if (gtLabel === 'ok' && pred === 'defect') outcome = 'FP';
        else if (gtLabel === 'ok' && pred === 'ok') outcome = 'TN';
      }
      return {
        filename: fname,
        origName,
        gtLabel,
        imgUrl: `/python-outputs/annotated_v2/${dataset.name}/${fname}`,
        bestScore,
        pred,
        outcome,
        scores: {
          baseline_maxz: meta.baseline_maxz ?? null,
          patch_knn: meta.patch_knn ?? null,
          sup_G_lr: meta.sup_G_lr ?? null,
          sup_GS_lr: meta.sup_GS_lr ?? null,
          sup_GAS_rf: meta.sup_GAS_rf ?? null
        }
      };
    });

    const stats = {
      total: items.length,
      gtOk: items.filter(x => x.gtLabel === 'ok').length,
      gtDefect: items.filter(x => x.gtLabel === 'defect').length,
      tp: items.filter(x => x.outcome === 'TP').length,
      fn: items.filter(x => x.outcome === 'FN').length,
      fp: items.filter(x => x.outcome === 'FP').length,
      tn: items.filter(x => x.outcome === 'TN').length,
      bestMethod,
      opThr
    };

    res.render('pipeline-annotated-v2', {
      title: `Output Bước 3 (Annotated v2): ${dataset.name}`,
      dataset,
      active: 'pipeline',
      items,
      scoreRows,
      stats,
      benchJson,
      benchTxt,
      benchFigUrl: benchFigExists ? `/python-outputs/figures/bench_${dataset.name}.png` : null
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ----------------------------------------------------------------
// GET /pipeline/:id/report-table - Page xem Output Bước 4 (Report FN/FP/TP & Table tổng hợp)
// ----------------------------------------------------------------
async function renderReportTablePage(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).render('error', { title: '404', error: 'Không tìm thấy dataset.' });

    const fprTarget = 0.10;
    const benchJson = readJsonSafe(path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}.json`));
    const scoreRows = readCsvFile(path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}_scores.csv`));
    const bestMethod = benchJson?.annot_method || 'baseline_maxz';

    const okScores = scoreRows.filter(r => r.label === 'ok').map(r => Number(r[bestMethod])).filter(v => !Number.isNaN(v));
    const thr = okScores.length ? quantile(okScores, 1 - fprTarget) : null;

    const tpList = [], fnList = [], fpList = [], tnList = [];
    for (const r of scoreRows) {
      const score = Number(r[bestMethod]);
      if (Number.isNaN(score) || thr === null) continue;
      const predBad = score > thr;
      const isDef = r.label === 'defect';
      const entry = {
        file: r.file,
        label: r.label,
        score,
        imgUrl: `/python-outputs/annotated_v2/${dataset.name}/${r.file}`,
        allScores: {
          baseline_maxz: r.baseline_maxz,
          patch_knn: r.patch_knn,
          sup_G_lr: r.sup_G_lr,
          sup_GS_lr: r.sup_GS_lr,
          sup_GAS_rf: r.sup_GAS_rf
        }
      };
      if (predBad && isDef) tpList.push(entry);
      else if (!predBad && isDef) fnList.push(entry);
      else if (predBad && !isDef) fpList.push(entry);
      else tnList.push(entry);
    }

    // Sắp xếp giống report_figs.py: FN điểm thấp nhất trước, FP & TP điểm cao nhất trước
    fnList.sort((a, b) => a.score - b.score);
    fpList.sort((a, b) => b.score - a.score);
    tpList.sort((a, b) => b.score - a.score);

    const recall = (tpList.length + fnList.length) ? tpList.length / (tpList.length + fnList.length) : 0;
    const precision = (tpList.length + fpList.length) ? tpList.length / (tpList.length + fpList.length) : 0;
    const errorsFigExists = fs.existsSync(path.join(PYTHON_OUT_DIR, 'figures', `errors_${dataset.name}.png`));

    // Đọc bảng tổng hợp từ tất cả summary_*.json và benchmark_*.json (giống make_table.py)
    const outFiles = fs.existsSync(PYTHON_OUT_DIR) ? fs.readdirSync(PYTHON_OUT_DIR).sort() : [];
    const baseTableRows = [];
    for (const f of outFiles.filter(x => /^summary_.*\.json$/i.test(x))) {
      const s = readJsonSafe(path.join(PYTHON_OUT_DIR, f));
      if (!s) continue;
      const auc = s.auc || {};
      const prfAll = s.prf?.all || {};
      baseTableRows.push({
        dataset: s.dataset || f,
        n_ok: s.n_ok ?? 0,
        n_defect: s.n_defect ?? 0,
        auc_shape: auc.shape?.mean ?? null,
        auc_surface: auc.surface?.mean ?? null,
        auc_all: auc.all?.mean ?? null,
        recall: prfAll.recall ?? null,
        fpr: prfAll.fpr ?? null
      });
    }

    const benchTableRows = [];
    let benchMethods = [];
    for (const f of outFiles.filter(x => /^benchmark_.*\.json$/i.test(x))) {
      const b = readJsonSafe(path.join(PYTHON_OUT_DIR, f));
      if (!b || !b.methods) continue;
      if (!benchMethods.length) benchMethods = Object.keys(b.methods);
      benchTableRows.push({
        dataset: b.dataset || f,
        annot_method: b.annot_method || '',
        methods: b.methods
      });
    }

    const summaryMdPath = path.join(PYTHON_OUT_DIR, 'summary_all.md');
    const summaryMd = fs.existsSync(summaryMdPath) ? fs.readFileSync(summaryMdPath, 'utf-8') : null;

    res.render('pipeline-report', {
      title: `Output Bước 4 (Report & Table): ${dataset.name}`,
      dataset,
      active: 'pipeline',
      bestMethod,
      fprTarget,
      thr,
      recall,
      precision,
      tpList,
      fnList,
      fpList,
      tnList,
      errorsFigUrl: errorsFigExists ? `/python-outputs/figures/errors_${dataset.name}.png` : null,
      baseTableRows,
      benchMethods,
      benchTableRows,
      summaryMd
    });
  } catch (err) {
    res.status(500).render('error', { title: 'Lỗi', error: err.message });
  }
}

// ----------------------------------------------------------------
// GET /pipeline/:id/download/:type - Tải CSV / Markdown từ outputs/
// ----------------------------------------------------------------
async function downloadOutputFile(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).send('Không tìm thấy dataset.');
    const type = req.params.type;
    let target = null;
    if (type === 'results_csv') target = path.join(PYTHON_OUT_DIR, `results_${dataset.name}.csv`);
    else if (type === 'benchmark_csv') target = path.join(PYTHON_OUT_DIR, `benchmark_${dataset.name}_scores.csv`);
    else if (type === 'summary_md') target = path.join(PYTHON_OUT_DIR, 'summary_all.md');
    if (!target || !fs.existsSync(target)) return res.status(404).send('File output chưa tồn tại.');
    res.download(target);
  } catch (err) {
    res.status(500).send(err.message);
  }
}

// ----------------------------------------------------------------
// POST /pipeline/:id/upload - Upload ảnh OK + Defect cho dataset tùy chỉnh
// ----------------------------------------------------------------
async function uploadImages(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    const okFiles = req.files['ok_images'] || [];
    const defectFiles = req.files['defect_images'] || [];

    for (const f of okFiles) {
      await db.insertTrainingImage({ dataset_id: dataset.id, filename: f.filename, filepath: f.path, label: 'ok', file_size: f.size });
    }
    for (const f of defectFiles) {
      await db.insertTrainingImage({ dataset_id: dataset.id, filename: f.filename, filepath: f.path, label: 'defect', file_size: f.size });
    }

    const counts = await db.countTrainingImages(dataset.id);
    res.json({
      success: true,
      message: `Đã upload ${okFiles.length} ảnh OK và ${defectFiles.length} ảnh Defect.`,
      counts
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

// ----------------------------------------------------------------
// POST /pipeline/:id/explore - Bước 1
// ----------------------------------------------------------------
async function runExplore(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    const data_dir = getDataDir(dataset);
    if (!fs.existsSync(data_dir)) {
      return res.status(400).json({ success: false, error: `Không tìm thấy dữ liệu tại: ${data_dir}. Vui lòng upload ảnh trước.` });
    }

    const result = await bridge.explore(dataset.name, {
      data_dir,
      cfg: getCustomCfg(dataset),
      n_per_class: 4
    });

    await db.updateDatasetStatus(dataset.id, {
      pipeline_step: 'explore',
      explore_fig: result.figure_path || null
    });

    res.json({
      success: true,
      step: 'explore',
      figure_path: result.figure_path,
      samples: result.samples,
      total_ok: result.total_ok,
      total_defect: result.total_defect,
      processed: result.processed,
      console: result.console
    });
  } catch (err) {
    console.error('[runExplore]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

// ----------------------------------------------------------------
// POST /pipeline/:id/evaluate - Bước 2
// ----------------------------------------------------------------
async function runEvaluate(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    const allowedSteps = ['explore', 'evaluate', 'benchmark', 'ready'];
    if (!allowedSteps.includes(dataset.pipeline_step)) {
      return res.status(400).json({ success: false, error: 'Vui lòng chạy Bước 1 (Explore) trước.' });
    }

    const data_dir = getDataDir(dataset);
    const result = await bridge.evaluate(dataset.name, {
      data_dir,
      cfg: getCustomCfg(dataset),
      reps: 30,
      threshold_k: dataset.threshold_k || 3.0
    });

    const aucAll = result.auc && result.auc.all ? result.auc.all.mean : null;
    const evalFigPath = result.figure_path || path.join(PYTHON_OUT_DIR, 'figures', `eval_${dataset.name}.png`);

    await db.updateDatasetStatus(dataset.id, {
      pipeline_step: 'evaluate',
      auc_score: aucAll,
      eval_data: result.auc,
      eval_fig: fs.existsSync(evalFigPath) ? evalFigPath : null
    });

    res.json({
      success: true,
      step: 'evaluate',
      n_ok: result.n_ok,
      n_defect: result.n_defect,
      auc: result.auc,
      prf: result.prf,
      skipped: result.skipped,
      annotated_url: `/pipeline/${dataset.id}/annotated`
    });
  } catch (err) {
    console.error('[runEvaluate]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

// ----------------------------------------------------------------
// POST /pipeline/:id/benchmark - Bước 3
// ----------------------------------------------------------------
async function runBenchmark(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    const allowedSteps = ['evaluate', 'benchmark', 'ready'];
    if (!allowedSteps.includes(dataset.pipeline_step)) {
      return res.status(400).json({ success: false, error: 'Vui lòng chạy Bước 2 (Evaluate) trước.' });
    }

    const data_dir = getDataDir(dataset);
    const result = await bridge.benchmark(dataset.name, {
      data_dir,
      cfg: getCustomCfg(dataset),
      splits: 5,
      reps: 10
    });

    const benchFigPath = result.figure_path || path.join(PYTHON_OUT_DIR, 'figures', `bench_${dataset.name}.png`);

    await db.updateDatasetStatus(dataset.id, {
      pipeline_step: 'benchmark',
      bench_data: result.result || {},
      bench_fig: fs.existsSync(benchFigPath) ? benchFigPath : null
    });

    res.json({
      success: true,
      step: 'benchmark',
      result: result.result,
      figure_path: result.figure_path,
      console: result.console,
      annotated_v2_url: `/pipeline/${dataset.id}/annotated-v2`
    });
  } catch (err) {
    console.error('[runBenchmark]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

// ----------------------------------------------------------------
// POST /pipeline/:id/report - Bước 4 (Hoàn thành → status='trained', pipeline_step='ready')
// ----------------------------------------------------------------
async function runReport(req, res) {
  try {
    const dataset = await db.getDatasetById(req.params.id);
    if (!dataset) return res.status(404).json({ success: false, error: 'Không tìm thấy dataset.' });

    const allowedSteps = ['benchmark', 'ready'];
    if (!allowedSteps.includes(dataset.pipeline_step)) {
      return res.status(400).json({ success: false, error: 'Vui lòng chạy Bước 3 (Benchmark) trước.' });
    }

    const result = await bridge.report(dataset.name, {});

    const errorsFigPath = result.errors_figure_path ||
      path.join(PYTHON_OUT_DIR, 'figures', `errors_${dataset.name}.png`);

    // status của bảng datasets chỉ cho phép ('pending', 'trained', 'error') theo datasets_status_check
    await db.updateDatasetStatus(dataset.id, {
      status: 'trained',
      pipeline_step: 'ready',
      errors_fig: fs.existsSync(errorsFigPath) ? errorsFigPath : null
    });

    res.json({
      success: true,
      step: 'report',
      errors_figure_path: result.errors_figure_path,
      table_path: result.table_path,
      best_method: result.best_method,
      stats_summary: result.stats_summary,
      console: result.console,
      report_table_url: `/pipeline/${dataset.id}/report-table`
    });
  } catch (err) {
    console.error('[runReport]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
}

module.exports = {
  renderPipelinePage,
  renderPipelineDetail,
  renderAnnotatedPage,
  renderAnnotatedV2Page,
  renderReportTablePage,
  downloadOutputFile,
  uploadImages,
  runExplore,
  runEvaluate,
  runBenchmark,
  runReport
};
