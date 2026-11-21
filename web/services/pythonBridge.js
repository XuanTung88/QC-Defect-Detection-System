/**
 * pythonBridge.js - Giao tiếp với web_bridge.py theo đúng 5-bước pipeline
 * Bước 1: explore → Trực quan hóa Otsu + Convex Hull
 * Bước 2: evaluate → Đánh giá one-class z-score AUC
 * Bước 3: benchmark → So sánh 5 phương pháp cross-validation
 * Bước 4: report → Phân tích lỗi FN/FP/TP
 * Phân loại: inspect_batch → Chỉ dùng sau khi hoàn thành 4 bước
 */
const { spawn } = require('child_process');
const path = require('path');
require('dotenv').config();

const PYTHON_BIN = process.env.PYTHON_PATH || 'python';
const BRIDGE_SCRIPT = process.env.BRIDGE_SCRIPT ||
  path.resolve(__dirname, '../../Detai_img_processing/solidity_qc/web_bridge.py');

/**
 * Hàm chung thực thi web_bridge.py với bộ tham số CLI
 */
function runBridgeCommand(commandArgs, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    const fullArgs = [BRIDGE_SCRIPT, ...commandArgs];
    console.log(`[PythonBridge] ${PYTHON_BIN} web_bridge.py ${commandArgs.join(' ')}`);

    const child = spawn(PYTHON_BIN, fullArgs, {
      cwd: path.dirname(BRIDGE_SCRIPT),
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
    });

    let stdoutData = '';
    let stderrData = '';

    // Timeout để tránh treo vô thời hạn
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error(`Python bridge timeout sau ${timeoutMs / 1000}s`));
    }, timeoutMs);

    child.stdout.on('data', (chunk) => { stdoutData += chunk.toString('utf-8'); });
    child.stderr.on('data', (chunk) => { stderrData += chunk.toString('utf-8'); });

    child.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Khong the khoi dong Python: ${err.message}`));
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      if (stderrData.trim()) {
        console.warn(`[PythonBridge stderr]:\n${stderrData.trim().slice(0, 1000)}`);
      }
      if (code !== 0) {
        return reject(new Error(
          `Python bridge thoat ma loi ${code}:\n${(stderrData || stdoutData).slice(0, 500)}`
        ));
      }
      // Tìm dòng JSON cuối cùng (đề phòng có output khác trước đó)
      const lines = stdoutData.trim().split('\n').filter(l => l.trim().startsWith('{'));
      const jsonStr = lines[lines.length - 1] || stdoutData.trim();
      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed.success === false) {
          return reject(new Error(parsed.error || 'Xu ly Python that bai.'));
        }
        resolve(parsed);
      } catch (parseErr) {
        console.error('[PythonBridge raw stdout]:', stdoutData.slice(0, 500));
        reject(new Error(`Loi parse JSON tu Python: ${parseErr.message}`));
      }
    });
  });
}

// ============================================================
// BƯỚC 1: explore - Trực quan hóa phân vùng Otsu + Convex Hull
// ============================================================
function explore(datasetName, options = {}) {
  const args = ['explore', '--dataset_name', datasetName];
  if (options.data_dir) args.push('--data_dir', options.data_dir);
  if (options.cfg) args.push('--cfg', JSON.stringify(options.cfg));
  if (options.out_dir) args.push('--out_dir', options.out_dir);
  if (options.n_per_class) args.push('--n_per_class', String(options.n_per_class));
  return runBridgeCommand(args, 120000);
}

// ============================================================
// BƯỚC 2: evaluate - One-class z-score AUC từng đặc trưng
// ============================================================
function evaluate(datasetName, options = {}) {
  const args = ['evaluate', '--dataset_name', datasetName];
  if (options.data_dir) args.push('--data_dir', options.data_dir);
  if (options.cfg) args.push('--cfg', JSON.stringify(options.cfg));
  if (options.out_dir) args.push('--out_dir', options.out_dir);
  if (options.reps) args.push('--reps', String(options.reps));
  if (options.cal_ratio) args.push('--cal_ratio', String(options.cal_ratio));
  if (options.threshold_k) args.push('--threshold_k', String(options.threshold_k));
  return runBridgeCommand(args, 180000);
}

// ============================================================
// BƯỚC 3: benchmark - So sánh 5 phương pháp cross-validation
// ============================================================
function benchmark(datasetName, options = {}) {
  const args = ['benchmark', '--dataset_name', datasetName];
  if (options.data_dir) args.push('--data_dir', options.data_dir);
  if (options.cfg) args.push('--cfg', JSON.stringify(options.cfg));
  if (options.out_dir) args.push('--out_dir', options.out_dir);
  if (options.splits) args.push('--splits', String(options.splits));
  if (options.reps) args.push('--reps', String(options.reps));
  return runBridgeCommand(args, 600000); // benchmark có thể mất nhiều thời gian
}

// ============================================================
// BƯỚC 4: report - Phân tích lỗi FN/FP/TP
// ============================================================
function report(datasetName, options = {}) {
  const args = ['report', '--dataset_name', datasetName];
  if (options.out_dir) args.push('--out_dir', options.out_dir);
  return runBridgeCommand(args, 120000);
}

// ============================================================
// PHÂN LOẠI: inspect_batch - Dùng SAU khi hoàn thành 4 bước
// ============================================================
function inspectBatch(datasetName, imagesDir, outputDir, options = {}) {
  const args = [
    'inspect_batch',
    '--dataset_name', datasetName,
    '--images_dir', imagesDir,
    '--output_dir', outputDir
  ];
  if (options.data_dir) args.push('--data_dir', options.data_dir);
  if (options.model_path) args.push('--model_path', options.model_path);
  if (options.threshold_k) args.push('--threshold_k', String(options.threshold_k));
  if (options.cfg) args.push('--cfg', JSON.stringify(options.cfg));
  return runBridgeCommand(args, 300000);
}

// ============================================================
// MODULE MỚI: defect_scan — Quét dataset (ok/ + defect/)
// ============================================================
function defectScan(dataDir, options = {}) {
  const args = ['defect_scan', '--data_dir', dataDir];
  if (options.channel)       args.push('--channel', options.channel);
  if (options.object_bright !== undefined) args.push('--object_bright', String(options.object_bright));
  if (options.threshold_k)   args.push('--threshold_k', String(options.threshold_k));
  if (options.reps)          args.push('--reps', String(options.reps));
  if (options.out_dir)       args.push('--out_dir', options.out_dir);
  if (options.session_name)  args.push('--session_name', options.session_name);
  return runBridgeCommand(args, 240000);
}

// ============================================================
// MODULE MỚI: defect_inspect — Phân loại batch ảnh mới
// ============================================================
function defectInspect(imagesDir, okDir, outDir, options = {}) {
  const args = [
    'defect_inspect',
    '--images_dir', imagesDir,
    '--ok_dir',     okDir,
    '--out_dir',    outDir
  ];
  if (options.channel)        args.push('--channel', options.channel);
  if (options.object_bright !== undefined) args.push('--object_bright', String(options.object_bright));
  if (options.threshold_k)    args.push('--threshold_k', String(options.threshold_k));
  return runBridgeCommand(args, 240000);
}

module.exports = { explore, evaluate, benchmark, report, inspectBatch, defectScan, defectInspect };
