const express = require('express');
const router = express.Router();
const ctrl = require('../controllers/pipelineController');
const upload = require('../middleware/upload');

// Trang danh sách dataset + trạng thái pipeline
router.get('/', ctrl.renderPipelinePage);

// Trang xem Output chi tiết từng bước (đặt rõ trước các route khác nếu cần)
router.get('/:id/annotated', ctrl.renderAnnotatedPage);
router.get('/:id/annotated-v2', ctrl.renderAnnotatedV2Page);
router.get('/:id/report-table', ctrl.renderReportTablePage);
router.get('/:id/download/:type', ctrl.downloadOutputFile);

// Trang pipeline từng dataset (4 bước)
router.get('/:id', ctrl.renderPipelineDetail);

// Upload ảnh OK + Defect cho dataset tùy chỉnh
router.post('/:id/upload',
  upload.fields([
    { name: 'ok_images', maxCount: 200 },
    { name: 'defect_images', maxCount: 200 }
  ]),
  ctrl.uploadImages
);

// Bước 1: Explore
router.post('/:id/explore', ctrl.runExplore);

// Bước 2: Evaluate
router.post('/:id/evaluate', ctrl.runEvaluate);

// Bước 3: Benchmark
router.post('/:id/benchmark', ctrl.runBenchmark);

// Bước 4: Report (hoàn thành → status='trained', pipeline_step='ready')
router.post('/:id/report', ctrl.runReport);

module.exports = router;
