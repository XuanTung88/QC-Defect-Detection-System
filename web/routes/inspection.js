const express = require('express');
const router = express.Router();
const inspectionController = require('../controllers/inspectionController');
const upload = require('../middleware/upload');

// Trang view kiểm tra phân loại
router.get('/', inspectionController.renderInspectionPage);

// Xem chi tiết phiên kiểm tra đã thực hiện
router.get('/session/:id', inspectionController.renderInspectionSessionPage);

// API upload hàng loạt ảnh và kích hoạt phân loại
router.post(
  '/api/run',
  upload.array('inspection_images', 200),
  inspectionController.runBatchInspection
);

// Xuất file CSV kết quả
router.get('/api/:id/export', inspectionController.exportInspectionCsv);

module.exports = router;
