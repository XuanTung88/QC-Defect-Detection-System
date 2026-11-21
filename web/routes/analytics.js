const express = require('express');
const router = express.Router();
const analyticsController = require('../controllers/analyticsController');

// Trang phân tích
router.get('/', analyticsController.renderAnalyticsPage);

// API số liệu thống kê Dashboard
router.get('/api/stats', analyticsController.getStatsApi);

// API chi tiết 1 ảnh kiểm tra
router.get('/api/image/:resultId', analyticsController.getImageDetailApi);

module.exports = router;
