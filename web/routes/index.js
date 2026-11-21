const express = require('express');
const router = express.Router();
const analyticsController = require('../controllers/analyticsController');

// Trang chủ hiển thị Dashboard
router.get('/', analyticsController.renderDashboardPage);

module.exports = router;
