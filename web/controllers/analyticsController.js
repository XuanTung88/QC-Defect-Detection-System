const db = require('../models/db-queries');

async function renderDashboardPage(req, res) {
  try {
    const stats = await db.getDashboardStats();
    const datasets = await db.getAllDatasets();
    res.render('dashboard', { title: 'Bảng Điều Khiển - Tổng Quan', stats, datasets });
  } catch (err) {
    res.status(500).render('error', { error: err.message });
  }
}

async function renderAnalyticsPage(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    const inspections = await db.getAllInspections(20);
    res.render('analytics', { title: 'Phân Tích Chi Tiết', datasets, inspections });
  } catch (err) {
    res.status(500).render('error', { error: err.message });
  }
}

async function getStatsApi(req, res) {
  try {
    const stats = await db.getDashboardStats();
    res.json({ success: true, stats });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

async function getImageDetailApi(req, res) {
  try {
    const id = req.params.resultId;
    const result = await db.getInspectionResultById(id);
    if (!result) {
      return res.status(404).json({ success: false, error: 'Không tìm thấy dữ liệu ảnh.' });
    }
    res.json({ success: true, result });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

module.exports = {
  renderDashboardPage,
  renderAnalyticsPage,
  getStatsApi,
  getImageDetailApi
};
