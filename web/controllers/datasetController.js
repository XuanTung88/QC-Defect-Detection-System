const db = require('../models/db-queries');

async function renderDatasetsPage(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    res.render('datasets', { title: 'Quản lý Bộ dữ liệu', datasets });
  } catch (err) {
    res.status(500).render('error', { error: err.message });
  }
}

async function getDatasetsApi(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    res.json({ success: true, datasets });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

async function createDatasetApi(req, res) {
  try {
    const { name, description, channel, object_bright, crop, interior, threshold_k } = req.body;
    if (!name || name.trim().length === 0) {
      return res.status(400).json({ success: false, error: 'Tên bộ dữ liệu không được để trống.' });
    }
    const cleanName = name.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    const created = await db.createDataset({
      name: cleanName,
      description,
      channel: channel || 'gray',
      object_bright: object_bright === 'true' || object_bright === true,
      crop: parseInt(crop || '0', 10),
      interior: interior || 'filled',
      threshold_k: parseFloat(threshold_k || '3.0')
    });
    res.status(201).json({ success: true, dataset: created });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

async function deleteDatasetApi(req, res) {
  try {
    const id = req.params.id;
    const deleted = await db.deleteDataset(id);
    if (!deleted) {
      return res.status(404).json({ success: false, error: 'Không tìm thấy bộ dữ liệu cần xóa.' });
    }
    res.json({ success: true, message: 'Đã xóa bộ dữ liệu thành công.' });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

module.exports = {
  renderDatasetsPage,
  getDatasetsApi,
  createDatasetApi,
  deleteDatasetApi
};
