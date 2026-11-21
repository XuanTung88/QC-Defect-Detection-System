const express = require('express');
const router = express.Router();
const datasetController = require('../controllers/datasetController');

// Trang view
router.get('/', datasetController.renderDatasetsPage);

// API
router.get('/api/list', datasetController.getDatasetsApi);
router.post('/api/create', datasetController.createDatasetApi);
router.delete('/api/:id', datasetController.deleteDatasetApi);

module.exports = router;
