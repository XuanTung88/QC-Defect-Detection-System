const express = require('express');
const router = express.Router();
const trainingController = require('../controllers/trainingController');
const upload = require('../middleware/upload');

// Trang view
router.get('/', trainingController.renderTrainingPage);

// API upload ảnh huấn luyện (hỗ trợ cả ảnh OK và DEFECT)
router.post(
  '/api/:datasetId/upload',
  upload.fields([
    { name: 'ok_images', maxCount: 100 },
    { name: 'defect_images', maxCount: 100 }
  ]),
  trainingController.uploadTrainingImages
);

// API kích hoạt huấn luyện mô hình
router.post('/api/:datasetId/train', trainingController.trainModel);

module.exports = router;
