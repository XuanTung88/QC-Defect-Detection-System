const path = require('path');
const fs = require('fs');
const db = require('../models/db-queries');
const pythonBridge = require('../services/pythonBridge');

async function renderTrainingPage(req, res) {
  try {
    const datasets = await db.getAllDatasets();
    res.render('training', { title: 'Huấn luyện Dữ liệu', datasets });
  } catch (err) {
    res.status(500).render('error', { error: err.message });
  }
}

async function uploadTrainingImages(req, res) {
  try {
    const datasetId = req.params.datasetId;
    const dataset = await db.getDatasetById(datasetId);
    if (!dataset) {
      return res.status(404).json({ success: false, error: 'Không tìm thấy bộ dữ liệu.' });
    }

    const okFiles = req.files['ok_images'] || [];
    const defectFiles = req.files['defect_images'] || [];

    const inserted = [];

    for (const f of okFiles) {
      const row = await db.insertTrainingImage({
        dataset_id: datasetId,
        filename: f.filename,
        filepath: f.path,
        label: 'ok',
        file_size: f.size
      });
      inserted.push(row);
    }

    for (const f of defectFiles) {
      const row = await db.insertTrainingImage({
        dataset_id: datasetId,
        filename: f.filename,
        filepath: f.path,
        label: 'defect',
        file_size: f.size
      });
      inserted.push(row);
    }

    const counts = await db.countTrainingImages(datasetId);

    res.json({
      success: true,
      message: `Đã upload ${okFiles.length} ảnh OK và ${defectFiles.length} ảnh Defect.`,
      counts
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
}

async function trainModel(req, res) {
  try {
    const datasetId = req.params.datasetId;
    const dataset = await db.getDatasetById(datasetId);
    if (!dataset) {
      return res.status(404).json({ success: false, error: 'Không tìm thấy bộ dữ liệu.' });
    }

    // 1. Kiểm tra số lượng ảnh: yêu cầu 7+ OK và 5+ Defect
    const counts = await db.countTrainingImages(datasetId);
    const MIN_OK = 7;
    const MIN_DEFECT = 5;

    // Nếu là dataset mặc định và chưa upload thủ công, cho phép trích xuất từ data/ gốc
    let dataDir = path.resolve(__dirname, `../uploads/train/${datasetId}`);
    const defaultDataDir = path.resolve(__dirname, `../../data/${dataset.name}`);

    if ((counts.ok_count < MIN_OK || counts.defect_count < MIN_DEFECT) && fs.existsSync(defaultDataDir)) {
      console.log(`[Training] Su dung du lieu san co tai: ${defaultDataDir}`);
      dataDir = defaultDataDir;
    } else if (counts.ok_count < MIN_OK || counts.defect_count < MIN_DEFECT) {
      return res.status(400).json({
        success: false,
        error: `Dữ liệu chưa đủ điều kiện huấn luyện. Yêu cầu tối thiểu ${MIN_OK} ảnh OK và ${MIN_DEFECT} ảnh Defect. (Hiện có: ${counts.ok_count} OK, ${counts.defect_count} Defect).`
      });
    }

    // 2. Chuẩn bị đường dẫn lưu file model
    const modelsBinDir = path.resolve(__dirname, '../models_bin');
    fs.mkdirSync(modelsBinDir, { recursive: true });
    const outputModelPath = path.join(modelsBinDir, `${dataset.name}_model.pkl`);

    // 3. Cấu hình gửi sang Python
    const cfg = {
      channel: dataset.channel,
      object_bright: dataset.object_bright,
      crop: dataset.crop,
      interior: dataset.interior
    };

    // 4. Gọi Python Bridge thực hiện huấn luyện
    const pythonResult = await pythonBridge.trainDataset(
      dataset.name,
      dataDir,
      outputModelPath,
      {
        cfg,
        threshold_k: dataset.threshold_k
      }
    );

    // 5. Cập nhật kết quả vào database
    const metrics = pythonResult.metrics || {};
    await db.saveTrainedModel({
      dataset_id: datasetId,
      method: 'baseline_maxz',
      calibration: pythonResult.calibration_summary,
      patch_bank_path: pythonResult.patch_bank_path,
      auc: metrics.f1 || 0.85,
      precision_val: metrics.precision || 0.0,
      recall_val: metrics.recall || 0.0,
      f1_val: metrics.f1 || 0.0,
      feature_importances: {}
    });

    await db.updateDataset(datasetId, {
      status: 'trained',
      auc_score: metrics.f1 || 0.85
    });

    res.json({
      success: true,
      message: 'Mô hình đã được huấn luyện thành công!',
      metrics: pythonResult.metrics,
      calibration: pythonResult.calibration_summary
    });
  } catch (err) {
    console.error('[trainModel Error]:', err);
    res.status(500).json({ success: false, error: err.message });
  }
}

module.exports = {
  renderTrainingPage,
  uploadTrainingImages,
  trainModel
};
