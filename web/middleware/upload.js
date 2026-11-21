const multer = require('multer');
const path = require('path');
const fs = require('fs');

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    let targetDir = path.resolve(__dirname, '../uploads');

    if (req.baseUrl.includes('pipeline') || req.baseUrl.includes('training')) {
      const datasetId = req.params.id || req.params.datasetId || 'temp';
      const label = file.fieldname === 'defect_images' ? 'defect' : 'ok';
      targetDir = path.resolve(__dirname, `../uploads/train/${datasetId}/${label}`);
    } else if (req.baseUrl.includes('inspection')) {
      const sessionId = req.params.inspectionId || req.body.sessionId || 'session_' + Date.now();
      targetDir = path.resolve(__dirname, `../uploads/inspect/${sessionId}`);
    }

    fs.mkdirSync(targetDir, { recursive: true });
    cb(null, targetDir);
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    const baseName = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_.-]/g, '_');
    cb(null, baseName + ext); // Giữ tên gốc (không thêm timestamp để tránh trùng)
  }
});

const fileFilter = (req, file, cb) => {
  const allowed = ['.png', '.jpg', '.jpeg'];
  const ext = path.extname(file.originalname).toLowerCase();
  if (allowed.includes(ext)) cb(null, true);
  else cb(new Error(`Chi chap nhan: ${allowed.join(', ')}`), false);
};

module.exports = multer({
  storage,
  fileFilter,
  limits: { fileSize: 30 * 1024 * 1024 } // 30MB
});
