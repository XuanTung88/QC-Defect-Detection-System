const express = require('express');
const path = require('path');
const cors = require('cors');
const expressLayouts = require('express-ejs-layouts');
require('dotenv').config();

const { testConnection } = require('./config/db');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// View Engine (EJS + Layouts)
app.use(expressLayouts);
app.set('layout', 'layout');
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static Files
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Serve Python outputs (figures từ pipeline)
const PYTHON_OUT_DIR = path.resolve(__dirname,
  '../Detai_img_processing/solidity_qc/outputs');
app.use('/python-outputs', express.static(PYTHON_OUT_DIR));

// Routes
app.use('/', require('./routes/index'));
app.use('/datasets', require('./routes/datasets'));
app.use('/pipeline', require('./routes/pipeline'));       // <-- Pipeline 4 bước (MỚI)
app.use('/training', require('./routes/training'));       // giữ lại tương thích cũ
app.use('/inspection', require('./routes/inspection'));
app.use('/analytics', require('./routes/analytics'));

// ── MODULE MỚI: Defect QC (độc lập, không ảnh hưởng pipeline cũ) ──
app.use('/defectqc', require('./routes/defectqc'));

// Static: ảnh annotated từ defect_qc (outputs/defect_qc/<session>/annotated/)
const DEFECT_QC_OUT = path.resolve(__dirname,
  '../Detai_img_processing/solidity_qc/outputs/defect_qc');
app.use('/defectqc-outputs', express.static(DEFECT_QC_OUT));

// 404
app.use((req, res) => {
  res.status(404).render('error', {
    title: '404 - Không tìm thấy trang',
    error: 'Trang bạn yêu cầu không tồn tại.'
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('[Server Error]:', err.stack);
  res.status(500).render('error', {
    title: 'Lỗi Hệ Thống',
    error: err.message || 'Đã có lỗi nội bộ xảy ra.'
  });
});

// Khởi động
app.listen(PORT, async () => {
  console.log('====================================================');
  console.log(`🚀 Solidity QC Vision chạy tại: http://localhost:${PORT}`);
  console.log('====================================================');
  await testConnection();
});
