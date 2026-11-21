/**
 * defectqc.js — Route cho module Defect QC
 * PREFIX: /defectqc
 * Độc lập, không ảnh hưởng các route cũ.
 */
const express = require('express');
const router  = express.Router();
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const ctrl    = require('../controllers/defectQcController');

// ── Multer riêng cho Defect QC ──────────────────────────────
const UPLOAD_BASE = path.resolve(__dirname, '../uploads/defectqc');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const isScan    = req.originalUrl.includes('/api/scan');
    const isInspect = req.originalUrl.includes('/api/inspect');

    // Sinh sessionId 1 lần duy nhất cho cả request (lưu trên req)
    if (!req._defectSessionId) {
      req._defectSessionId = 'sess_' + Date.now();
    }
    const sessionId = req._defectSessionId;

    let subDir;
    if (isScan) {
      subDir = file.fieldname === 'defect_images' ? 'defect' : 'ok';
    } else if (isInspect) {
      subDir = file.fieldname === 'ok_ref' ? 'ok_ref' : 'new_images';
    } else {
      subDir = 'misc';
    }

    const dir = path.join(UPLOAD_BASE, sessionId, subDir);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const ext  = path.extname(file.originalname).toLowerCase();
    const base = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_.-]/g, '_');
    cb(null, base + ext);
  }
});

const fileFilter = (req, file, cb) => {
  const allowed = ['.png', '.jpg', '.jpeg'];
  if (allowed.includes(path.extname(file.originalname).toLowerCase())) cb(null, true);
  else cb(new Error('Chỉ chấp nhận PNG, JPG, JPEG'), false);
};

const upload = multer({ storage, fileFilter, limits: { fileSize: 30 * 1024 * 1024 } });

// ── Routes ──────────────────────────────────────────────────

// Trang chính (giới thiệu + danh sách session)
router.get('/', ctrl.renderMainPage);

// Trang phân loại ảnh mới (không nhãn)
router.get('/inspect', ctrl.renderInspectPage);

// Xem chi tiết 1 session scan
router.get('/session/:name', ctrl.renderSessionDetail);

// API: Scan dataset (upload OK + Defect)
router.post('/api/scan',
  upload.fields([
    { name: 'ok_images',     maxCount: 300 },
    { name: 'defect_images', maxCount: 300 }
  ]),
  ctrl.runScan
);

// API: Phân loại ảnh mới
router.post('/api/inspect',
  upload.fields([
    { name: 'ok_ref',    maxCount: 200 },
    { name: 'new_images', maxCount: 200 }
  ]),
  ctrl.runInspect
);

// Xuất CSV
router.get('/api/session/:name/export', ctrl.exportSessionCsv);

module.exports = router;
