-- ====================================================================
-- SHAPE DEFECT DETECTION SYSTEM (SOLIDITY / CONVEX HULL)
-- POSTGRESQL DATABASE SCHEMA
-- ====================================================================

-- Thiết lập mã hóa client là UTF8 để tránh lỗi WIN1252 trên Windows
SET client_encoding = 'UTF8';

-- 0. Dọn dẹp bảng cũ nếu cần thiết lập lại
DROP TABLE IF EXISTS inspection_results CASCADE;
DROP TABLE IF EXISTS inspections CASCADE;
DROP TABLE IF EXISTS trained_models CASCADE;
DROP TABLE IF EXISTS training_features CASCADE;
DROP TABLE IF EXISTS training_images CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;

-- 1. Bảng lưu trữ loại đối tượng / bộ dữ liệu (Datasets)
CREATE TABLE datasets (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE,
    description   TEXT,
    channel       VARCHAR(10) DEFAULT 'gray' CHECK (channel IN ('gray', 'sat')),
    object_bright BOOLEAN DEFAULT TRUE,
    crop          INT DEFAULT 0,
    interior      VARCHAR(10) DEFAULT 'filled' CHECK (interior IN ('mask', 'filled')),
    status        VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'trained', 'error')),
    auc_score     FLOAT,
    threshold_k   FLOAT DEFAULT 3.0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Bảng lưu ảnh dùng để huấn luyện (Training Images)
CREATE TABLE training_images (
    id          SERIAL PRIMARY KEY,
    dataset_id  INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    filename    VARCHAR(255) NOT NULL,
    filepath    TEXT NOT NULL,
    label       VARCHAR(10) NOT NULL CHECK (label IN ('ok', 'defect')),
    file_size   INT,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Bảng trích xuất đặc trưng hình học + bề mặt của ảnh huấn luyện
CREATE TABLE training_features (
    id              SERIAL PRIMARY KEY,
    image_id        INT NOT NULL REFERENCES training_images(id) ON DELETE CASCADE,
    solidity        FLOAT,
    circularity     FLOAT,
    edge_density    FLOAT,
    sobel_mean      FLOAT,
    gray_std        FLOAT,
    hist_dist       FLOAT,
    z_score         FLOAT,
    contour_area    FLOAT,
    hull_area       FLOAT,
    perimeter       FLOAT,
    fill_ratio      FLOAT,
    patch_max       FLOAT,
    patch_top5      FLOAT,
    patch_p90       FLOAT,
    patch_mean      FLOAT
);

-- 4. Bảng lưu trữ mô hình và tham số hiệu chuẩn đã huấn luyện (Trained Models)
CREATE TABLE trained_models (
    id                   SERIAL PRIMARY KEY,
    dataset_id           INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    method               VARCHAR(50) NOT NULL DEFAULT 'baseline_maxz',
    calibration          JSONB,
    patch_bank_path      TEXT,
    auc                  FLOAT,
    precision_val        FLOAT,
    recall_val           FLOAT,
    f1_val               FLOAT,
    feature_importances  JSONB,
    trained_at           TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Bảng phiên kiểm tra phân loại hàng loạt (Inspection Sessions)
CREATE TABLE inspections (
    id              SERIAL PRIMARY KEY,
    dataset_id      INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    total_images    INT DEFAULT 0,
    ok_count        INT DEFAULT 0,
    defect_count    INT DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'error')),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Bảng chi tiết kết quả phân loại từng ảnh trong phiên kiểm tra
CREATE TABLE inspection_results (
    id                  SERIAL PRIMARY KEY,
    inspection_id       INT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    filename            VARCHAR(255) NOT NULL,
    original_path       TEXT NOT NULL,
    annotated_path      TEXT,
    prediction          VARCHAR(10) CHECK (prediction IN ('ok', 'defect', 'error')),
    anomaly_score       FLOAT,
    solidity            FLOAT,
    circularity         FLOAT,
    edge_density        FLOAT,
    sobel_mean          FLOAT,
    gray_std            FLOAT,
    hist_dist           FLOAT,
    patch_max           FLOAT,
    patch_top5          FLOAT,
    patch_p90           FLOAT,
    patch_mean          FLOAT,
    processed_at        TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Chỉ mục (Indexes)
CREATE INDEX idx_training_images_dataset ON training_images(dataset_id);
CREATE INDEX idx_training_features_image ON training_features(image_id);
CREATE INDEX idx_inspections_dataset ON inspections(dataset_id);
CREATE INDEX idx_inspection_results_session ON inspection_results(inspection_id);

-- 8. Thêm dữ liệu mặc định cho 3 dataset mẫu có sẵn
INSERT INTO datasets (name, description, channel, object_bright, crop, interior, status, auc_score, threshold_k)
VALUES 
('biscuits', 'Banh quy - Kiem tra khuyet tat hinh dang (me, sut canh, vo)', 'gray', TRUE, 0, 'mask', 'trained', 0.902, 3.0),
('hazelnut', 'Hat phi - Khuyet tat be mat (nut, lo sau, vet cat) va hinh dang', 'sat', TRUE, 0, 'filled', 'trained', 0.925, 3.0),
('pills', 'Vien thuoc - Khuyet tat tinh te vien thuoc va be mat', 'gray', TRUE, 0, 'filled', 'trained', 0.880, 3.0)
ON CONFLICT (name) DO NOTHING;
