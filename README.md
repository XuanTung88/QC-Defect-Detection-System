# 🔬 Hệ Thống Phát Hiện Khuyết Tật Hình Dạng & Bề Mặt Sản Phẩm (Solidity / Convex Hull QC & Defect QC)

Dự án nghiên cứu và ứng dụng Thị giác máy tính (Computer Vision) kết hợp phương pháp phát hiện bất thường Một lớp (One-class Anomaly Detection) nhằm tự động kiểm tra ngoại quan và phát hiện lỗi sản phẩm công nghiệp.

---

## 📑 Mục Lục
1. [Kiến Trúc & Thuật Toán](#-kiến-trúc--thuật-toán)
2. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
3. [Yêu Cầu Hệ Thống & Cài Đặt](#-yêu-cầu-hệ-thống--cài-đặt)
4. [Hướng Dẫn Chạy Bằng Python CLI](#-hướng-dẫn-chạy-bằng-python-cli)
   - [Pipeline Chuẩn 4 Bước (3 Bộ Data Gốc)](#1-pipeline-chuẩn-4-bước-cho-3-bộ-data-gốc)
   - [Module Ngoại Lai Mới (Defect QC - Vật Thể Có Lỗ / Sứt Mẻ)](#2-module-ngoại-lai-mới-defect-qc)
5. [Hướng Dẫn Chạy Giao Diện Web](#-hướng-dẫn-chạy-giao-diện-web)
6. [Tóm Tắt Các Chỉ Số Đo Lường](#-tóm-tắt-các-chỉ-số-đo-lường)

---

## 🧠 Kiến Trúc & Thuật Toán

Hệ thống được thiết kế thành **2 luồng song song độc lập**, đảm bảo không ảnh hưởng đến độ chính xác chuẩn của 3 bộ dữ liệu gốc:

### 1. Luồng Chuẩn (Standard Pipeline - `solidity_qc`)
Phù hợp với các sản phẩm dạng đơn lẻ, kiểm soát nghiêm ngặt hình dạng và bề mặt:
- **`biscuits`**: Bánh quy tròn phẳng, viền hoa văn — phát hiện sứt mẻ viền qua **Solidity & Circularity** (AUC > 0.93).
- **`hazelnut`**: Hạt phỉ — phát hiện vết nứt, đốm sâu, vết cắt qua **Đặc trưng bề mặt & Ngân hàng ô PatchBank + KNN** (AUC > 0.98).
- **`pills`**: Viên thuốc nén — phát hiện mẻ vỡ góc tinh tế qua **Robust Z-Score One-Class** (AUC > 0.88).

**Quy trình 4 bước:**
1. **Explore**: Tiền xử lý (Median blur, HSV/Grayscale), phân vùng Otsu, tính Convex Hull và trực quan hóa phân vùng.
2. **Evaluate**: Hiệu chuẩn phân phối trên tập OK (Robust Z-score với Median & MAD), đánh giá AUC từng đặc trưng.
3. **Benchmark**: Đánh giá chéo lặp lại (Stratified K-Fold) so sánh 5 mô hình: `baseline_maxz`, `patch_knn`, `sup_G_lr`, `sup_GS_lr`, `sup_GAS_rf`.
4. **Report**: Phân tích các trường hợp lỗi FN/FP/TP, xuất bảng tổng hợp và biểu đồ ROC.

### 2. Module Ngoại Lai Mới (Defect QC - `defect_qc.py`)
Chuyên trách xử lý các trường hợp:
- **Vật thể có lỗ bên trong**: Vòng đệm (washers), đai ốc (nuts), bạc đạn, bánh răng...
- **Vật thể khép kín bất kỳ bị sứt mẻ / vỡ / nứt**: Không ép buộc theo một hình dạng chuẩn cố định, chỉ tập trung dò khuyết tật cấu trúc hình học.
- **5 chỉ số đặc thù**:
  - `solidity`: Tỉ lệ diện tích với bao lồi (bắt sứt mẻ, vỡ góc).
  - `convexity`: Tỉ lệ chu vi bao lồi / chu vi viền (bắt vết lõm, gồ ghề).
  - `extent`: Tỉ lệ diện tích vật / khung bao chữ nhật (bắt thiếu góc).
  - `hole_ratio`: Tỉ lệ diện tích lỗ bên trong (bắt lỗ thủng bất thường, biến dạng lỗ tâm).
  - `boundary_roughness`: Độ biến thiên bán kính cực quanh tâm (bắt nứt, viền vỡ gồ ghề).

---

## 📁 Cấu Trúc Thư Mục

```text
XuLyAnh/
├── data/                                  # Dữ liệu ảnh mẫu
│   ├── biscuits/                          # 48 ok / 67 defect
│   ├── hazelnut/                          # 22 ok / 17 defect
│   └── pills/                             # 26 ok / 24 defect
├── Detai_img_processing/
│   └── solidity_qc/
│       ├── src/
│       │   ├── config.py                  # Cấu hình 3 bộ dataset chuẩn
│       │   ├── dataset.py                 # Hàm nạp dữ liệu
│       │   ├── pipeline.py                # Pipeline trích xuất đặc trưng chuẩn
│       │   ├── patches.py                 # Thuật toán PatchBank + KNN cục bộ
│       │   ├── oneclass.py                # Robust Z-score phân lớp 1 lớp
│       │   └── defect_qc.py               # ✨ MODULE MỚI: Xử lý vật có lỗ / sứt mẻ độc lập
│       ├── scripts/
│       │   ├── explore.py                 # Bước 1: Trực quan phân vùng
│       │   ├── evaluate.py                # Bước 2: Đánh giá One-Class
│       │   ├── benchmark.py               # Bước 3: Benchmark 5 mô hình
│       │   ├── report_figs.py             # Bước 4: Tạo ảnh báo cáo FN/FP
│       │   └── make_table.py              # Tổng hợp bảng so sánh Markdown
│       ├── outputs/                       # Thư mục lưu kết quả (CSV, JSON, hình ảnh)
│       └── web_bridge.py                  # Cầu nối CLI JSON cho Web Backend
├── web/                                   # Giao diện ứng dụng Web (Node.js + Express + EJS)
│   ├── config/db.js                       # Kết nối PostgreSQL
│   ├── controllers/                       # Xử lý logic từng trang
│   ├── routes/                            # Định tuyến RESTful API & Views
│   │   ├── pipeline.js                    # Quản lý 4 bước pipeline chuẩn
│   │   ├── defectqc.js                    # ✨ Route cho Module Defect QC mới
│   │   └── inspection.js                  # Phân loại ảnh mới hàng loạt
│   ├── views/                             # Giao diện EJS (Bootstrap 5)
│   ├── server.js                          # Entry point của Web Server
│   └── .env                               # Cấu hình biến môi trường
├── schema.sql                             # DDL khởi tạo Database PostgreSQL
└── README.md                              # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Yêu Cầu
- **Hệ điều hành**: Windows 10/11, Linux, hoặc macOS.
- **Python**: Phiên bản 3.10 trở lên.
- **Node.js**: Phiên bản 18 trở lên & npm.
- **PostgreSQL**: Phiên bản 14 trở lên (cho ứng dụng Web).

### 2. Cài Đặt Môi Trường Python
Dự án đã có sẵn môi trường `image_processing_env` trong thư mục gốc. Nếu muốn thiết lập mới hoặc cập nhật thư viện:

```bash
# Kích hoạt môi trường (trên Windows PowerShell):
.\image_processing_env\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết:
pip install opencv-python numpy scikit-learn matplotlib
```

### 3. Cài Đặt Giao Diện Web & Cơ Sở Dữ Liệu
1. Mở PostgreSQL (pgAdmin hoặc psql) và tạo cơ sở dữ liệu:
   ```sql
   CREATE DATABASE defect_detection_db;
   ```
2. Nạp cấu trúc bảng từ file `schema.sql`:
   ```bash
   psql -U postgres -d defect_detection_db -f schema.sql
   ```
3. Cài đặt các gói npm cho thư mục `web`:
   ```bash
   cd web
   npm install
   ```
4. Kiểm tra file `web/.env` đảm bảo thông tin khớp với PostgreSQL và Python của bạn:
   ```env
   PORT=3000
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=defect_detection_db
   DB_USER=postgres
   DB_PASSWORD=mật_khẩu_của_bạn
   PYTHON_PATH=d:\hocki7\XuLyAnh\image_processing_env\Scripts\python.exe
   BRIDGE_SCRIPT=d:\hocki7\XuLyAnh\Detai_img_processing\solidity_qc\web_bridge.py
   ```

---

## 🚀 Hướng Dẫn Chạy Bằng Python CLI

### 1. Pipeline Chuẩn 4 Bước Cho 3 Bộ Data Gốc
Di chuyển vào thư mục `Detai_img_processing/solidity_qc`:
```bash
cd Detai_img_processing\solidity_qc
```

#### Bước 1: Trực quan hóa phân vùng (Explore)
Kiểm tra ngưỡng Otsu, contour, convex hull và cạnh Canny:
```bash
python scripts/explore.py biscuits 4
python scripts/explore.py hazelnut 4
python scripts/explore.py pills 4
```
*(Kết quả lưu tại `outputs/figures/explore_<dataset>.png`)*

#### Bước 2: Đánh giá phân lớp một lớp (Evaluate)
Tính Robust Z-score từ tập ảnh OK và đo AUC từng đặc trưng:
```bash
python scripts/evaluate.py biscuits --cal 0.6 --reps 30 --k 3.0
python scripts/evaluate.py hazelnut --cal 0.6 --reps 30 --k 3.0
python scripts/evaluate.py pills    --cal 0.6 --reps 30 --k 3.0
```
*(Kết quả lưu tại `outputs/summary_<dataset>.txt`, `outputs/results_<dataset>.csv` và `outputs/annotated/`)*

#### Bước 3: Kiểm chuẩn đa mô hình (Benchmark)
So sánh 5 phương pháp qua kiểm tra chéo (K-Fold cross-validation):
```bash
python scripts/benchmark.py biscuits --splits 5 --reps 10 --k 3
python scripts/benchmark.py hazelnut --splits 5 --reps 10 --k 3
python scripts/benchmark.py pills    --splits 5 --reps 10 --k 3
```
*(Kết quả lưu tại `outputs/benchmark_<dataset>.json` và `outputs/annotated_v2/`)*

#### Bước 4: Báo cáo & Bảng tổng hợp (Report)
Tạo lưới ảnh phân tích lỗi FN/FP và xuất bảng kết quả Markdown:
```bash
python scripts/report_figs.py biscuits
python scripts/report_figs.py hazelnut
python scripts/report_figs.py pills
python scripts/make_table.py
```
*(Bảng tổng hợp lưu tại `outputs/summary_all.md`)*

---

### 2. Module Ngoại Lai Mới (`Defect QC`)
Dành cho vật thể bất kỳ, vật thể có lỗ (đai ốc, vòng đệm) hoặc sứt mẻ/vỡ:

#### Cách 1: Quét và đánh giá toàn bộ Dataset (có thư mục `ok/` và `defect/`)
```bash
python web_bridge.py defect_scan --data_dir ../../data/biscuits --threshold_k 3.0 --channel gray --object_bright true
```
*Lệnh sẽ tự động:*
- Tính 5 đặc trưng: `solidity`, `convexity`, `extent`, `hole_ratio`, `boundary_roughness`.
- Chạy đánh giá one-class Monte Carlo tính AUC.
- Vẽ ảnh annotated với contour ngoài (xanh/đỏ), viền lỗ trong (cyan), và vùng khuyết tật (highlight hồng).
- Xuất kết quả chi tiết dạng JSON.

#### Cách 2: Phân loại hàng loạt ảnh mới (chưa có nhãn) từ một tập ảnh OK tham chiếu
```bash
python web_bridge.py defect_inspect \
  --ok_dir "../../data/biscuits/ok" \
  --images_dir "đường_dẫn_thư_mục_ảnh_mới" \
  --out_dir "đường_dẫn_lưu_kết_quả" \
  --threshold_k 3.0
```

---

## 🌐 Hướng Dẫn Chạy Giao Diện Web

### 1. Khởi Động Server
Di chuyển vào thư mục `web` và khởi động:
```bash
cd web
npm start
# Hoặc chế độ tự động reload khi sửa file:
npm run dev
```

Khi server khởi động thành công, màn hình sẽ thông báo:
```text
====================================================
🚀 Solidity QC Vision chạy tại: http://localhost:3000
====================================================
[DB] Kết nối PostgreSQL thành công!
```

### 2. Hướng Dẫn Sử Dụng Các Trang

Truy cập trình duyệt tại: **`http://localhost:3000`**

- **Trang Chủ (Dashboard - `/`)**: Tổng quan số lượng bộ dữ liệu, số phiên kiểm tra, tỷ lệ lỗi và danh sách hoạt động gần đây.
- **Bộ Dữ Liệu (Datasets - `/datasets`)**: Quản lý các tập dữ liệu, cấu hình kênh màu (`gray`/`sat`), cách chọn vùng lõi (`filled`/`mask`) và ngưỡng phát hiện $k$.
- **Pipeline Phân Tích (`/pipeline`)**: Thực hiện tuần tự 4 bước (Explore $\rightarrow$ Evaluate $\rightarrow$ Benchmark $\rightarrow$ Report) với giao diện trực quan, cho phép tải về CSV và xem toàn bộ ảnh khoanh vùng khuyết tật.
- **Phân Loại Ảnh Hàng Loạt (`/inspection`)**: Tải lên nhiều ảnh sản phẩm mới cần kiểm tra, hệ thống tự động gán nhãn OK/DEFECT theo mô hình đã huấn luyện của bộ dữ liệu chuẩn.
- **Module Mới: Defect QC (`/defectqc`)**:
  - **Scan Dataset**: Tải lên đồng thời ảnh OK và Defect của bất kỳ sản phẩm nào (kể cả có lỗ tâm như vòng đệm, bu-lông), hệ thống tự động phân tích và đưa ra chỉ số AUC, Precision, Recall và Confusion Matrix.
  - **Phân Loại Ảnh Mới (`/defectqc/inspect`)**: Chỉ cần tải lên tối thiểu 3 ảnh OK chuẩn và danh sách ảnh mới cần kiểm tra để nhận kết quả phân loại tức thì.
  - **Xuất Báo Cáo**: Nút xuất file CSV kèm đầy đủ 5 thông số khuyết tật hình học.

---

## 📊 Tóm Tắt Các Chỉ Số Đo Lường

| Chỉ số | Công thức / Ý nghĩa | Ngưỡng bình thường (OK) | Khi phát hiện Lỗi (DEFECT) |
|---|---|---|---|
| **Solidity** | $\frac{\text{Area}}{\text{HullArea}}$ (Độ lồi đặc) | Tiệm cận $1.0$ ($> 0.94$) | Giảm mạnh do sứt mẻ, vỡ lõm viền |
| **Circularity** | $\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$ (Độ tròn) | Cao & ổn định | Giảm do viền biến dạng, gồ ghề |
| **Convexity** | $\frac{\text{Perimeter}_{\text{hull}}}{\text{Perimeter}_{\text{contour}}}$ | Xấp xỉ $1.0$ | Giảm khi biên có nhiều vết lõm/nứt |
| **Hole Ratio** | $\frac{\sum \text{Area}_{\text{holes}}}{\text{Area}_{\text{outer}}}$ | Cố định theo thiết kế chuẩn | Tăng khi vỡ thủng hoặc lỗ bị méo |
| **Boundary Roughness** | $\frac{\sigma(r)}{\mu(r)}$ (Độ gồ ghề bán kính) | Thấp ($< 0.15$) | Tăng mạnh khi viền vỡ nham nhở |
| **Robust Z-score** | $Z = \frac{X - \text{Median}}{1.4826 \cdot \text{MAD}}$ | $Z \le 3.0$ | $Z > 3.0$ kết luận bất thường |

---

> [!NOTE]
> Mọi thay đổi và bổ sung đều tuân thủ nguyên lý Đóng-Mở (OCP): **100% mã nguồn ban đầu của đề tài được bảo toàn nguyên vẹn**, các tính năng mở rộng cho vật thể ngoại lai hoạt động hoàn toàn độc lập và linh hoạt.
