const { query } = require('../config/db');

// ==================== DATASETS ====================
async function getAllDatasets() {
  const sql = `
    SELECT d.*,
      COUNT(DISTINCT ti.id) FILTER (WHERE ti.label = 'ok') as ok_count,
      COUNT(DISTINCT ti.id) FILTER (WHERE ti.label = 'defect') as defect_count
    FROM datasets d
    LEFT JOIN training_images ti ON d.id = ti.dataset_id
    GROUP BY d.id
    ORDER BY d.created_at DESC;
  `;
  const res = await query(sql);
  return res.rows;
}

async function getDatasetById(id) {
  const res = await query('SELECT * FROM datasets WHERE id = $1', [id]);
  return res.rows[0];
}

async function getDatasetByName(name) {
  const res = await query('SELECT * FROM datasets WHERE name = $1', [name]);
  return res.rows[0];
}

async function createDataset({ name, description, channel = 'gray', object_bright = true, crop = 0, interior = 'filled', threshold_k = 3.0 }) {
  const sql = `
    INSERT INTO datasets (name, description, channel, object_bright, crop, interior, threshold_k, status)
    VALUES ($1,$2,$3,$4,$5,$6,$7,'pending') RETURNING *;
  `;
  const res = await query(sql, [name, description, channel, object_bright, crop, interior, threshold_k]);
  return res.rows[0];
}

async function updateDatasetStatus(id, updates) {
  const { status, pipeline_step, auc_score, explore_fig, eval_fig, bench_fig, errors_fig, bench_data, eval_data } = updates;
  const sql = `
    UPDATE datasets SET
      status        = CASE WHEN $1::text IS NOT NULL THEN $1 ELSE status END,
      pipeline_step = CASE WHEN $2::text IS NOT NULL THEN $2 ELSE pipeline_step END,
      auc_score     = CASE WHEN $3::float8 IS NOT NULL THEN $3 ELSE auc_score END,
      explore_fig   = CASE WHEN $4::text IS NOT NULL THEN $4 ELSE explore_fig END,
      eval_fig      = CASE WHEN $5::text IS NOT NULL THEN $5 ELSE eval_fig END,
      bench_fig     = CASE WHEN $6::text IS NOT NULL THEN $6 ELSE bench_fig END,
      errors_fig    = CASE WHEN $7::text IS NOT NULL THEN $7 ELSE errors_fig END,
      bench_data    = CASE WHEN $8::jsonb IS NOT NULL THEN $8 ELSE bench_data END,
      eval_data     = CASE WHEN $9::jsonb IS NOT NULL THEN $9 ELSE eval_data END,
      updated_at    = NOW()
    WHERE id = $10 RETURNING *;
  `;
  const res = await query(sql, [
    status ?? null,
    pipeline_step ?? null,
    auc_score ?? null,
    explore_fig ?? null,
    eval_fig ?? null,
    bench_fig ?? null,
    errors_fig ?? null,
    bench_data ? JSON.stringify(bench_data) : null,
    eval_data  ? JSON.stringify(eval_data)  : null,
    id
  ]);
  return res.rows[0];
}

async function deleteDataset(id) {
  const res = await query('DELETE FROM datasets WHERE id = $1 RETURNING *', [id]);
  return res.rows[0];
}

// ==================== TRAINING IMAGES ====================
async function insertTrainingImage({ dataset_id, filename, filepath, label, file_size }) {
  const sql = `INSERT INTO training_images (dataset_id,filename,filepath,label,file_size) VALUES ($1,$2,$3,$4,$5) RETURNING *`;
  const res = await query(sql, [dataset_id, filename, filepath, label, file_size]);
  return res.rows[0];
}

async function countTrainingImages(dataset_id) {
  const sql = `SELECT
    COUNT(*) FILTER (WHERE label='ok') as ok_count,
    COUNT(*) FILTER (WHERE label='defect') as defect_count,
    COUNT(*) as total_count
    FROM training_images WHERE dataset_id=$1`;
  const res = await query(sql, [dataset_id]);
  return {
    ok_count: parseInt(res.rows[0].ok_count || 0),
    defect_count: parseInt(res.rows[0].defect_count || 0),
    total_count: parseInt(res.rows[0].total_count || 0)
  };
}

// ==================== INSPECTIONS ====================
async function createInspection({ dataset_id, name, total_images = 0 }) {
  const sql = `INSERT INTO inspections (dataset_id,name,total_images,status) VALUES ($1,$2,$3,'processing') RETURNING *`;
  const res = await query(sql, [dataset_id, name, total_images]);
  return res.rows[0];
}

async function updateInspectionStatus(id, { total_images, ok_count, defect_count, status }) {
  const sql = `UPDATE inspections SET
    total_images=COALESCE($1,total_images),
    ok_count=COALESCE($2,ok_count),
    defect_count=COALESCE($3,defect_count),
    status=COALESCE($4,status)
    WHERE id=$5 RETURNING *`;
  const res = await query(sql, [total_images, ok_count, defect_count, status, id]);
  return res.rows[0];
}

async function getInspectionById(id) {
  const sql = `SELECT i.*, d.name as dataset_name, d.threshold_k
    FROM inspections i JOIN datasets d ON i.dataset_id=d.id WHERE i.id=$1`;
  const res = await query(sql, [id]);
  return res.rows[0];
}

async function getAllInspections(limit = 20) {
  const sql = `SELECT i.*, d.name as dataset_name
    FROM inspections i JOIN datasets d ON i.dataset_id=d.id
    ORDER BY i.created_at DESC LIMIT $1`;
  const res = await query(sql, [limit]);
  return res.rows;
}

// ==================== INSPECTION RESULTS ====================
async function insertInspectionResult({ inspection_id, filename, original_path, annotated_path, prediction,
  anomaly_score, solidity, circularity, edge_density, sobel_mean, gray_std, hist_dist,
  patch_max, patch_top5, patch_p90, patch_mean }) {
  const sql = `INSERT INTO inspection_results
    (inspection_id,filename,original_path,annotated_path,prediction,
     anomaly_score,solidity,circularity,edge_density,sobel_mean,gray_std,hist_dist,
     patch_max,patch_top5,patch_p90,patch_mean)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) RETURNING *`;
  const res = await query(sql, [
    inspection_id, filename, original_path, annotated_path, prediction,
    anomaly_score, solidity, circularity, edge_density, sobel_mean, gray_std, hist_dist,
    patch_max, patch_top5, patch_p90, patch_mean
  ]);
  return res.rows[0];
}

async function getInspectionResults(inspection_id) {
  const sql = `SELECT * FROM inspection_results WHERE inspection_id=$1 ORDER BY anomaly_score DESC NULLS LAST`;
  const res = await query(sql, [inspection_id]);
  return res.rows;
}

async function getInspectionResultById(id) {
  const res = await query('SELECT * FROM inspection_results WHERE id=$1', [id]);
  return res.rows[0];
}

// ==================== DASHBOARD ====================
async function getDashboardStats() {
  const [d, i, r, recent] = await Promise.all([
    query('SELECT COUNT(*) as count FROM datasets'),
    query('SELECT COUNT(*) as count FROM inspections'),
    query(`SELECT COUNT(*) as total,
      COUNT(*) FILTER (WHERE prediction='defect') as defects
      FROM inspection_results`),
    query(`SELECT i.*, d.name as dataset_name FROM inspections i
      JOIN datasets d ON i.dataset_id=d.id ORDER BY i.created_at DESC LIMIT 5`)
  ]);
  return {
    total_datasets: parseInt(d.rows[0]?.count || 0),
    total_inspections: parseInt(i.rows[0]?.count || 0),
    total_inspected_images: parseInt(r.rows[0]?.total || 0),
    total_defects_found: parseInt(r.rows[0]?.defects || 0),
    recent_inspections: recent.rows
  };
}

module.exports = {
  getAllDatasets, getDatasetById, getDatasetByName, createDataset,
  updateDatasetStatus, deleteDataset,
  insertTrainingImage, countTrainingImages,
  createInspection, updateInspectionStatus, getInspectionById, getAllInspections,
  insertInspectionResult, getInspectionResults, getInspectionResultById,
  getDashboardStats
};
