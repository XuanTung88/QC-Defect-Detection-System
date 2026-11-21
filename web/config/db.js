const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432', 10),
  database: process.env.DB_NAME || 'defect_detection_db',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

pool.on('error', (err) => {
  console.error('[DB Pool Error]: Unexpected error on idle client', err);
});

async function query(text, params) {
  const start = Date.now();
  try {
    const res = await pool.query(text, params);
    const duration = Date.now() - start;
    if (process.env.NODE_ENV === 'development') {
      console.log('[DB Query]', { text, duration, rows: res.rowCount });
    }
    return res;
  } catch (err) {
    console.error('[DB Query Error]:', { text, error: err.message });
    throw err;
  }
}

async function testConnection() {
  try {
    const res = await pool.query('SELECT NOW() as current_time, current_database() as db_name');
    console.log(`[Database Connected]: DB="${res.rows[0].db_name}", Time=${res.rows[0].current_time}`);
    return true;
  } catch (err) {
    console.warn(`[Database Notice]: Chua the ket noi toi PostgreSQL: ${err.message}`);
    console.warn(`[Huong dan]: Hay khoi dong PostgreSQL va tao database 'defect_detection_db' bang schema.sql`);
    return false;
  }
}

module.exports = {
  pool,
  query,
  testConnection
};
