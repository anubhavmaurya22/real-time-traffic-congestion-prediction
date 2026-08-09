// ─────────────────────────────────────────────────────────────────────────────
// Central API Base Configuration for STREETFLOW LIVE
// ─────────────────────────────────────────────────────────────────────────────

export const getApiBase = () => {
  const isProd = window.location.hostname.includes('web.app') ||
                 window.location.hostname.includes('firebaseapp.com');
  if (isProd) {
    return 'https://43-204-232-243.sslip.io';
  }
  return 'http://127.0.0.1:8000';
};

export const API_BASE = getApiBase();
window.__STREETFLOW_API_BASE = API_BASE;
