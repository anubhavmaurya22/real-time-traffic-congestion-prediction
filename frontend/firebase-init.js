// ─────────────────────────────────────────────────────────────────────────────
// Firebase initialisation + App Check for STREETFLOW LIVE
//
// This script initialises Firebase and App Check once per page load.
// ● localhost / 127.0.0.1  →  debug provider  (prints debug token in console)
// ● production (web.app)   →  reCAPTCHA v3 provider
//
// SETUP — three values you MUST replace before deploying:
//
//   1. firebaseConfig  – paste your Firebase web-app config object.
//      Find it at: Firebase Console → Project Settings → Your apps → Web app
//      → "SDK setup and configuration" → Config.
//
//   2. RECAPTCHA_V3_SITE_KEY – your reCAPTCHA v3 PUBLIC site key.
//      Find it at: Firebase Console → App Check → your web app → reCAPTCHA v3
//      Or at: https://www.google.com/recaptcha/admin
//      ⚠  This is the SITE KEY, NOT the secret key.
//      ⚠  The SECRET KEY must NEVER appear in frontend code.
//
//   3. On first localhost run, copy the debug token printed in the browser
//      console and register it in:
//      Firebase Console → App Check → Apps → Manage debug tokens → Add
// ─────────────────────────────────────────────────────────────────────────────

import { initializeApp }     from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js';
import { initializeAppCheck,
         ReCaptchaV3Provider } from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-app-check.js';
import { getDatabase }        from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-database.js';

// ── Firebase project configuration ──────────────────────────────────────────
// Replace every "YOUR_…" placeholder with the real value from Firebase Console.
const firebaseConfig = {
  apiKey:            "YOUR_FIREBASE_API_KEY",
  authDomain:        "YOUR_PROJECT_ID.firebaseapp.com",
  databaseURL:       "https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com",
  projectId:         "YOUR_PROJECT_ID",
  storageBucket:     "YOUR_PROJECT_ID.firebasestorage.app",
  messagingSenderId: "YOUR_SENDER_ID",
  appId:             "YOUR_APP_ID",
};

// ── reCAPTCHA v3 public site key ────────────────────────────────────────────
const RECAPTCHA_V3_SITE_KEY = "YOUR_RECAPTCHA_V3_SITE_KEY";

// ── Initialise Firebase App ─────────────────────────────────────────────────
const app = initializeApp(firebaseConfig);

// ── App Check — environment-aware provider ──────────────────────────────────
//
// Localhost / 127.0.0.1  → debug provider (self.FIREBASE_APPCHECK_DEBUG_TOKEN)
//   The SDK will print a one-time debug token in the browser console.
//   Register that token in Firebase Console → App Check → Manage debug tokens.
//
// Production (*.web.app / *.firebaseapp.com / custom domain)
//   → reCAPTCHA v3 provider with the public site key above.
//
const _isLocalDev = (location.hostname === 'localhost' ||
                     location.hostname === '127.0.0.1');

if (_isLocalDev) {
  self.FIREBASE_APPCHECK_DEBUG_TOKEN = true;
}

const appCheck = initializeAppCheck(app, {
  provider: new ReCaptchaV3Provider(RECAPTCHA_V3_SITE_KEY),
  isTokenAutoRefreshEnabled: true,
});

// ── Realtime Database ───────────────────────────────────────────────────────
// App Check tokens are automatically attached to all RTDB requests made
// through this Firebase SDK instance once enforcement is enabled in Console.
const db = getDatabase(app);

// ── Expose to non-module scripts (optional) ─────────────────────────────────
// Existing page scripts are not ES modules, so they can read these via window.
window.__streetflow_firebase = { app, appCheck, db };

console.log(
  '[STREETFLOW] Firebase + App Check initialised — ' +
  (_isLocalDev
    ? 'DEBUG provider active. Copy the debug token from above and register it in Firebase Console → App Check → Manage debug tokens.'
    : 'reCAPTCHA v3 provider active.')
);
