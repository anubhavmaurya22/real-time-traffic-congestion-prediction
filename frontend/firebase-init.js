// ─────────────────────────────────────────────────────────────────────────────
// STREETFLOW LIVE — Firebase Auth + App Check Init (Hosting Only)
// ─────────────────────────────────────────────────────────────────────────────

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js';
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
} from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-auth.js';
import {
  initializeAppCheck,
  ReCaptchaV3Provider
} from 'https://www.gstatic.com/firebasejs/10.14.0/firebase-app-check.js';

// ── Firebase project configuration (traff2ic-detector) ──────────────────────
const firebaseConfig = {
  apiKey:            "AIzaSyDJn67yUdialq8dwh6-fVidva3e1UCj1HU",
  authDomain:        "traff2ic-detector.firebaseapp.com",
  projectId:         "traff2ic-detector",
  storageBucket:     "traff2ic-detector.firebasestorage.app",
  messagingSenderId: "1055184397335",
  appId:             "1:1055184397335:web:e42d7c3824ee4249022852",
  measurementId:     "G-3JMDM2NXM8"
};

// ── reCAPTCHA v3 public site key ────────────────────────────────────────────
const RECAPTCHA_V3_SITE_KEY = "YOUR_RECAPTCHA_V3_SITE_KEY";

// ── Initialise Firebase Services ────────────────────────────────────────────
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// ── App Check — environment-aware provider ──────────────────────────────────
let appCheck = null;
const _isLocalDev = (location.hostname === 'localhost' || location.hostname === '127.0.0.1');

try {
  if (_isLocalDev) {
    self.FIREBASE_APPCHECK_DEBUG_TOKEN = true;
    console.log('[STREETFLOW APP CHECK] Localhost detected — Debug Provider flag set.');
  }

  if (RECAPTCHA_V3_SITE_KEY && RECAPTCHA_V3_SITE_KEY !== "YOUR_RECAPTCHA_V3_SITE_KEY") {
    appCheck = initializeAppCheck(app, {
      provider: new ReCaptchaV3Provider(RECAPTCHA_V3_SITE_KEY),
      isTokenAutoRefreshEnabled: true,
    });
    console.log('[STREETFLOW APP CHECK] reCAPTCHA v3 Provider active.');
  }
} catch (err) {
  console.warn('[STREETFLOW APP CHECK] App Check notice:', err.message || err);
}

// ── Attach to window for global access ──────────────────────────────────────
window.__streetflow_firebase = {
  app,
  auth,
  appCheck,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
};

// ── Export ES Module interface ──────────────────────────────────────────────
export {
  app,
  auth,
  appCheck,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
};
