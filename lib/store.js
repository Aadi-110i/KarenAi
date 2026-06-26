'use client';

// Karen AI — Client-Side Storage
// SSR-safe localStorage wrapper with KAREN_AI_ key prefix

const KEYS = {
  PROFILE: 'KAREN_AI_PROFILE',
  JOURNAL: 'KAREN_AI_JOURNAL_ENTRIES',
  CHAT: 'KAREN_AI_CHAT_HISTORY',
  THEME: 'KAREN_AI_THEME',
  ONBOARDING: 'KAREN_AI_ONBOARDING_COMPLETE',
};

// ── Helpers ───────────────────────────────────────────────────────

/**
 * Checks whether localStorage is available (guards against SSR).
 * @returns {boolean}
 */
function isStorageAvailable() {
  try {
    return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
  } catch {
    return false;
  }
}

/**
 * Safely reads and parses a JSON value from localStorage.
 * @param {string} key
 * @param {*} fallback - Value returned when key is missing or unparseable
 * @returns {*}
 */
function readJSON(key, fallback) {
  if (!isStorageAvailable()) return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

/**
 * Safely serializes and writes a value to localStorage.
 * @param {string} key
 * @param {*} value
 */
function writeJSON(key, value) {
  if (!isStorageAvailable()) return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage may be full or blocked — fail silently
  }
}

// ── Profile ───────────────────────────────────────────────────────

/**
 * Retrieves the user profile from storage.
 * @returns {{ name: string, age: number, gender: string } | null}
 */
export function getProfile() {
  return readJSON(KEYS.PROFILE, null);
}

/**
 * Saves the user profile to storage.
 * @param {{ name: string, age: number, gender: string }} profile
 */
export function setProfile(profile) {
  writeJSON(KEYS.PROFILE, profile);
}

// ── Journal Entries ───────────────────────────────────────────────

/**
 * Retrieves all journal entries from storage.
 * @returns {Array<{ id: string, title: string, content: string, date: string, mood?: string }>}
 */
export function getJournalEntries() {
  return readJSON(KEYS.JOURNAL, []);
}

/**
 * Adds a new journal entry to storage.
 * @param {{ title: string, content: string, mood?: string }} entry
 * @returns {{ id: string, title: string, content: string, date: string, mood?: string }}
 */
export function addJournalEntry(entry) {
  const entries = getJournalEntries();
  const newEntry = {
    id: generateId(),
    date: new Date().toISOString(),
    ...entry,
  };
  entries.unshift(newEntry); // newest first
  writeJSON(KEYS.JOURNAL, entries);
  return newEntry;
}

/**
 * Deletes a journal entry by ID.
 * @param {string} id
 * @returns {boolean} Whether an entry was actually removed
 */
export function deleteJournalEntry(id) {
  const entries = getJournalEntries();
  const filtered = entries.filter((e) => e.id !== id);
  if (filtered.length === entries.length) return false;
  writeJSON(KEYS.JOURNAL, filtered);
  return true;
}

// ── Chat History ──────────────────────────────────────────────────

/**
 * Retrieves the full chat history from storage.
 * @returns {Array<{ role: string, content: string, timestamp: string }>}
 */
export function getChatHistory() {
  return readJSON(KEYS.CHAT, []);
}

/**
 * Appends a chat message to storage.
 * @param {{ role: string, content: string }} msg
 * @returns {{ role: string, content: string, timestamp: string }}
 */
export function addChatMessage(msg) {
  const history = getChatHistory();
  const newMsg = {
    ...msg,
    timestamp: new Date().toISOString(),
  };
  history.push(newMsg);
  writeJSON(KEYS.CHAT, history);
  return newMsg;
}

/**
 * Clears all chat history from storage.
 */
export function clearChatHistory() {
  if (!isStorageAvailable()) return;
  try {
    localStorage.removeItem(KEYS.CHAT);
  } catch {
    // fail silently
  }
}

// ── Theme Preference ──────────────────────────────────────────────

/**
 * Retrieves the saved theme preference.
 * @returns {string | null}
 */
export function getThemePreference() {
  if (!isStorageAvailable()) return null;
  try {
    return localStorage.getItem(KEYS.THEME);
  } catch {
    return null;
  }
}

/**
 * Saves the theme preference.
 * @param {string} theme - e.g. 'light', 'dark', 'system'
 */
export function setThemePreference(theme) {
  if (!isStorageAvailable()) return;
  try {
    localStorage.setItem(KEYS.THEME, theme);
  } catch {
    // fail silently
  }
}

// ── Onboarding ────────────────────────────────────────────────────

/**
 * Checks whether the user has completed onboarding.
 * @returns {boolean}
 */
export function hasCompletedOnboarding() {
  if (!isStorageAvailable()) return false;
  try {
    return localStorage.getItem(KEYS.ONBOARDING) === 'true';
  } catch {
    return false;
  }
}

/**
 * Marks onboarding as complete.
 */
export function setOnboardingComplete() {
  if (!isStorageAvailable()) return;
  try {
    localStorage.setItem(KEYS.ONBOARDING, 'true');
  } catch {
    // fail silently
  }
}

// ── Utilities ─────────────────────────────────────────────────────

/**
 * Generates a simple unique ID for entries.
 * @returns {string}
 */
function generateId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}
