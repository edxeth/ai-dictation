/**
 * Persisted backend keys and labels, in switching order. The keys and order match
 * BACKEND_LABELS in src/local_ai_dictation/backend_state.py.
 */
export const BACKEND_LABELS = {
  whisper: "Whisper",
  parakeet: "Parakeet Ultra",
  "parakeet-redux": "Parakeet Redux",
  bark: "Bark 3.0",
  "bark-pro": "Bark 3.0 Pro",
  willow: "Willow Voice",
} as const;

export type BackendName = keyof typeof BACKEND_LABELS;

const BACKEND_ORDER = Object.keys(BACKEND_LABELS) as BackendName[];

export function isBackendName(value: unknown): value is BackendName {
  return typeof value === "string" && Object.hasOwn(BACKEND_LABELS, value);
}

/** Return the backend, or Whisper (the CLI default) for anything unknown. */
export function normalizeBackendName(value: unknown): BackendName {
  return isBackendName(value) ? value : "whisper";
}

export function nextBackend(backend: BackendName): BackendName {
  return BACKEND_ORDER[(BACKEND_ORDER.indexOf(backend) + 1) % BACKEND_ORDER.length];
}
