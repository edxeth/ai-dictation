import Electrobun, { Electroview } from "electrobun/view";

type TranscriptPayload = {
  schema_version: number;
  transcript: string;
  normalized_transcript?: string;
  device?: string;
  metadata?: Record<string, unknown>;
};

type TranscriptHistoryItem = {
  id: string;
  completed_at: number;
  payload: TranscriptPayload;
};

type SessionPayload = {
  schema_version: number;
  state: "stopped" | "idle" | "starting" | "recording" | "transcribing" | "error";
  started_at: number | null;
  recording_started_at: number | null;
  last_completed_at: number | null;
  clipboard_copied_at: number | null;
  last_transcript: TranscriptPayload | null;
  last_error: string | null;
  model_loaded: boolean;
  model_loading: boolean;
  history: TranscriptHistoryItem[];
  config: Record<string, unknown>;
  stderr_tail: string[];
};

type BackendName = "whisper" | "parakeet" | "willow";

type BridgeViewState = {
  bridgeUrl: string;
  bridgeStartCommand: string;
  preferredBackend: BackendName;
  hotkey: string;
  hotkeyRegistered: boolean;
  connected: boolean;
  session: SessionPayload;
};

type BridgeHealthPayload = {
  schema_version: number;
  ok: boolean;
  bridge: {
    backend: string;
    model_backend: BackendName;
    model_loaded: boolean;
    model_loading: boolean;
    e2e_mode: boolean;
  };
  session: SessionPayload;
};

type SessionStatus = SessionPayload["state"] | "offline";

type RendererAutomationSnapshot = {
  statusBadgeText: string;
  statusBadgeClassName: string;
  statusLine: string;
  toggleButtonText: string;
  toggleButtonDisabled: boolean;
  clearHistoryButtonDisabled: boolean;
  busyOverlayVisible: boolean;
  bridgeUrl: string;
  bridgeCommand: string;
  errorText: string;
  historyCount: number;
  historyTexts: string[];
};

type RendererAutomationActionId = "toggle-recording";
type SessionCueKind = "start" | "complete";

type DesktopRPC = {
  bun: {
    requests: {
      getBridgeState: { params: {}; response: BridgeViewState };
      startRecording: { params: {}; response: BridgeViewState };
      stopRecording: { params: {}; response: BridgeViewState };
      toggleRecording: { params: {}; response: BridgeViewState };
      toggleBackend: { params: {}; response: BridgeViewState };
      clearHistory: { params: {}; response: BridgeViewState };
      showWindow: { params: {}; response: { success: true } };
      minimizeWindow: { params: {}; response: { success: true } };
      closeWindow: { params: {}; response: { success: true } };
      reportRendererReady: { params: { userAgent: string }; response: { success: true } };
    };
    messages: {};
  };
  webview: {
    requests: {
      getAutomationSnapshot: { params: {}; response: RendererAutomationSnapshot };
      runAutomationAction: { params: { action: RendererAutomationActionId }; response: RendererAutomationSnapshot };
      playSessionCue: { params: { kind: SessionCueKind }; response: { success: true } };
    };
    messages: {};
  };
};

const rpc = Electroview.defineRPC<DesktopRPC>({
  maxRequestTime: 30000,
  handlers: {
    requests: {
      getAutomationSnapshot: async () => buildRendererAutomationSnapshot(currentState),
      runAutomationAction: async ({ action }) => {
        switch (action) {
          case "toggle-recording":
            await toggleRecording();
            return buildRendererAutomationSnapshot(currentState);
        }
      },
      playSessionCue: async ({ kind }) => {
        await primeSessionAudio();
        playSessionCue(kind);
        return { success: true } as const;
      },
    },
    messages: {},
  },
});

const electrobun = new Electrobun.Electroview({ rpc });

const busyOverlay = document.getElementById("busyOverlay") as HTMLDivElement;
const busyTitle = document.getElementById("busyTitle") as HTMLDivElement;
const busyMessage = document.getElementById("busyMessage") as HTMLDivElement;
const confirmOverlay = document.getElementById("confirmOverlay") as HTMLDivElement;
const confirmTitle = document.getElementById("confirmTitle") as HTMLDivElement;
const confirmMessage = document.getElementById("confirmMessage") as HTMLDivElement;
const confirmCancelButton = document.getElementById("confirmCancelButton") as HTMLButtonElement;
const confirmOkButton = document.getElementById("confirmOkButton") as HTMLButtonElement;
const appShell = document.getElementById("appShell") as HTMLDivElement;
const minimizeWindowButton = document.getElementById("minimizeWindowButton") as HTMLButtonElement;
const closeWindowButton = document.getElementById("closeWindowButton") as HTMLButtonElement;
const statusBadge = document.getElementById("statusBadge") as HTMLDivElement;
const hotkeyValue = document.getElementById("hotkeyValue") as HTMLDivElement;
const modelValue = document.getElementById("modelValue") as HTMLDivElement;
const switchModelButton = document.getElementById("switchModelButton") as HTMLButtonElement;
const toggleButton = document.getElementById("toggleButton") as HTMLButtonElement;
const statusLine = document.getElementById("statusLine") as HTMLParagraphElement;
const bridgeUrl = document.getElementById("bridgeUrl") as HTMLDivElement;
const bridgeCommand = document.getElementById("bridgeCommand") as HTMLPreElement;
const historyMeta = document.getElementById("historyMeta") as HTMLDivElement;
const historyList = document.getElementById("historyList") as HTMLDivElement;
const showMoreButton = document.getElementById("showMoreButton") as HTMLButtonElement;
const errorBox = document.getElementById("errorBox") as HTMLPreElement;
const clearHistoryButton = document.getElementById("clearHistoryButton") as HTMLButtonElement;

window.addEventListener("error", (event) => {
  errorBox.textContent = event.message || "unknown renderer error";
});

let currentState: BridgeViewState | null = null;
let lastRenderedStateSignature: string | null = null;
let visibleHistoryCount = 10;
let historyClearedAfter: number | null = null;
const copiedHistoryIds = new Set<string>();
const copiedHistoryTimers = new Map<string, number>();
let sessionAudioContext: AudioContext | null = null;
let bridgeEventSource: EventSource | null = null;
const APP_STARTED_AT = Date.now() / 1000;
let lastObservedSessionState: SessionStatus = "offline";
let lastPlayedStartCueAt: number | null = null;
let lastPlayedCompleteCueAt: number | null = null;
const sessionCueAudio: Record<SessionCueKind, HTMLAudioElement> = {
  start: new Audio("assets/session-start.wav"),
  complete: new Audio("assets/session-complete.wav"),
};
for (const audio of Object.values(sessionCueAudio)) {
  audio.preload = "auto";
  audio.volume = 1;
}

function formatTimestamp(timestamp: number | null): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function renderHotkey(accelerator: string): string {
  const isMac = navigator.platform.toLowerCase().includes("mac");
  const keyMap: Record<string, string> = {
    CommandOrControl: isMac ? "⌘" : "Ctrl",
    Command: "⌘",
    Control: "Ctrl",
    Alt: isMac ? "⌥" : "Alt",
    Option: "⌥",
    Shift: "⇧",
    Super: isMac ? "⌃" : "Win",
  };
  const parts = accelerator.split("+").map((part) => part.trim()).filter(Boolean);
  return `<span class="hotkey-display">${parts
    .map((part) => `<span class="hotkey-key">${escapeHtml(keyMap[part] || part)}</span>`)
    .join("")}</span>`;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderBackendLabel(backend: BackendName): string {
  if (backend === "parakeet") return "Parakeet";
  if (backend === "willow") return "Willow Voice";
  return "Whisper";
}

function nextBackend(backend: BackendName): BackendName {
  if (backend === "whisper") return "parakeet";
  if (backend === "parakeet") return "willow";
  return "whisper";
}

function renderHistoryActionIcon(copied: boolean): string {
  if (copied) {
    return `
      <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
        <path d="M16.4 5.4a1 1 0 0 1 0 1.4l-7 7a1 1 0 0 1-1.4 0L4.6 10.4A1 1 0 1 1 6 9l2.7 2.7 6.3-6.3a1 1 0 0 1 1.4 0Z" fill="currentColor"></path>
      </svg>
    `;
  }

  return `
    <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <path d="M7 3.5A2.5 2.5 0 0 1 9.5 1h5A2.5 2.5 0 0 1 17 3.5v7A2.5 2.5 0 0 1 14.5 13h-5A2.5 2.5 0 0 1 7 10.5v-7Zm2.5-.5a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h5a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5h-5Z" fill="currentColor"></path>
      <path d="M3 6.5A2.5 2.5 0 0 1 5.5 4H6v2H5.5a.5.5 0 0 0-.5.5v8a.5.5 0 0 0 .5.5h6a.5.5 0 0 0 .5-.5V14h2v.5A2.5 2.5 0 0 1 11.5 17h-6A2.5 2.5 0 0 1 3 14.5v-8Z" fill="currentColor"></path>
    </svg>
  `;
}

function applyHistoryCopyButtonState(button: HTMLButtonElement, copied: boolean) {
  button.classList.toggle("is-copied", copied);
  button.setAttribute("aria-label", copied ? "Copied" : "Copy transcript");
  button.setAttribute("title", copied ? "Copied" : "Copy transcript");
  button.innerHTML = renderHistoryActionIcon(copied);
}

function syncViewportDensity() {
  const viewportHeight = window.innerHeight;
  appShell.classList.toggle("compact-window", viewportHeight <= 880);
  appShell.classList.toggle("tight-window", viewportHeight <= 760);
}

function getSessionAudioContext(): AudioContext | null {
  const AudioContextCtor = window.AudioContext;
  if (!AudioContextCtor) return null;
  sessionAudioContext ??= new AudioContextCtor();
  return sessionAudioContext;
}

async function primeSessionAudio(): Promise<void> {
  const context = getSessionAudioContext();
  if (!context || context.state !== "suspended") return;
  try {
    await context.resume();
  } catch {
    return;
  }
}

function playSessionCue(kind: SessionCueKind) {
  void playSessionCueAsync(kind);
}

async function playSessionCueAsync(kind: SessionCueKind): Promise<void> {
  const context = getSessionAudioContext();
  if (context) {
    if (context.state === "suspended") {
      try {
        await context.resume();
      } catch {
        // fall through to audio-element fallback
      }
    }

    if (context.state === "running") {
      const pulseStart = context.currentTime + 0.003;
      const pulseDuration = 0.1;
      const pulseEnd = pulseStart + pulseDuration;
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();

      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(kind === "start" ? 1046 : 740, pulseStart);

      gainNode.gain.setValueAtTime(0.0001, pulseStart);
      gainNode.gain.exponentialRampToValueAtTime(kind === "start" ? 0.62 : 0.56, pulseStart + 0.006);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, pulseEnd);

      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      oscillator.start(pulseStart);
      oscillator.stop(pulseEnd + 0.01);
      return;
    }
  }

  const fallback = sessionCueAudio[kind].cloneNode(true) as HTMLAudioElement;
  fallback.volume = 1;
  fallback.currentTime = 0;
  try {
    await fallback.play();
  } catch {
    return;
  }
}

function setBusy(busy: boolean) {
  toggleButton.disabled = busy;
  clearHistoryButton.disabled = busy;
}

function setOverlay(visible: boolean, title = "Working…", message = "Please wait.") {
  busyOverlay.classList.toggle("hidden", !visible);
  appShell.classList.toggle("locked", visible);
  busyOverlay.setAttribute("aria-hidden", visible ? "false" : "true");
  busyTitle.textContent = title;
  busyMessage.textContent = message;
}

function confirmAction(title: string, message: string): Promise<boolean> {
  confirmTitle.textContent = title;
  confirmMessage.textContent = message;
  confirmOverlay.classList.remove("hidden");
  confirmOverlay.setAttribute("aria-hidden", "false");

  return new Promise((resolve) => {
    const cleanup = () => {
      confirmOverlay.classList.add("hidden");
      confirmOverlay.setAttribute("aria-hidden", "true");
      confirmCancelButton.removeEventListener("click", onCancel);
      confirmOkButton.removeEventListener("click", onOk);
    };
    const onCancel = () => {
      cleanup();
      resolve(false);
    };
    const onOk = () => {
      cleanup();
      resolve(true);
    };
    confirmCancelButton.addEventListener("click", onCancel, { once: true });
    confirmOkButton.addEventListener("click", onOk, { once: true });
  });
}

async function copyTranscript(itemId: string, text: string, button: HTMLButtonElement) {
  try {
    await navigator.clipboard.writeText(text);
    copiedHistoryIds.add(itemId);
    applyHistoryCopyButtonState(button, true);

    const existingTimer = copiedHistoryTimers.get(itemId);
    if (existingTimer) {
      window.clearTimeout(existingTimer);
    }

    const timer = window.setTimeout(() => {
      copiedHistoryIds.delete(itemId);
      copiedHistoryTimers.delete(itemId);
      if (button.isConnected) {
        applyHistoryCopyButtonState(button, false);
      }
    }, 1000);

    copiedHistoryTimers.set(itemId, timer);
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  }
}

function getFilteredHistory(viewState: BridgeViewState | null): TranscriptHistoryItem[] {
  if (!viewState) return [];
  return (viewState.session.history || []).filter((item) => {
    return historyClearedAfter === null || item.completed_at > historyClearedAfter;
  });
}

function buildRendererAutomationSnapshot(viewState: BridgeViewState | null): RendererAutomationSnapshot {
  const filteredHistory = getFilteredHistory(viewState);
  return {
    statusBadgeText: statusBadge.textContent?.trim() || "",
    statusBadgeClassName: statusBadge.className,
    statusLine: statusLine.textContent?.trim() || "",
    toggleButtonText: toggleButton.textContent?.trim() || "",
    toggleButtonDisabled: toggleButton.disabled,
    clearHistoryButtonDisabled: clearHistoryButton.disabled,
    busyOverlayVisible: !busyOverlay.classList.contains("hidden"),
    bridgeUrl: bridgeUrl.textContent?.trim() || (viewState?.bridgeUrl ?? ""),
    bridgeCommand: bridgeCommand.textContent?.trim() || (viewState?.bridgeStartCommand ?? ""),
    errorText: errorBox.textContent?.trim() || "",
    historyCount: filteredHistory.length,
    historyTexts: filteredHistory.map((item) => item.payload.transcript || ""),
  };
}

function buildStateSignature(viewState: BridgeViewState): string {
  return JSON.stringify({
    connected: viewState.connected,
    bridgeUrl: viewState.bridgeUrl,
    bridgeStartCommand: viewState.bridgeStartCommand,
    preferredBackend: viewState.preferredBackend,
    hotkey: viewState.hotkey,
    hotkeyRegistered: viewState.hotkeyRegistered,
    session: viewState.session,
    historyClearedAfter,
    visibleHistoryCount,
  });
}


function normalizeBackendName(value: unknown): BackendName {
  return value === "parakeet" || value === "willow" ? value : "whisper";
}

function buildViewStateFromHealthPayload(payload: BridgeHealthPayload): BridgeViewState {
  return {
    bridgeUrl: currentState?.bridgeUrl ?? "",
    bridgeStartCommand: currentState?.bridgeStartCommand ?? "",
    preferredBackend: normalizeBackendName(payload.bridge?.model_backend ?? currentState?.preferredBackend),
    hotkey: currentState?.hotkey ?? "Ctrl+Alt+R",
    hotkeyRegistered: currentState?.hotkeyRegistered ?? false,
    connected: true,
    session: payload.session,
  };
}

function applyObservedBridgeState(viewState: BridgeViewState, { allowCue = true }: { allowCue?: boolean } = {}) {
  const nextSessionState: SessionStatus = viewState.connected ? viewState.session.state : "offline";
  const nextStartCueAt = viewState.connected
    ? (typeof viewState.session.recording_started_at === "number" ? viewState.session.recording_started_at : viewState.session.started_at)
    : null;
  const nextCompleteCueAt = viewState.connected
    ? (typeof viewState.session.clipboard_copied_at === "number" ? viewState.session.clipboard_copied_at : viewState.session.last_completed_at)
    : null;

  if (allowCue) {
    if (nextSessionState === "recording" && nextStartCueAt !== null && nextStartCueAt >= APP_STARTED_AT && nextStartCueAt !== lastPlayedStartCueAt) {
      lastPlayedStartCueAt = nextStartCueAt;
      playSessionCue("start");
    }

    if (nextCompleteCueAt !== null && nextCompleteCueAt >= APP_STARTED_AT && nextCompleteCueAt !== lastPlayedCompleteCueAt) {
      lastPlayedCompleteCueAt = nextCompleteCueAt;
      playSessionCue("complete");
    }
  }

  lastObservedSessionState = nextSessionState;
}

function applyBridgeViewState(viewState: BridgeViewState, { allowCue = true }: { allowCue?: boolean } = {}) {
  applyObservedBridgeState(viewState, { allowCue });
  const nextSignature = buildStateSignature(viewState);
  if (nextSignature !== lastRenderedStateSignature) {
    renderState(viewState);
    lastRenderedStateSignature = nextSignature;
  } else {
    currentState = viewState;
  }
}

function connectBridgeEvents() {
  const nextUrl = currentState?.bridgeUrl;
  if (!nextUrl || bridgeEventSource) {
    return;
  }

  bridgeEventSource = new EventSource(`${nextUrl}/events`);
  bridgeEventSource.addEventListener("session", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent<string>).data) as BridgeHealthPayload;
      applyBridgeViewState(buildViewStateFromHealthPayload(payload));
    } catch (error) {
      errorBox.textContent = error instanceof Error ? error.message : String(error);
    }
  });
  bridgeEventSource.onerror = () => {
    if (bridgeEventSource?.readyState === EventSource.CLOSED) {
      bridgeEventSource.close();
      bridgeEventSource = null;
    }
  };
}

function renderEmptyHistory() {
  historyMeta.textContent = "Logbook empty";
  historyList.innerHTML = `
    <div class="history-empty">
      <div class="history-empty-kicker">Awaiting first pass</div>
      <div class="history-empty-title">No transcripts yet</div>
      <div class="history-empty-copy">Start a recording and completed transcripts will land here for quick copy and review.</div>
    </div>
  `;
  showMoreButton.classList.add("hidden");
}

function renderHistory(items: TranscriptHistoryItem[]) {
  if (!items.length) {
    renderEmptyHistory();
    return;
  }

  const total = items.length;
  const visibleItems = items.slice().reverse().slice(0, visibleHistoryCount);
  historyMeta.textContent = `${total} entr${total === 1 ? "y" : "ies"} in the logbook`;
  historyList.innerHTML = visibleItems
    .map((item) => {
      const payload = item.payload;
      const cancelled = Boolean(payload.metadata?.cancelled_before_recording);
      const transcript = payload.transcript?.trim() || "";
      const body = transcript || (cancelled ? "Cancelled before recording started." : "No transcript text was returned.");
      const device = payload.device ? ` · ${escapeHtml(String(payload.device))}` : "";
      const isCopied = copiedHistoryIds.has(item.id);
      const copyAction = transcript
        ? `<button class="history-copy secondary ${isCopied ? "is-copied" : ""}" data-id="${escapeHtml(item.id)}" data-copy="${encodeURIComponent(transcript)}" aria-label="${isCopied ? "Copied" : "Copy transcript"}" title="${isCopied ? "Copied" : "Copy transcript"}">${renderHistoryActionIcon(isCopied)}</button>`
        : "";
      return `
        <article class="history-item ${cancelled ? "cancelled" : ""}">
          <div class="history-header">
            <div class="history-time">${escapeHtml(formatTimestamp(item.completed_at))}${device}</div>
            <div class="history-actions">${copyAction}</div>
          </div>
          <div class="history-body ${transcript ? "" : "muted"}">${escapeHtml(body)}</div>
        </article>
      `;
    })
    .join("");

  showMoreButton.classList.toggle("hidden", visibleHistoryCount >= total);
  historyList.querySelectorAll<HTMLButtonElement>(".history-copy").forEach((button) => {
    applyHistoryCopyButtonState(button, copiedHistoryIds.has(button.dataset.id || ""));
    button.addEventListener("click", () => {
      const itemId = button.dataset.id || "";
      void copyTranscript(itemId, decodeURIComponent(button.dataset.copy || ""), button);
    });
  });
}

function renderState(viewState: BridgeViewState) {
  const previousState = currentState;
  const previousFilteredCount = getFilteredHistory(previousState).length;
  const nextFilteredHistory = getFilteredHistory(viewState);
  if (nextFilteredHistory.length > previousFilteredCount) {
    visibleHistoryCount = Math.max(visibleHistoryCount, 10);
  }

  currentState = viewState;
  const sessionState: SessionStatus = viewState.connected ? viewState.session.state : "offline";
  const runningBackend = String(viewState.session.config.backend || viewState.preferredBackend) as BackendName;
  statusBadge.textContent = viewState.connected ? viewState.session.state : "Disconnected";
  statusBadge.className = `badge ${sessionState}`;

  hotkeyValue.innerHTML = renderHotkey(viewState.hotkey);
  modelValue.textContent = renderBackendLabel(runningBackend);
  switchModelButton.textContent = `Use ${renderBackendLabel(nextBackend(runningBackend))}`;
  bridgeUrl.textContent = viewState.bridgeUrl;
  bridgeCommand.textContent = viewState.bridgeStartCommand;
  renderHistory(nextFilteredHistory);

  const lockedForBusyWork = viewState.connected && (viewState.session.model_loading || ["starting", "transcribing"].includes(viewState.session.state));
  if (viewState.session.model_loading) {
    setOverlay(true, "Loading model…", "The bridge is warming the model in the background. Recording will unlock automatically when it is ready.");
  } else if (viewState.session.state === "starting") {
    setOverlay(true, "Preparing to record…", "Recording is being prepared. Please wait a moment.");
  } else if (viewState.session.state === "transcribing") {
    setOverlay(true, "Transcribing…", "Audio has been captured. Please wait while the active model generates the transcript.");
  } else {
    setOverlay(false);
  }

  toggleButton.disabled = !viewState.connected || viewState.session.model_loading || viewState.session.state === "transcribing";
  switchModelButton.disabled = lockedForBusyWork;
  clearHistoryButton.disabled = !nextFilteredHistory.length || lockedForBusyWork;

  if (!viewState.connected) {
    toggleButton.textContent = "Bridge offline";
    statusLine.textContent = viewState.hotkeyRegistered
      ? "Start the bridge command below, then use the button or hotkey."
      : "Start the bridge command below, then use the button or your configured hotkey.";
  } else if (viewState.session.model_loading) {
    toggleButton.textContent = "Loading model…";
    statusLine.textContent = "Bridge is warming the model. Recording will unlock automatically when ready.";
  } else if (viewState.session.state === "starting") {
    toggleButton.textContent = "Cancel loading";
    statusLine.textContent = "Preparing to record…";
  } else if (viewState.session.state === "recording") {
    toggleButton.textContent = "Stop recording";
    statusLine.textContent = `Recording in progress since ${formatTimestamp(viewState.session.started_at)}.`;
  } else if (viewState.session.state === "transcribing") {
    toggleButton.textContent = "Transcribing…";
    statusLine.textContent = "Audio captured. Waiting for transcript…";
  } else if (viewState.session.state === "error") {
    toggleButton.textContent = "Try again";
    statusLine.textContent = "Bridge reachable but the backend reported an error.";
  } else {
    toggleButton.textContent = viewState.session.model_loaded ? "Start recording" : "Load + record";
    if (runningBackend !== viewState.preferredBackend) {
      statusLine.textContent = `Bridge is using ${renderBackendLabel(runningBackend)}. Restart the bridge to switch to ${renderBackendLabel(viewState.preferredBackend)}.`;
    } else {
      statusLine.textContent = viewState.session.model_loaded
        ? viewState.hotkeyRegistered
          ? "Model ready. Press the button or the hotkey to begin."
          : "Model ready. Press the button or your configured hotkey to begin."
        : runningBackend === "willow"
          ? "Ready. The first recording connects securely to Willow Voice."
          : viewState.hotkeyRegistered
            ? "Ready. First recording will load the model; after that it stays warm."
            : "Ready. First recording will load the model; after that it stays warm. You can also use your configured hotkey.";
    }
  }

  const errorLines = [];
  if (viewState.session.last_error) errorLines.push(viewState.session.last_error);
  if (viewState.session.stderr_tail.length) errorLines.push(...viewState.session.stderr_tail.slice(-20));
  errorBox.textContent = errorLines.length ? errorLines.join("\n") : "No bridge errors.";
  errorBox.scrollTop = errorBox.scrollHeight;
}

async function refreshState({ quiet = false }: { quiet?: boolean } = {}) {
  if (!quiet) {
    setBusy(true);
  }
  try {
    const state = await electrobun.rpc!.request.getBridgeState({});
    applyBridgeViewState(state);
  } finally {
    if (!quiet) {
      setBusy(false);
    }
  }
}

async function toggleRecording() {
  setBusy(true);
  try {
    const current = await electrobun.rpc!.request.getBridgeState({});
    const next = current.connected && ["starting", "recording"].includes(current.session.state)
      ? await electrobun.rpc!.request.stopRecording({})
      : await electrobun.rpc!.request.startRecording({});
    applyBridgeViewState(next);
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

async function toggleModel() {
  setBusy(true);
  try {
    const next = await electrobun.rpc!.request.toggleBackend({});
    applyBridgeViewState(next, { allowCue: false });
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

async function clearHistory() {
  const shouldClear = await confirmAction(
    "Clear history?",
    "Clear all transcription history for this bridge session? This cannot be undone.",
  );
  if (!shouldClear) return;
  setBusy(true);
  try {
    historyClearedAfter = Date.now() / 1000;
    const next = await electrobun.rpc!.request.clearHistory({});
    visibleHistoryCount = 10;
    applyBridgeViewState(next, { allowCue: false });
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

async function minimizeWindow() {
  try {
    await electrobun.rpc!.request.minimizeWindow({});
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  }
}

async function closeWindow() {
  try {
    await electrobun.rpc!.request.closeWindow({});
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  }
}

minimizeWindowButton.addEventListener("click", () => {
  void minimizeWindow();
});
closeWindowButton.addEventListener("click", () => {
  void closeWindow();
});
toggleButton.addEventListener("click", () => {
  void primeSessionAudio();
  void toggleRecording();
});
switchModelButton.addEventListener("click", () => {
  void toggleModel();
});
clearHistoryButton.addEventListener("click", () => {
  void clearHistory();
});
showMoreButton.addEventListener("click", () => {
  visibleHistoryCount += 10;
  renderHistory(getFilteredHistory(currentState));
});
window.addEventListener("keydown", (event) => {
  void primeSessionAudio();
  const pressedR = event.key.toLowerCase() === "r";
  if (pressedR && event.ctrlKey && event.altKey && currentState && !currentState.hotkeyRegistered) {
    event.preventDefault();
    void toggleRecording();
  }
});
window.addEventListener("pointerdown", () => {
  void primeSessionAudio();
}, { passive: true });
window.addEventListener("resize", syncViewportDensity);

async function bootstrap() {
  try {
    await electrobun.rpc!.request.reportRendererReady({
      userAgent: navigator.userAgent,
    });
    await refreshState();
    connectBridgeEvents();
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : String(error);
  }
}

syncViewportDensity();
void bootstrap();
