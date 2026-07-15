import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { spawn } from "node:child_process";

import { FFIType, dlopen } from "bun:ffi";
import { BrowserView, BrowserWindow, GlobalShortcut, Tray, type RPCSchema } from "electrobun/bun";
import { native, toCString } from "../../node_modules/electrobun/dist/api/bun/proc/native";
import { backendToggleCommand, bridgeEndpoint } from "./backend-command";

type SessionPayload = {
  schema_version: number;
  state: "stopped" | "idle" | "starting" | "recording" | "transcribing" | "error";
  started_at: number | null;
  recording_started_at: number | null;
  last_completed_at: number | null;
  clipboard_copied_at: number | null;
  last_transcript: {
    schema_version: number;
    transcript: string;
    normalized_transcript?: string;
    device?: string;
    metadata?: Record<string, unknown>;
  } | null;
  last_error: string | null;
  model_loaded: boolean;
  model_loading: boolean;
  history: Array<{ id: string; completed_at: number; payload: Record<string, unknown> }>;
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

type RendererReadyPayload = {
  userAgent: string;
};

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

type StartupDiagnostics = {
  bridgeUrl: string;
  hotkey: string;
  e2eEnabled: boolean;
  automationPort: number | null;
  automationReady: boolean;
  automationReadyAt: string | null;
  bunReady: boolean;
  bunReadyAt: string;
  trayCreated: boolean;
  trayCreatedAt: string | null;
  hotkeyRegistered: boolean;
  hotkeyRegisteredAt: string | null;
  rendererReady: boolean;
  rendererReadyAt: string | null;
  rendererRpcReady: boolean;
  rendererUserAgent: string | null;
  lastAutomationAction: string | null;
  shutdownReason: string | null;
  shutdownAt: string | null;
  lastError: string | null;
};

type AutomationActionId =
  | "show-window"
  | "window/toggle-recording"
  | "start-recording"
  | "stop-recording"
  | "toggle-recording"
  | "clear-history"
  | "tray/open"
  | "tray/toggle"
  | "tray/quit"
  | "hotkey/trigger"
  | "quit";

type AutomationState = {
  enabled: boolean;
  port: number | null;
  tray: {
    created: boolean;
    actions: string[];
  };
  hotkey: {
    accelerator: string;
    registered: boolean;
  };
  renderer: {
    ready: boolean;
    rpcReady: boolean;
    userAgent: string | null;
    snapshot: RendererAutomationSnapshot | null;
  };
  bridge: BridgeViewState;
  diagnostics: StartupDiagnostics;
};

type DesktopRPC = {
  bun: RPCSchema<{
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
      reportRendererReady: { params: RendererReadyPayload; response: { success: true } };
    };
    messages: {};
  }>;
  webview: RPCSchema<{
    requests: {
      getAutomationSnapshot: { params: {}; response: RendererAutomationSnapshot };
      runAutomationAction: { params: { action: RendererAutomationActionId }; response: RendererAutomationSnapshot };
      playSessionCue: { params: { kind: SessionCueKind }; response: { success: true } };
    };
    messages: {};
  }>;
};

const DEFAULT_BRIDGE_URL = "http://127.0.0.1:8765";
const DEFAULT_BACKEND: BackendName = "whisper";
const BRIDGE_URL = Bun.env.LOCAL_AI_DICTATION_BRIDGE_URL || DEFAULT_BRIDGE_URL;
const BRIDGE_COMMAND_OVERRIDE = Bun.env.LOCAL_AI_DICTATION_BRIDGE_COMMAND || "";
const IS_LINUX_WAYLAND = process.platform === "linux"
  && ((Bun.env.XDG_SESSION_TYPE || "").toLowerCase() === "wayland" || Boolean(Bun.env.WAYLAND_DISPLAY));
const DEFAULT_HOTKEY = process.platform === "linux" ? "Control+Alt+R" : "CommandOrControl+Alt+R";

function normalizeHotkey(accelerator: string): string {
  return accelerator
    .split("+")
    .map((part) => {
      const trimmed = part.trim();
      const normalized = trimmed.toLowerCase();
      if (normalized === "ctrl") {
        return "Control";
      }
      if (process.platform === "linux" && (normalized === "commandorcontrol" || normalized === "cmdorctrl")) {
        return "Control";
      }
      return trimmed;
    })
    .filter(Boolean)
    .join("+");
}

const HOTKEY = normalizeHotkey(Bun.env.LOCAL_AI_DICTATION_HOTKEY || DEFAULT_HOTKEY);
const GUI_E2E_ENABLED = Bun.env.LOCAL_AI_DICTATION_GUI_E2E === "1";
const GUI_E2E_PORT = Number(Bun.env.LOCAL_AI_DICTATION_GUI_E2E_PORT || "0") || 0;
const DEFAULT_STARTUP_LOG_DIR = Bun.env.LOCALAPPDATA
  ? join(Bun.env.LOCALAPPDATA, "local-ai-dictation.desktop.local", "stable", "logs")
  : Bun.env.XDG_STATE_HOME
    ? join(Bun.env.XDG_STATE_HOME, "local-ai-dictation", "desktop")
    : Bun.env.HOME
      ? join(Bun.env.HOME, ".local", "state", "local-ai-dictation", "desktop")
      : "";
const GUI_LOG_PATH = Bun.env.LOCAL_AI_DICTATION_GUI_LOG_PATH || (DEFAULT_STARTUP_LOG_DIR ? join(DEFAULT_STARTUP_LOG_DIR, "startup.log") : "");
const GUI_STARTUP_DIAGNOSTICS_PATH = Bun.env.LOCAL_AI_DICTATION_GUI_STARTUP_DIAGNOSTICS_PATH
  || (DEFAULT_STARTUP_LOG_DIR ? join(DEFAULT_STARTUP_LOG_DIR, "startup-diagnostics.json") : "");
const GUI_AUTO_EXIT_MS = Number(Bun.env.LOCAL_AI_DICTATION_GUI_AUTO_EXIT_MS || "0") || 0;
const APP_STARTED_AT = Date.now() / 1000;
const APP_ICON_URL = "views://mainview/assets/local-ai-dictation-icon.png";
const BACKEND_STATE_PATH = Bun.env.XDG_STATE_HOME
  ? join(Bun.env.XDG_STATE_HOME, "local-ai-dictation", "backend.json")
  : Bun.env.HOME
    ? join(Bun.env.HOME, ".local", "state", "local-ai-dictation", "backend.json")
    : "/tmp/local-ai-dictation-backend.json";
const APP_WINDOW_ICON_PATH = join(import.meta.dir, "..", "..", "app.ico");

const SND_ASYNC = 0x0001;
const SND_NODEFAULT = 0x0002;
const SND_FILENAME = 0x00020000;

const nativeSoundPlayer = process.platform === "win32"
  ? dlopen("winmm.dll", {
      PlaySoundA: {
        args: [FFIType.cstring, FFIType.ptr, FFIType.u32],
        returns: FFIType.bool,
      },
    })
  : null;
const nativeCuePlayerCommands = [
  [Bun.which("paplay"), []],
  [Bun.which("pw-play"), []],
  [Bun.which("ffplay"), ["-v", "quiet", "-nodisp", "-autoexit"]],
] as const;

function normalizeBackend(value: string | null | undefined): BackendName {
  return value === "parakeet" || value === "willow" ? value : "whisper";
}

function readPreferredBackend(): BackendName {
  try {
    const payload = JSON.parse(Bun.file(BACKEND_STATE_PATH).textSync());
    return normalizeBackend(typeof payload?.backend === "string" ? payload.backend : undefined);
  } catch {
    return DEFAULT_BACKEND;
  }
}

function writePreferredBackend(backend: BackendName): BackendName {
  mkdirSync(dirname(BACKEND_STATE_PATH), { recursive: true });
  writeFileSync(BACKEND_STATE_PATH, `${JSON.stringify({ backend })}\n`);
  return backend;
}

function togglePreferredBackend(): BackendName {
  const order: BackendName[] = ["whisper", "parakeet", "willow"];
  const currentIndex = order.indexOf(readPreferredBackend());
  return writePreferredBackend(order[(currentIndex + 1) % order.length]);
}

function localCliPath(): string {
  const home = Bun.env.HOME;
  if (home) {
    const preferred = join(home, ".local", "bin", "local-ai-dictation");
    if (existsSync(preferred)) {
      return preferred;
    }
  }
  return Bun.which("local-ai-dictation") || "local-ai-dictation";
}

function toggleBackendAndRestartBridge(): BackendName {
  const cli = localCliPath();
  const endpoint = bridgeEndpoint(BRIDGE_URL);
  appendGuiLog("INFO", `Switching backend via ${cli} for ${endpoint.host}:${endpoint.port}`);
  const result = Bun.spawnSync({
    cmd: backendToggleCommand(cli, BRIDGE_URL),
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      PATH: `${Bun.env.HOME ? `${join(Bun.env.HOME, ".local", "bin")}:` : ""}${process.env.PATH || ""}`,
    },
  });
  if (result.exitCode !== 0) {
    const error = Buffer.from(result.stderr).toString("utf8").trim() || `backend toggle failed with exit code ${result.exitCode}`;
    throw new Error(error);
  }
  const text = Buffer.from(result.stdout).toString("utf8").trim();
  return normalizeBackend(text || readPreferredBackend());
}

function buildBridgeStartCommand(backend: BackendName): string {
  let host = "127.0.0.1";
  let port = "8765";
  try {
    const parsed = new URL(BRIDGE_URL);
    host = parsed.hostname || host;
    port = parsed.port || port;
  } catch {
    // keep defaults
  }
  return `local-ai-dictation bridge --host ${host} --port ${port} --backend ${backend}`;
}

function resolveSessionCueAssetPath(kind: SessionCueKind): string | null {
  const filename = kind === "start" ? "session-start.wav" : "session-complete.wav";
  const candidates = [
    join(import.meta.dir, "..", "mainview", "assets", filename),
    join(import.meta.dir, "..", "views", "mainview", "assets", filename),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}
const emptySession = (): SessionPayload => ({
  schema_version: 1,
  state: "stopped",
  started_at: null,
  recording_started_at: null,
  last_completed_at: null,
  clipboard_copied_at: null,
  last_transcript: null,
  last_error: null,
  model_loaded: false,
  model_loading: false,
  history: [],
  config: {},
  stderr_tail: [],
});

function timestamp(): string {
  return new Date().toISOString();
}

const startupDiagnostics: StartupDiagnostics = {
  bridgeUrl: BRIDGE_URL,
  hotkey: HOTKEY,
  e2eEnabled: GUI_E2E_ENABLED,
  automationPort: GUI_E2E_PORT > 0 ? GUI_E2E_PORT : null,
  automationReady: false,
  automationReadyAt: null,
  bunReady: true,
  bunReadyAt: timestamp(),
  trayCreated: false,
  trayCreatedAt: null,
  hotkeyRegistered: false,
  hotkeyRegisteredAt: null,
  rendererReady: false,
  rendererReadyAt: null,
  rendererRpcReady: false,
  rendererUserAgent: null,
  lastAutomationAction: null,
  shutdownReason: null,
  shutdownAt: null,
  lastError: null,
};

function ensureFileParent(path: string) {
  mkdirSync(dirname(path), { recursive: true });
}

function writeStartupDiagnostics() {
  if (!GUI_STARTUP_DIAGNOSTICS_PATH) {
    return;
  }
  ensureFileParent(GUI_STARTUP_DIAGNOSTICS_PATH);
  writeFileSync(GUI_STARTUP_DIAGNOSTICS_PATH, `${JSON.stringify(startupDiagnostics, null, 2)}\n`);
}

function updateStartupDiagnostics(patch: Partial<StartupDiagnostics>) {
  Object.assign(startupDiagnostics, patch);
  writeStartupDiagnostics();
}

function appendGuiLog(level: "INFO" | "WARN" | "ERROR", message: string) {
  const line = `[${timestamp()}] [${level}] ${message}`;
  if (level === "ERROR") {
    console.error(message);
  } else if (level === "WARN") {
    console.warn(message);
  } else {
    console.log(message);
  }
  if (GUI_LOG_PATH) {
    ensureFileParent(GUI_LOG_PATH);
    appendFileSync(GUI_LOG_PATH, `${line}\n`);
  }
}

async function fetchBridgeJson(path: string, init?: RequestInit): Promise<any> {
  const response = await fetch(`${BRIDGE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  let payload: any = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.error || `Bridge request failed: ${response.status}`);
  }

  return payload;
}

async function readBridgeState(): Promise<BridgeViewState> {
  const preferredBackend = readPreferredBackend();
  const bridgeStartCommand = BRIDGE_COMMAND_OVERRIDE || buildBridgeStartCommand(preferredBackend);
  try {
    const health = await fetchBridgeJson("/health");
    return {
      bridgeUrl: BRIDGE_URL,
      bridgeStartCommand,
      preferredBackend,
      hotkey: HOTKEY,
      hotkeyRegistered: startupDiagnostics.hotkeyRegistered,
      connected: true,
      session: health.session as SessionPayload,
    };
  } catch (error) {
    return {
      bridgeUrl: BRIDGE_URL,
      bridgeStartCommand,
      preferredBackend,
      hotkey: HOTKEY,
      hotkeyRegistered: startupDiagnostics.hotkeyRegistered,
      connected: false,
      session: {
        ...emptySession(),
        state: "error",
        last_error: error instanceof Error ? error.message : String(error),
      },
    };
  }
}

const HOTKEY_COOLDOWN_MS = 450;

let lastObservedSessionState: SessionPayload["state"] | "offline" = "offline";
let lastPlayedStartCueAt: number | null = null;
let lastPlayedCompleteCueAt: number | null = null;

function playNativeSessionCue(kind: SessionCueKind): boolean {
  const assetPath = resolveSessionCueAssetPath(kind);
  if (!assetPath) {
    appendGuiLog("WARN", `Session cue asset missing for ${kind}`);
    return false;
  }

  if (process.platform === "win32" && nativeSoundPlayer) {
    try {
      const started = nativeSoundPlayer.symbols.PlaySoundA(toCString(assetPath), null, SND_ASYNC | SND_NODEFAULT | SND_FILENAME);
      if (!started) {
        appendGuiLog("WARN", `Native PlaySound failed for ${kind}`);
        return false;
      }
      appendGuiLog("INFO", `Playing ${kind} cue natively with PlaySound`);
      return true;
    } catch (error) {
      appendGuiLog("WARN", `Failed to play ${kind} cue natively: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  for (const [command, args] of nativeCuePlayerCommands) {
    if (!command) {
      continue;
    }
    try {
      const child = spawn(command, [...args, assetPath], {
        detached: true,
        stdio: "ignore",
      });
      child.unref();
      appendGuiLog("INFO", `Playing ${kind} cue via ${command}`);
      return true;
    } catch (error) {
      appendGuiLog("WARN", `Failed to play ${kind} cue via ${command}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return false;
}

async function requestRendererSessionCue(kind: "start" | "complete") {
  if (startupDiagnostics.rendererRpcReady && mainWindow?.webview.rpc) {
    try {
      await mainWindow.webview.rpc.request.playSessionCue({ kind });
      return;
    } catch (error) {
      appendGuiLog("WARN", `Failed to play ${kind} cue in renderer: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  playNativeSessionCue(kind);
}

function applyObservedBridgeState(state: BridgeViewState, { allowCue = true }: { allowCue?: boolean } = {}) {
  const nextSessionState = state.connected ? state.session.state : "offline";
  const nextStartCueAt = state.connected
    ? (typeof state.session.recording_started_at === "number" ? state.session.recording_started_at : state.session.started_at)
    : null;
  const nextCompleteCueAt = state.connected
    ? (typeof state.session.clipboard_copied_at === "number" ? state.session.clipboard_copied_at : state.session.last_completed_at)
    : null;

  if (allowCue) {
    if (nextSessionState === "recording" && nextStartCueAt !== null && nextStartCueAt >= APP_STARTED_AT && nextStartCueAt !== lastPlayedStartCueAt) {
      lastPlayedStartCueAt = nextStartCueAt;
      void requestRendererSessionCue("start");
    }

    if (nextCompleteCueAt !== null && nextCompleteCueAt >= APP_STARTED_AT && nextCompleteCueAt !== lastPlayedCompleteCueAt) {
      lastPlayedCompleteCueAt = nextCompleteCueAt;
      void requestRendererSessionCue("complete");
    }
  }

  lastObservedSessionState = nextSessionState;
}

async function refreshBridgeStateCache() {
  return await readBridgeState();
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json",
      "cache-control": "no-store",
    },
  });
}

function isAutomationAction(value: string): value is AutomationActionId {
  return [
    "show-window",
    "window/toggle-recording",
    "start-recording",
    "stop-recording",
    "toggle-recording",
    "clear-history",
    "tray/open",
    "tray/toggle",
    "tray/quit",
    "hotkey/trigger",
    "quit",
  ].includes(value);
}

async function readRendererAutomationSnapshot(): Promise<RendererAutomationSnapshot | null> {
  if (!startupDiagnostics.rendererRpcReady || !mainWindow?.webview.rpc) {
    return null;
  }
  try {
    return await mainWindow.webview.rpc.request.getAutomationSnapshot({});
  } catch (error) {
    appendGuiLog("WARN", `Failed to read renderer automation snapshot: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}

async function runRendererAutomationAction(action: RendererAutomationActionId): Promise<RendererAutomationSnapshot> {
  if (!startupDiagnostics.rendererRpcReady || !mainWindow?.webview.rpc) {
    throw new Error("Renderer automation is not ready yet");
  }
  return await mainWindow.webview.rpc.request.runAutomationAction({ action });
}

async function readAutomationState(): Promise<AutomationState> {
  return {
    enabled: GUI_E2E_ENABLED,
    port: GUI_E2E_PORT > 0 ? GUI_E2E_PORT : null,
    tray: {
      created: startupDiagnostics.trayCreated,
      actions: ["open", "toggle", "quit"],
    },
    hotkey: {
      accelerator: HOTKEY,
      registered: startupDiagnostics.hotkeyRegistered,
    },
    renderer: {
      ready: startupDiagnostics.rendererReady,
      rpcReady: startupDiagnostics.rendererRpcReady,
      userAgent: startupDiagnostics.rendererUserAgent,
      snapshot: await readRendererAutomationSnapshot(),
    },
    bridge: await refreshBridgeStateCache(),
    diagnostics: { ...startupDiagnostics },
  };
}

let mainWindow: BrowserWindow<any> | null = null;
let tray: Tray | null = null;
let autoExitTimer: ReturnType<typeof setTimeout> | null = null;
let automationServer: ReturnType<typeof Bun.serve> | null = null;
let shuttingDown = false;
let sessionToggleInFlight = false;
let lastHotkeyInvocationAt = 0;

function exitApp(reason: string, exitCode = 0) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  if (autoExitTimer) {
    clearTimeout(autoExitTimer);
    autoExitTimer = null;
  }
  updateStartupDiagnostics({
    shutdownReason: reason,
    shutdownAt: timestamp(),
  });
  appendGuiLog("INFO", `Shutting down app: ${reason}`);
  automationServer?.stop(true);
  GlobalShortcut.unregisterAll();
  tray?.remove();
  process.exit(exitCode);
}

function scheduleAutoExit() {
  if (!GUI_E2E_ENABLED || GUI_AUTO_EXIT_MS <= 0 || autoExitTimer) {
    return;
  }
  appendGuiLog("INFO", `Scheduling E2E auto-exit in ${GUI_AUTO_EXIT_MS}ms`);
  autoExitTimer = setTimeout(() => {
    exitApp("e2e-auto-exit");
  }, GUI_AUTO_EXIT_MS);
}

process.on("uncaughtException", (error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  updateStartupDiagnostics({ lastError: message });
  appendGuiLog("ERROR", `Uncaught exception: ${message}`);
  exitApp("uncaught-exception", 1);
});

process.on("unhandledRejection", (reason) => {
  const message = reason instanceof Error ? reason.stack || reason.message : String(reason);
  updateStartupDiagnostics({ lastError: message });
  appendGuiLog("ERROR", `Unhandled rejection: ${message}`);
  exitApp("unhandled-rejection", 1);
});

writeStartupDiagnostics();
appendGuiLog("INFO", "Creating BrowserWindow...");
mainWindow = new BrowserWindow({
  title: "Local AI Dictation",
  url: "views://mainview/index.html",
  titleBarStyle: "default",
  transparent: false,
  frame: {
    width: 540,
    height: 1080,
    x: 220,
    y: 120,
  },
  rpc: BrowserView.defineRPC<DesktopRPC>({
    maxRequestTime: 30000,
    handlers: {
      requests: {
        getBridgeState: async () => {
          return await refreshBridgeStateCache();
        },
        startRecording: async () => {
          await fetchBridgeJson("/session/start", { method: "POST", body: "{}" });
          return await refreshBridgeStateCache();
        },
        stopRecording: async () => {
          await fetchBridgeJson("/session/stop", { method: "POST", body: "{}" });
          return await refreshBridgeStateCache();
        },
        toggleRecording: async () => {
          await fetchBridgeJson("/session/toggle", { method: "POST", body: "{}" });
          return await refreshBridgeStateCache();
        },
        toggleBackend: async () => {
          toggleBackendAndRestartBridge();
          return await refreshBridgeStateCache();
        },
        clearHistory: async () => {
          await fetchBridgeJson("/session/clear-history", { method: "POST", body: "{}" });
          return await refreshBridgeStateCache();
        },
        showWindow: async () => {
          showMainWindow();
          return { success: true } as const;
        },
        minimizeWindow: async () => {
          mainWindow?.minimize();
          return { success: true } as const;
        },
        closeWindow: async () => {
          triggerQuit("window-control:close");
          return { success: true } as const;
        },
        reportRendererReady: async ({ userAgent }) => {
          updateStartupDiagnostics({
            rendererReady: true,
            rendererReadyAt: timestamp(),
            rendererRpcReady: true,
            rendererUserAgent: userAgent,
          });
          appendGuiLog("INFO", `Renderer ready via RPC: ${userAgent}`);
          scheduleAutoExit();
          return { success: true } as const;
        },
      },
      messages: {},
    },
  }),
});
appendGuiLog("INFO", "BrowserWindow created");
applyWindowsWindowIcon();

const MIN_WINDOW_WIDTH = 540;
const MIN_WINDOW_HEIGHT = 1080;

async function toggleFromBackground(source: "hotkey" | "tray" | "automation" = "tray") {
  const now = Date.now();
  if (source === "hotkey" && now - lastHotkeyInvocationAt < HOTKEY_COOLDOWN_MS) {
    return;
  }
  if (source === "hotkey") {
    lastHotkeyInvocationAt = now;
  }
  if (sessionToggleInFlight) {
    appendGuiLog("WARN", `Ignoring ${source} toggle while another session action is in flight`);
    return;
  }

  sessionToggleInFlight = true;
  try {
    await fetchBridgeJson("/session/toggle", { method: "POST", body: "{}" });
    await refreshBridgeStateCache();
  } catch (error) {
    appendGuiLog("ERROR", `Background toggle failed: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    sessionToggleInFlight = false;
  }
}

function ensureMainWindowSize() {
  if (!mainWindow) return;
  const { width, height } = mainWindow.getSize();
  if (width < MIN_WINDOW_WIDTH || height < MIN_WINDOW_HEIGHT) {
    mainWindow.setSize(Math.max(width, MIN_WINDOW_WIDTH), Math.max(height, MIN_WINDOW_HEIGHT));
  }
}

function applyWindowsWindowIcon() {
  if (process.platform !== "win32" || !mainWindow?.ptr) {
    return;
  }

  try {
    native.symbols.setWindowIcon(mainWindow.ptr, toCString(APP_WINDOW_ICON_PATH));
  } catch (error) {
    appendGuiLog("WARN", `Failed to set window icon: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function showMainWindow() {
  ensureMainWindowSize();
  if (mainWindow?.isMinimized()) {
    mainWindow.unminimize();
  }
  mainWindow?.show();
  mainWindow?.focus();
}

async function triggerHotkeyCallback() {
  appendGuiLog("INFO", `Hotkey callback invoked: ${HOTKEY}`);
  await toggleFromBackground("hotkey");
}

function triggerQuit(action: string) {
  setTimeout(() => {
    exitApp(action);
  }, 0);
}

async function handleTrayAction(action: string, source: "tray" | "automation") {
  switch (action) {
    case "open":
      showMainWindow();
      break;
    case "toggle":
      await toggleFromBackground(source);
      break;
    case "quit":
      triggerQuit(source === "automation" ? "automation:tray/quit" : "tray-quit");
      break;
  }
}

async function runAutomationAction(action: AutomationActionId): Promise<AutomationState> {
  updateStartupDiagnostics({ lastAutomationAction: action });
  appendGuiLog("INFO", `Automation action: ${action}`);

  switch (action) {
    case "show-window":
    case "tray/open":
      await handleTrayAction("open", "automation");
      break;
    case "window/toggle-recording":
      await runRendererAutomationAction("toggle-recording");
      break;
    case "start-recording":
      await fetchBridgeJson("/session/start", { method: "POST", body: "{}" });
      await refreshBridgeStateCache();
      break;
    case "stop-recording":
      await fetchBridgeJson("/session/stop", { method: "POST", body: "{}" });
      await refreshBridgeStateCache();
      break;
    case "toggle-recording":
      await fetchBridgeJson("/session/toggle", { method: "POST", body: "{}" });
      await refreshBridgeStateCache();
      break;
    case "clear-history":
      await fetchBridgeJson("/session/clear-history", { method: "POST", body: "{}" });
      await refreshBridgeStateCache();
      break;
    case "tray/toggle":
      await handleTrayAction("toggle", "automation");
      break;
    case "hotkey/trigger":
      await triggerHotkeyCallback();
      break;
    case "tray/quit":
      await handleTrayAction("quit", "automation");
      break;
    case "quit":
      triggerQuit("automation:quit");
      break;
  }

  return await readAutomationState();
}

function startAutomationServer() {
  if (!GUI_E2E_ENABLED || GUI_E2E_PORT <= 0 || automationServer) {
    return;
  }

  appendGuiLog("INFO", `Starting automation server on http://127.0.0.1:${GUI_E2E_PORT}`);
  try {
    automationServer = Bun.serve({
      hostname: "127.0.0.1",
      port: GUI_E2E_PORT,
      fetch: async (request) => {
        const url = new URL(request.url);
        try {
          if (request.method === "GET" && url.pathname === "/health") {
            return jsonResponse({ ok: true });
          }
          if (request.method === "GET" && url.pathname === "/state") {
            return jsonResponse({ ok: true, state: await readAutomationState() });
          }
          if (request.method === "POST" && url.pathname.startsWith("/actions/")) {
            const action = decodeURIComponent(url.pathname.slice("/actions/".length));
            if (!isAutomationAction(action)) {
              return jsonResponse({ ok: false, error: `Unknown automation action: ${action}` }, 404);
            }
            return jsonResponse({ ok: true, state: await runAutomationAction(action) });
          }
          return jsonResponse({ ok: false, error: `Unknown automation route: ${url.pathname}` }, 404);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          updateStartupDiagnostics({ lastError: message });
          appendGuiLog("ERROR", `Automation request failed: ${message}`);
          return jsonResponse({ ok: false, error: message }, 500);
        }
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.stack || error.message : String(error);
    updateStartupDiagnostics({ lastError: message });
    appendGuiLog("ERROR", `Failed to start automation server: ${message}`);
    return;
  }

  updateStartupDiagnostics({
    automationPort: GUI_E2E_PORT,
    automationReady: true,
    automationReadyAt: timestamp(),
  });
  appendGuiLog("INFO", `Automation server listening on http://127.0.0.1:${GUI_E2E_PORT}`);
}

ensureMainWindowSize();

tray = new Tray({
  title: "Local AI Dictation",
  image: APP_ICON_URL,
  width: 18,
  height: 18,
});
updateStartupDiagnostics({ trayCreated: true, trayCreatedAt: timestamp() });
appendGuiLog("INFO", "Tray created");

mainWindow.on("close", () => {
  exitApp("window-close");
});

tray.setMenu([
  { type: "normal", label: "Open Local AI Dictation", action: "open" },
  { type: "normal", label: `Toggle Recording (${HOTKEY})`, action: "toggle" },
  { type: "divider" },
  { type: "normal", label: "Quit", action: "quit" },
]);
tray.on("tray-clicked", async (event: any) => {
  const action = event.data?.action;
  if (typeof action === "string") {
    await handleTrayAction(action, "tray");
  }
});

appendGuiLog("INFO", "Registering global hotkey...");
let registeredHotkey = false;
if (IS_LINUX_WAYLAND) {
  updateStartupDiagnostics({
    hotkeyRegistered: false,
    hotkeyRegisteredAt: null,
  });
  appendGuiLog("WARN", `Skipping global hotkey registration on Linux Wayland: ${HOTKEY}`);
} else {
  registeredHotkey = GlobalShortcut.register(HOTKEY, () => {
    void triggerHotkeyCallback();
  });
  updateStartupDiagnostics({
    hotkeyRegistered: registeredHotkey,
    hotkeyRegisteredAt: registeredHotkey ? timestamp() : null,
  });
  if (!registeredHotkey) {
    appendGuiLog("WARN", `Failed to register global hotkey: ${HOTKEY}`);
  } else {
    appendGuiLog("INFO", `Global hotkey registered: ${HOTKEY}`);
  }
}

startAutomationServer();

appendGuiLog("INFO", "Local AI Dictation desktop app started");
appendGuiLog("INFO", `Bridge URL: ${BRIDGE_URL}`);
appendGuiLog("INFO", `Hotkey: ${HOTKEY}`);
