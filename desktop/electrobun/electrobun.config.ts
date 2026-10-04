import type { ElectrobunConfig } from "electrobun";

export default {
  app: {
    name: "local-ai-dictation-desktop",
    identifier: "local-ai-dictation.desktop.local",
    version: "0.0.1",
  },
  runtime: {
    exitOnLastWindowClosed: false,
  },
  build: {
    bun: {
      entrypoint: "src/bun/index.ts",
    },
    views: {
      mainview: {
        entrypoint: "src/mainview/index.ts",
      },
    },
    copy: {
      "src/mainview/index.html": "views/mainview/index.html",
      "src/mainview/index.css": "views/mainview/index.css",
      "src/mainview/assets/local-ai-dictation-icon.svg":
        "views/mainview/assets/local-ai-dictation-icon.svg",
      "src/mainview/assets/local-ai-dictation-icon.png":
        "views/mainview/assets/local-ai-dictation-icon.png",
      "src/mainview/assets/session-start.wav": "views/mainview/assets/session-start.wav",
      "src/mainview/assets/session-complete.wav": "views/mainview/assets/session-complete.wav",
    },
    mac: {
      bundleCEF: false,
    },
    linux: {
      bundleCEF: false,
      icon: "src/mainview/assets/local-ai-dictation-icon.png",
    },
    win: {
      bundleCEF: false,
      icon: "src/mainview/assets/local-ai-dictation-icon.ico",
    },
  },
} satisfies ElectrobunConfig;
