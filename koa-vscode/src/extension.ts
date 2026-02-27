import * as vscode from "vscode";
import type {
  Jupyter,
  JupyterServer,
  JupyterServerCommand,
  JupyterServerCommandProvider,
  JupyterServerProvider,
} from "@vscode/jupyter-extension";
import { SessionManager } from "./sessionManager";
import { KoaStatusBar } from "./statusBar";
import { pickGpuType } from "./gpuPicker";
import { KoaSessionInfo } from "./types";

let sessionManager: SessionManager;
let statusBar: KoaStatusBar;

function sessionToServer(info: KoaSessionInfo): JupyterServer {
  return {
    id: info.jobId,
    label: `KOA: ${info.gpuType || "CPU"} (job ${info.jobId})`,
    connectionInformation: {
      baseUrl: vscode.Uri.parse(`http://localhost:${info.localPort}/`),
      token: info.token,
    },
  };
}

export function activate(context: vscode.ExtensionContext): void {
  // Get the Jupyter extension API
  const jupyterExt =
    vscode.extensions.getExtension<Jupyter>("ms-toolsai.jupyter");
  if (!jupyterExt) {
    vscode.window.showErrorMessage(
      "KOA: Jupyter extension (ms-toolsai.jupyter) is required but not installed."
    );
    return;
  }

  // Ensure Jupyter extension is active
  const jupyterReady = jupyterExt.isActive
    ? Promise.resolve(jupyterExt.exports)
    : jupyterExt.activate();

  Promise.resolve(jupyterReady).then((jupyter) => {
    setup(context, jupyter);
  });
}

function setup(context: vscode.ExtensionContext, jupyter: Jupyter): void {
  sessionManager = new SessionManager();
  statusBar = new KoaStatusBar(sessionManager);
  context.subscriptions.push(sessionManager, statusBar);

  // Server provider: returns active sessions as selectable servers
  const serverProvider: JupyterServerProvider = {
    provideJupyterServers: () => {
      return sessionManager.getAll().map(sessionToServer);
    },
    resolveJupyterServer: (server: JupyterServer) => {
      return server;
    },
  };

  // Create the KOA section in the kernel picker
  const collection = jupyter.createJupyterServerCollection(
    "koa-vscode:hpc",
    "KOA HPC",
    serverProvider
  );
  context.subscriptions.push(collection);

  // Command provider: the actions shown when user picks "KOA HPC"
  const commandProvider: JupyterServerCommandProvider = {
    provideCommands: () => {
      const commands: JupyterServerCommand[] = [
        { label: "Auto-connect", description: "Best available GPU" },
        { label: "New Server...", description: "Choose GPU type" },
        { label: "Open Web UI", description: "Open Jupyter in browser" },
      ];
      return commands;
    },

    handleCommand: async (
      command: JupyterServerCommand,
      token: vscode.CancellationToken
    ): Promise<JupyterServer | undefined> => {
      switch (command.label) {
        case "Auto-connect":
          return handleAutoConnect(token);
        case "New Server...":
          return handleNewServer(token);
        case "Open Web UI":
          return handleOpenWeb();
        default:
          return undefined;
      }
    },
  };
  collection.commandProvider = commandProvider;

  // Register VS Code commands
  context.subscriptions.push(
    vscode.commands.registerCommand("koa.showSessions", showSessions),
    vscode.commands.registerCommand("koa.killSession", killSession),
    vscode.commands.registerCommand("koa.openWebUI", () => handleOpenWeb())
  );
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async function handleAutoConnect(
  token: vscode.CancellationToken
): Promise<JupyterServer | undefined> {
  try {
    // koa auto-selects best GPU when --gpu-type is not specified
    const info = await sessionManager.getOrCreate(undefined, 1);
    return sessionToServer(info);
  } catch (err) {
    if (token.isCancellationRequested) {
      return undefined;
    }
    vscode.window.showErrorMessage(`KOA: ${err}`);
    return undefined;
  }
}

async function handleNewServer(
  token: vscode.CancellationToken
): Promise<JupyterServer | undefined> {
  const pick = await pickGpuType();
  if (!pick) {
    return undefined; // User cancelled
  }

  try {
    const info = await sessionManager.getOrCreate(
      pick.gpuType || undefined,
      pick.gpus
    );
    return sessionToServer(info);
  } catch (err) {
    if (token.isCancellationRequested) {
      return undefined;
    }
    vscode.window.showErrorMessage(`KOA: ${err}`);
    return undefined;
  }
}

async function handleOpenWeb(): Promise<undefined> {
  const sessions = sessionManager.getAll();
  if (sessions.length === 0) {
    vscode.window.showInformationMessage(
      "KOA: No active sessions. Start one first."
    );
    return undefined;
  }

  let session = sessions[0];
  if (sessions.length > 1) {
    const picked = await vscode.window.showQuickPick(
      sessions.map((s) => ({
        label: `${s.gpuType || "CPU"} (job ${s.jobId})`,
        detail: s.url,
        session: s,
      })),
      { title: "Select session to open in browser" }
    );
    if (!picked) {
      return undefined;
    }
    session = picked.session;
  }

  vscode.env.openExternal(vscode.Uri.parse(session.url));
  return undefined;
}

// ---------------------------------------------------------------------------
// Palette commands
// ---------------------------------------------------------------------------

async function showSessions(): Promise<void> {
  const sessions = sessionManager.getAll();
  if (sessions.length === 0) {
    vscode.window.showInformationMessage("KOA: No active sessions.");
    return;
  }

  const items = sessions.map((s) => ({
    label: `${s.gpuType || "CPU"} (job ${s.jobId})`,
    description: `node: ${s.node}`,
    detail: `Port ${s.localPort} | Started ${s.startedAt.toLocaleTimeString()}`,
    session: s,
  }));

  const picked = await vscode.window.showQuickPick(items, {
    title: "KOA Active Sessions",
    placeHolder: "Select a session for options",
  });

  if (!picked) {
    return;
  }

  const action = await vscode.window.showQuickPick(
    [
      { label: "Open in browser", id: "web" },
      { label: "Terminate session", id: "kill" },
    ],
    { title: `Session ${picked.session.jobId}` }
  );

  if (action?.id === "web") {
    vscode.env.openExternal(vscode.Uri.parse(picked.session.url));
  } else if (action?.id === "kill") {
    sessionManager.kill(picked.session.jobId);
    vscode.window.showInformationMessage(
      `KOA: Session ${picked.session.jobId} terminated.`
    );
  }
}

async function killSession(): Promise<void> {
  const sessions = sessionManager.getAll();
  if (sessions.length === 0) {
    vscode.window.showInformationMessage("KOA: No active sessions.");
    return;
  }

  const picked = await vscode.window.showQuickPick(
    sessions.map((s) => ({
      label: `${s.gpuType || "CPU"} (job ${s.jobId})`,
      description: `node: ${s.node}`,
      session: s,
    })),
    { title: "KOA: Select session to terminate" }
  );

  if (picked) {
    sessionManager.kill(picked.session.jobId);
    vscode.window.showInformationMessage(
      `KOA: Session ${picked.session.jobId} terminated.`
    );
  }
}

export function deactivate(): void {
  // sessionManager.dispose() called via context.subscriptions
}
