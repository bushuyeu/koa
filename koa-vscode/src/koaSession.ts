import { ChildProcess } from "child_process";
import * as vscode from "vscode";
import { KoaSessionInfo } from "./types";
import { spawnJupyterSession } from "./koaCli";

/**
 * A single KOA Jupyter session backed by a `koa jupyter` child process.
 *
 * The child process manages the SLURM job and SSH tunnel.
 * Disposing the session sends SIGTERM, which triggers koa's cleanup
 * (scancel, tunnel kill, remote script removal).
 */
export class KoaSession implements vscode.Disposable {
  private child: ChildProcess | null = null;
  private _info: KoaSessionInfo | null = null;
  private _disposed = false;

  private readonly _onDidDispose = new vscode.EventEmitter<void>();
  readonly onDidDispose = this._onDidDispose.event;

  get info(): KoaSessionInfo | null {
    return this._info;
  }

  get isReady(): boolean {
    return this._info !== null && !this._disposed;
  }

  /**
   * Start the session. Shows a progress notification with cancel support.
   * Resolves when the Jupyter server is reachable via SSH tunnel.
   */
  async start(options: {
    gpuType?: string;
    gpus?: number;
  }): Promise<KoaSessionInfo> {
    const label = options.gpuType || "best GPU";

    return vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `KOA: Allocating ${label}...`,
        cancellable: true,
      },
      async (progress, cancelToken) => {
        const { child, ready } = spawnJupyterSession({
          gpuType: options.gpuType,
          gpus: options.gpus,
          onProgress: (msg) => {
            progress.report({ message: msg });
          },
        });

        this.child = child;

        // Wire cancellation to SIGTERM
        const cancelSub = cancelToken.onCancellationRequested(() => {
          child.kill("SIGTERM");
        });

        // Monitor unexpected exit after session is established
        child.on("exit", (code) => {
          if (!this._disposed && this._info) {
            vscode.window.showWarningMessage(
              `KOA session ${this._info.jobId} ended (exit code ${code})`
            );
            this.dispose();
          }
        });

        try {
          const output = await ready;
          cancelSub.dispose();

          this._info = {
            jobId: output.job_id,
            node: output.node,
            localPort: output.local_port,
            remotePort: output.remote_port,
            url: output.url,
            token: output.token,
            gpuType: output.gpu_type,
            gpus: output.gpus,
            startedAt: new Date(),
          };

          return this._info;
        } catch (err) {
          cancelSub.dispose();
          this.dispose();
          throw err;
        }
      }
    );
  }

  dispose(): void {
    if (this._disposed) {
      return;
    }
    this._disposed = true;

    if (this.child && this.child.exitCode === null) {
      this.child.kill("SIGTERM");
    }

    this._onDidDispose.fire();
    this._onDidDispose.dispose();
  }
}
