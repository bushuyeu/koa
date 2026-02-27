import * as vscode from "vscode";
import { KoaSession } from "./koaSession";
import { KoaSessionInfo } from "./types";

/**
 * Manages all active KOA sessions. Handles session reuse and cleanup.
 */
export class SessionManager implements vscode.Disposable {
  private sessions: Map<string, KoaSession> = new Map();

  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChange = this._onDidChange.event;

  /** Find an existing ready session with matching GPU type. */
  findByGpuType(gpuType: string | undefined): KoaSession | undefined {
    for (const session of this.sessions.values()) {
      if (!session.isReady) {
        continue;
      }
      const info = session.info!;
      if (gpuType === undefined || info.gpuType === gpuType) {
        return session;
      }
    }
    return undefined;
  }

  /** Get an existing session or create a new one. */
  async getOrCreate(
    gpuType?: string,
    gpus?: number
  ): Promise<KoaSessionInfo> {
    // Try to reuse an existing session
    const existing = this.findByGpuType(gpuType);
    if (existing?.isReady) {
      return existing.info!;
    }

    // Create a new session
    const session = new KoaSession();
    const info = await session.start({ gpuType, gpus });

    this.sessions.set(info.jobId, session);

    session.onDidDispose(() => {
      this.sessions.delete(info.jobId);
      this._onDidChange.fire();
    });

    this._onDidChange.fire();
    return info;
  }

  /** All active session infos. */
  getAll(): KoaSessionInfo[] {
    const result: KoaSessionInfo[] = [];
    for (const session of this.sessions.values()) {
      if (session.info) {
        result.push(session.info);
      }
    }
    return result;
  }

  /** Kill a specific session by job ID. */
  kill(jobId: string): void {
    const session = this.sessions.get(jobId);
    if (session) {
      session.dispose();
    }
  }

  /** Kill all sessions. */
  dispose(): void {
    for (const session of this.sessions.values()) {
      session.dispose();
    }
    this.sessions.clear();
    this._onDidChange.dispose();
  }
}
