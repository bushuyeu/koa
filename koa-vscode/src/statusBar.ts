import * as vscode from "vscode";
import { SessionManager } from "./sessionManager";

/**
 * Status bar item showing active KOA session count.
 */
export class KoaStatusBar implements vscode.Disposable {
  private item: vscode.StatusBarItem;
  private disposables: vscode.Disposable[] = [];

  constructor(private manager: SessionManager) {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = "koa.showSessions";

    this.disposables.push(manager.onDidChange(() => this.update()));
    this.update();
  }

  private update(): void {
    const sessions = this.manager.getAll();
    if (sessions.length === 0) {
      this.item.hide();
      return;
    }

    const gpuList = sessions.map((s) => s.gpuType || "CPU").join(", ");
    this.item.text = `$(server) KOA: ${sessions.length} session${sessions.length !== 1 ? "s" : ""}`;
    this.item.tooltip = `Active: ${gpuList}\nClick to manage sessions`;
    this.item.show();
  }

  dispose(): void {
    this.item.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}
