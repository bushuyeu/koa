import { ChildProcess, execFile, spawn } from "child_process";
import * as vscode from "vscode";
import { AvailabilityOutput, KoaJupyterOutput } from "./types";

const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;

function getCliPath(): string {
  return vscode.workspace.getConfiguration("koa").get<string>("cliPath", "koa");
}

function getConfig(): {
  walltime: string;
  mem: string;
  timeout: number;
  condaEnv: string;
} {
  const cfg = vscode.workspace.getConfiguration("koa");
  return {
    walltime: cfg.get<string>("defaultWalltime", "04:00:00"),
    mem: cfg.get<string>("defaultMemory", "16G"),
    timeout: cfg.get<number>("allocationTimeoutMinutes", 60),
    condaEnv: cfg.get<string>("condaEnv", ""),
  };
}

/**
 * Run `koa availability --format json` and return parsed GPU availability.
 */
export function getAvailability(): Promise<AvailabilityOutput> {
  return new Promise((resolve, reject) => {
    execFile(
      getCliPath(),
      ["availability", "--format", "json"],
      { timeout: 30_000 },
      (err, stdout, stderr) => {
        if (err) {
          reject(new Error(stderr || err.message));
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          reject(new Error(`Failed to parse availability JSON: ${stdout}`));
        }
      }
    );
  });
}

export interface SpawnResult {
  child: ChildProcess;
  ready: Promise<KoaJupyterOutput>;
}

/**
 * Spawn `koa jupyter --format json` with the given options.
 *
 * Returns the child process (which stays alive holding the SSH tunnel)
 * and a promise that resolves with the JSON output once the server is ready.
 *
 * The caller must eventually `child.kill('SIGTERM')` to clean up.
 */
export function spawnJupyterSession(options: {
  gpuType?: string;
  gpus?: number;
  onProgress?: (message: string) => void;
}): SpawnResult {
  const cfg = getConfig();

  const args = [
    "jupyter",
    "--format",
    "json",
    "--time",
    cfg.walltime,
    "--mem",
    cfg.mem,
    "--timeout",
    String(cfg.timeout),
  ];

  if (options.gpuType) {
    args.push("--gpu-type", options.gpuType);
  }
  if (options.gpus !== undefined) {
    args.push("--gpus", String(options.gpus));
  }
  if (cfg.condaEnv) {
    args.push("--conda-env", cfg.condaEnv);
  }

  const child = spawn(getCliPath(), args, {
    stdio: ["pipe", "pipe", "pipe"],
  });

  const ready = new Promise<KoaJupyterOutput>((resolve, reject) => {
    let stdout = "";
    let stderr = "";

    child.stdout!.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
      // Try to parse complete JSON
      try {
        const data = JSON.parse(stdout);
        resolve(data as KoaJupyterOutput);
      } catch {
        // Not complete yet, keep buffering
      }
    });

    child.stderr!.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      stderr += text;
      if (options.onProgress) {
        // Strip ANSI codes, split on \r and \n, take last meaningful line
        const clean = text.replace(ANSI_RE, "");
        const lines = clean.split(/[\r\n]+/).filter((l) => l.trim());
        const last = lines[lines.length - 1]?.trim();
        if (last) {
          options.onProgress(last);
        }
      }
    });

    child.on("error", (err) => {
      if (err && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT") {
        reject(
          new Error(
            `KOA CLI not found at "${getCliPath()}". Install with: uv tool install koa`
          )
        );
      } else {
        reject(err);
      }
    });

    child.on("exit", (code) => {
      // If we haven't resolved yet, the process exited before producing JSON
      if (code !== null && code !== 0) {
        // Extract useful error from stderr
        const clean = stderr.replace(ANSI_RE, "").trim();
        if (clean.includes("did not start within")) {
          reject(new Error("Allocation timed out. Try a different GPU type or increase timeout."));
        } else if (clean.includes("Permission denied") || clean.includes("authentication")) {
          reject(new Error("SSH authentication failed. Run koa-auth first."));
        } else if (clean.includes("tunnel failed")) {
          reject(new Error("SSH tunnel failed. Check network connection."));
        } else {
          reject(new Error(clean || `koa exited with code ${code}`));
        }
      }
    });
  });

  return { child, ready };
}
