/** JSON output from `koa jupyter --format json`. */
export interface KoaJupyterOutput {
  job_id: string;
  node: string;
  local_port: number;
  remote_port: number;
  url: string;
  token: string;
  gpu_type: string | null;
  gpus: number;
}

/** Single GPU entry from `koa availability --format json`. */
export interface GpuInfo {
  gpu_type: string;
  vram_gb: number;
  free: number;
  pending: number;
  total: number;
  max_per_node: number;
}

/** JSON output from `koa availability --format json`. */
export interface AvailabilityOutput {
  gpus: GpuInfo[];
}

/** Tracked info for an active KOA session. */
export interface KoaSessionInfo {
  jobId: string;
  node: string;
  localPort: number;
  remotePort: number;
  url: string;
  token: string;
  gpuType: string | null;
  gpus: number;
  startedAt: Date;
}
