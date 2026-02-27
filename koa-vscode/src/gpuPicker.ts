import * as vscode from "vscode";
import { getAvailability } from "./koaCli";

export interface GpuPickResult {
  gpuType: string | null; // null = CPU only
  gpus: number;
}

/**
 * Show a QuickPick populated from `koa availability --format json`.
 * Returns undefined if user cancels.
 */
export async function pickGpuType(): Promise<GpuPickResult | undefined> {
  // Show a loading picker while fetching availability
  const pick = vscode.window.createQuickPick();
  pick.title = "KOA: Select GPU Type";
  pick.placeholder = "Loading cluster availability...";
  pick.busy = true;
  pick.show();

  try {
    const data = await getAvailability();

    const items: (vscode.QuickPickItem & { result: GpuPickResult })[] = [];

    // GPU options sorted by contention (koa already sorts by priority)
    for (const gpu of data.gpus) {
      if (gpu.total === 0) {
        continue;
      }
      const freeTag =
        gpu.free > 0
          ? `$(pass) ${gpu.free} free`
          : `$(circle-slash) 0 free`;
      const waitTag =
        gpu.pending > 0 ? `, ${gpu.pending} waiting` : "";

      items.push({
        label: `$(chip) ${gpu.gpu_type}`,
        description: `${gpu.vram_gb}GB VRAM`,
        detail: `${freeTag}${waitTag} | ${gpu.total} total, max ${gpu.max_per_node}/node`,
        result: { gpuType: gpu.gpu_type, gpus: 1 },
      });
    }

    // CPU-only option at the end
    items.push({
      label: "$(terminal) CPU only",
      description: "No GPU",
      detail: "Request a CPU-only node",
      result: { gpuType: null, gpus: 0 },
    });

    pick.items = items;
    pick.busy = false;
    pick.placeholder = "Select GPU type (sorted by availability)";

    return new Promise<GpuPickResult | undefined>((resolve) => {
      pick.onDidAccept(() => {
        const selected = pick.selectedItems[0] as
          | (vscode.QuickPickItem & { result: GpuPickResult })
          | undefined;
        pick.dispose();
        resolve(selected?.result);
      });
      pick.onDidHide(() => {
        pick.dispose();
        resolve(undefined);
      });
    });
  } catch (err) {
    pick.dispose();
    vscode.window.showErrorMessage(
      `KOA: Failed to fetch availability: ${err}`
    );
    return undefined;
  }
}
