/// <reference lib="webworker" />
/**
 * Lab worker: keeps the prepared window in memory and answers evaluation requests.
 * Only the latest request matters; the page drops stale answers by id.
 */
import type { LabWindow } from "../api/client";
import { prepare, runLab, type LabParams, type PreparedWindow } from "./runLab";

export type WorkerRequest =
  | { type: "load"; window: LabWindow; realDvMmS: number | null }
  | { type: "run"; id: number; params: LabParams };

let prepared: PreparedWindow | null = null;

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === "load") {
    prepared = prepare(msg.window, msg.realDvMmS);
    self.postMessage({ type: "loaded", realDvMmS: prepared.realDvMmS });
  } else if (msg.type === "run" && prepared) {
    const t0 = performance.now();
    const result = runLab(prepared, msg.params);
    self.postMessage({ type: "result", id: msg.id, result, ms: performance.now() - t0 });
  }
};
