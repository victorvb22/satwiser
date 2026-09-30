/**
 * React hook around the lab worker. Parameter changes are coalesced to one request per
 * animation frame, and answers older than the latest request are ignored, so dragging a
 * slider stays fluid whatever the computation time.
 */
import { useEffect, useRef, useState } from "react";

import type { LabWindow } from "../api/client";
import type { LabConfig, LabParams, LabResult } from "./runLab";

export function useLab(window: LabWindow | undefined, realDvMmS: number | null,
                       params: LabParams, config: LabConfig) {
  const worker = useRef<Worker | null>(null);
  const latest = useRef(0);
  const frame = useRef<number | null>(null);
  const [ready, setReady] = useState(false);
  const [realDv, setRealDv] = useState<number | null>(null);
  const [result, setResult] = useState<LabResult | null>(null);
  const [ms, setMs] = useState<number | null>(null);

  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "loaded") {
        setRealDv(msg.realDvMmS);
        setReady(true);
      } else if (msg.type === "result" && msg.id === latest.current) {
        setResult(msg.result);
        setMs(msg.ms);
      }
    };
    return () => w.terminate();
  }, []);

  useEffect(() => {
    if (!window || !worker.current) return;
    // Invalidate any answer still in flight for the previous window.
    latest.current += 1;
    setReady(false);
    setResult(null);
    setRealDv(null);
    worker.current.postMessage({ type: "load", window, realDvMmS, config });
    // config is derived from the robustness grid, stable for the page's lifetime.
  }, [window, realDvMmS]);

  useEffect(() => {
    if (!ready || !worker.current) return;
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      latest.current += 1;
      worker.current?.postMessage({ type: "run", id: latest.current, params });
    });
  }, [ready, params]);

  return { ready, realDv, result, ms };
}
