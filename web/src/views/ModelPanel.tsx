// ModelPanel view () — control the serving model + view params + a per-module health
// dashboard, over the EXISTING endpoints (+ /model/files for the picker). The UI has ZERO
// authority: model-admin actions are gated SERVER-side. When the caller knows the credential lacks
// model:admin (`canAdmin === false`) the admin controls are DISABLED with a clear message — but this is
// UX only; the server is authoritative and a forced call still returns 403 (surfaced, never bypassed).
// When `canAdmin` is undefined (scope unknown — e.g. a pasted token), controls stay enabled and a 403
// is surfaced. Server-provided text is rendered as escaped React children (no injection).
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import { ModelController, type ModelState } from "../state/model";
import styles from "./ModelPanel.module.css";

const EMPTY: ModelState = {
 status: null, params: {}, files: [], modules: [], busy: false, notice: null, loadStartedAt: null,
};
const PARAM_KEYS = ["temperature", "top_p", "top_k", "max_tokens"] as const;

// mm:ss from a millisecond elapsed duration (display-only).
function fmtElapsed(ms: number): string {
 const total = Math.max(0, Math.floor(ms / 1000));
 const m = Math.floor(total / 60);
 const s = total % 60;
 return `${m}:${s.toString().padStart(2, "0")}`;
}

export function ModelPanel({ client, canAdmin }: { client: ApiClient; canAdmin?: boolean }) {
 const [state, setState] = useState<ModelState>(EMPTY);
 const [file, setFile] = useState("");
 const [params, setParams] = useState<Record<string, string>>({});
 const ctrlRef = useRef<ModelController | null>(null);

 useEffect(() => {
 const ctrl = new ModelController(client);
 ctrlRef.current = ctrl;
 const off = ctrl.subscribe(setState);
 void ctrl.refresh();
 return () => {
 off();
 ctrl.dispose();
 ctrlRef.current = null;
 };
 }, [client]);

 const adminLocked = canAdmin === false;
 const adminDisabled = adminLocked || state.busy;

 // ISSUE-002: while the engine is coming up, show an indeterminate bar + a ticking elapsed counter. The
 // controller stamps `loadStartedAt`; this effect ticks the display every second and clears itself when
 // loading ends or the component unmounts (no leaked interval, no stale elapsed). Display-only.
 const loading = state.status?.state === "loading" || state.status?.state === "launching";
 const loadStartedAt = state.loadStartedAt;
 const [elapsedMs, setElapsedMs] = useState(0);
 useEffect(() => {
 if (!loading || loadStartedAt == null) {
 setElapsedMs(0);
 return;
 }
 const tick = () => setElapsedMs(Math.max(0, Date.now() - loadStartedAt));
 tick();
 const id = setInterval(tick, 1000);
 return () => clearInterval(id);
 }, [loading, loadStartedAt]);

 function applyParams() {
 const out: Record<string, number> = {};
 for (const k of PARAM_KEYS) {
 const raw = params[k];
 if (raw !== undefined && raw.trim() !== "" && Number.isFinite(Number(raw))) out[k] = Number(raw);
 }
 if (Object.keys(out).length > 0) ctrlRef.current?.setParams(out);
 }

 const st = state.status;
 return (
 <section aria-label="model-panel" className={styles.panel}>
 <h2 className={styles.heading}>Model</h2>

 <div className={styles.statusRow}>
 state:
 <span className={styles.stateBadge} data-state={st?.state ?? "unknown"} data-testid="model-state">
 {st?.state ?? "unknown"}
 </span>
 {st?.loaded_file && <span className={styles.loaded}>{st.loaded_file}</span>}
 {st?.serving && <span data-testid="model-serving">· serving</span>}
 </div>

 {loading && (
 <div className={styles.loadProgress} data-testid="model-load-progress">
 {/* indeterminate (no aria-valuenow) — there is no true % signal from the backend yet (ISSUE-002
 part 2, deferred); this shows the load is alive + how long it's been running. */}
 <div className={styles.loadBar} role="progressbar" aria-label="model loading" aria-busy="true" />
 <div className={styles.loadMeta}>
 <span data-testid="model-load-elapsed">{fmtElapsed(elapsedMs)}</span>
 <span className={styles.loadHint}>
 large models can take several minutes; loading from the network mount is slower
 </span>
 </div>
 </div>
 )}

 {adminLocked && (
 <p className={styles.adminLocked} data-testid="admin-locked">
 model administration requires a <code>model:admin</code>-scoped token — controls disabled.
 </p>
 )}

 <div className={styles.group}>
 <span className={styles.groupLabel}>load / switch</span>
 <div className={styles.row}>
 <select
 aria-label="model-file"
 value={file}
 onChange={(e) => setFile(e.target.value)}
 disabled={adminDisabled}
 >
 <option value="">— select a .gguf —</option>
 {state.files.map((f) => (
 <option key={f} value={f}>
 {f}
 </option>
 ))}
 </select>
 <button type="button" disabled={adminDisabled} onClick={() => ctrlRef.current?.load(file || undefined)}>
 Load
 </button>
 <button type="button" disabled={adminDisabled || !file} onClick={() => file && ctrlRef.current?.switchModel(file)}>
 Switch
 </button>
 <button type="button" disabled={adminDisabled} onClick={() => ctrlRef.current?.unload()}>
 Unload
 </button>
 </div>
 </div>

 <div className={styles.group}>
 <span className={styles.groupLabel}>params</span>
 <div className={styles.params}>
 {PARAM_KEYS.map((k) => (
 <label key={k} className={styles.field}>
 {k} {state.params[k] !== undefined && <small>(now {state.params[k]})</small>}
 <input
 aria-label={k}
 inputMode="decimal"
 value={params[k] ?? ""}
 onChange={(e) => setParams((p) => ({ ...p, [k]: e.target.value }))}
 disabled={adminDisabled}
 />
 </label>
 ))}
 </div>
 <div className={styles.row}>
 <button type="button" disabled={adminDisabled} onClick={applyParams}>
 Apply params
 </button>
 </div>
 </div>

 {state.notice && (
 <p className={styles.notice} data-testid="model-notice">
 {state.notice}
 </p>
 )}

 <div className={styles.group}>
 <span className={styles.groupLabel}>module health</span>
 <ul className={styles.modules}>
 {state.modules.map((m) => (
 <li key={m.name} className={styles.module} data-testid="module-health" data-module={m.name} data-status={m.status}>
 <span>{m.name}</span>
 <span className={styles.moduleStatus} data-status={m.status}>
 {m.status}
 </span>
 </li>
 ))}
 </ul>
 </div>
 </section>
 );
}
