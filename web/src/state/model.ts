// ModelController — drives the model/API control panel () over the EXISTING endpoints
// (/model/status·load·switch·unload·params, /model/files , /health for per-module status). It holds
// ZERO authority: model-admin actions are gated SERVER-side (model:admin scope); this controller never
// bypasses — it surfaces the server's verdict (a 403 becomes a clear "requires model:admin" notice). All
// reads are malformed-safe (treated as "unknown", never a crash).

import type { ApiClient } from "../api/client";

export interface ModelStatus {
 state: string | null; // "unloaded" | "launching" | "ready" | "error" | null (unknown)
 loaded_file: string | null;
 serving: boolean;
}

export interface ModuleHealth {
 name: string;
 status: string; // "ok" | "degraded" | "down" | ...
}

export interface ModelState {
 status: ModelStatus | null;
 params: Record<string, number>;
 files: string[];
 modules: ModuleHealth[]; // per-module health, from /health.modules
 busy: boolean; // an admin action is in flight (single-flight)
 notice: string | null; // the last action outcome / error to surface (display-only)
 loadStartedAt: number | null; // ISSUE-002: wall-clock (ms) when the engine entered a loading state, else
 // null. Display-only — drives the indeterminate progress bar + elapsed timer;
 // never calls the server, never drives an action. Set on the transition INTO
 // loading, cleared on ready/error/unloaded.
}

const EMPTY: ModelState = {
 status: null, params: {}, files: [], modules: [], busy: false, notice: null, loadStartedAt: null,
};

// states that mean "the engine is coming up" — backend emits "loading"; "launching" kept defensively
// (the status-badge CSS references it) so the indicator is robust to either label.
function isLoadingState(state: string | null | undefined): boolean {
 return state === "loading" || state === "launching";
}

function asNumberRecord(v: unknown): Record<string, number> {
 const out: Record<string, number> = {};
 if (v && typeof v === "object") {
 for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
 if (typeof val === "number") out[k] = val;
 }
 }
 return out;
}

function asStringArray(v: unknown): string[] {
 return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

/** /health.modules is either { name: "ok" } or { name: { status: "ok", detail } } — normalize both. */
function parseModules(v: unknown): ModuleHealth[] {
 if (!v || typeof v !== "object") return [];
 const out: ModuleHealth[] = [];
 for (const [name, val] of Object.entries(v as Record<string, unknown>)) {
 if (typeof val === "string") out.push({ name, status: val });
 else if (val && typeof val === "object" && typeof (val as { status?: unknown }).status === "string")
 out.push({ name, status: (val as { status: string }).status });
 else out.push({ name, status: "unknown" });
 }
 return out.sort((a, b) => a.name.localeCompare(b.name));
}

// map an admin-action HTTP status to a human notice (server is authoritative; we only surface it).
function noticeFor(verb: string, status: number): string {
 if (status === 200) return `${verb}: ready`;
 if (status === 202) return `${verb}: launching…`;
 if (status === 400) return `${verb}: invalid model file`;
 if (status === 403) return `${verb}: requires model:admin scope`;
 if (status === 409) return `${verb}: model busy / no model loaded`;
 if (status === 503) return `${verb}: model stack not enabled`;
 return `${verb}: failed (${status})`;
}

export class ModelController {
 private readonly client: ApiClient;
 private state: ModelState = EMPTY;
 private readonly listeners = new Set<(s: ModelState) => void>();
 private readonly now: () => number;
 private readonly pollMs: number;
 private pollTimer: ReturnType<typeof setInterval> | null = null;

 // `now` is injectable so the load-start timestamp is deterministic in tests (defaults to wall-clock).
 // `pollMs` is how often /model/status is re-read WHILE loading (so the indicator clears when the engine
 // reaches ready — there is no server push); polling is read-only and stops as soon as loading ends.
 constructor(client: ApiClient, now: () => number = () => Date.now(), pollMs = 2500) {
 this.client = client;
 this.now = now;
 this.pollMs = pollMs;
 }

 get current(): ModelState {
 return this.state;
 }
 subscribe(fn: (s: ModelState) => void): () => void {
 this.listeners.add(fn);
 fn(this.state);
 return () => this.listeners.delete(fn);
 }
 private set(patch: Partial<ModelState>): void {
 this.state = { ...this.state, ...patch };
 for (const fn of this.listeners) fn(this.state);
 }

 /** Read the full picture: status, params, available GGUF files, per-module health. Safe on failure. */
 async refresh(): Promise<void> {
 const status = await this.safe(() => this.client.modelStatus(), null);
 const paramsRes = await this.safe(() => this.client.getParams(), null);
 const filesRes = await this.safe(() => this.client.listModelFiles(), null);
 const health = await this.safe(() => this.client.health(), null);
 const parsed = this.parseStatus(status);
 const loadStartedAt = this.deriveLoadStart(parsed);
 this.set({
 status: parsed,
 params: asNumberRecord((paramsRes as { params?: unknown } | null)?.params),
 files: asStringArray((filesRes as { files?: unknown } | null)?.files),
 modules: parseModules((health as { modules?: unknown } | null)?.modules),
 loadStartedAt,
 });
 this.syncPoll(loadStartedAt !== null);
 }

 // ISSUE-002: stamp the load-start time on the transition INTO a loading state (keep the existing stamp
 // while still loading), clear it otherwise. Pure display state — derived from the observed status only.
 private deriveLoadStart(status: ModelStatus | null): number | null {
 if (!isLoadingState(status?.state)) return null;
 return this.state.loadStartedAt ?? this.now();
 }

 // start a read-only status poll while loading; stop it the moment loading ends. The poll skips while an
 // admin action is in flight (act() does its own refresh) and is fully torn down by dispose().
 private syncPoll(loading: boolean): void {
 if (loading && this.pollTimer === null) {
 this.pollTimer = setInterval(() => {
 if (!this.state.busy) void this.refresh();
 }, this.pollMs);
 } else if (!loading && this.pollTimer !== null) {
 clearInterval(this.pollTimer);
 this.pollTimer = null;
 }
 }

 private parseStatus(v: unknown): ModelStatus | null {
 if (!v || typeof v !== "object") return null;
 const o = v as Record<string, unknown>;
 return {
 state: typeof o.state === "string" ? o.state : null,
 loaded_file: typeof o.loaded_file === "string" ? o.loaded_file : null,
 serving: o.serving === true,
 };
 }

 private async safe<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
 try {
 return await fn();
 } catch {
 return fallback;
 }
 }

 // -- admin actions (single-flight). The server enforces model:admin; we surface the verdict. -- //
 private async act(verb: string, fn: () => Promise<{ status: number }>): Promise<void> {
 if (this.state.busy) return; // single-flight guard
 this.set({ busy: true, notice: `${verb}…` });
 let status = 0;
 try {
 ({ status } = await fn());
 } catch {
 this.set({ busy: false, notice: `${verb}: request failed` });
 return;
 }
 this.set({ notice: noticeFor(verb, status) });
 await this.refresh(); // reflect the new server state
 this.set({ busy: false });
 }

 load(ggufFile?: string): Promise<void> {
 return this.act("load", () => this.client.modelLoad(ggufFile));
 }
 switchModel(ggufFile: string): Promise<void> {
 return this.act("switch", () => this.client.modelSwitch(ggufFile));
 }
 unload(): Promise<void> {
 return this.act("unload", () => this.client.modelUnload());
 }
 setParams(params: Record<string, number>): Promise<void> {
 return this.act("params", () => this.client.setParams(params));
 }

 dispose(): void {
 if (this.pollTimer !== null) {
 clearInterval(this.pollTimer);
 this.pollTimer = null;
 }
 this.listeners.clear();
 }
}
