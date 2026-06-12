// RunController — the chat run lifecycle over the WS protocol (DA pure client).
// Sends `run_task`, tracks the single in-flight run, and resolves it on the terminal `task_result`
// or `error` event. /it now ALSO accumulates live `token` deltas (streamed answer)
// and `tool_call`/`tool_result` activity during a run — DISPLAY ONLY (zero authority; the server gate
// is unaffected). Approvals share the same socket; this controller IGNORES approval_* events
// for state but treats ANY server activity as liveness (so a long human-approval wait never trips the
// timeout — the timeout is a dead-connection safety net, not an approval deadline). React-decoupled.

import type { ApiClient, ConnState } from "../api/client";
import type { HistoryMessage, ServerEvent, TaskResultEvent } from "../api/types";

export interface ToolActivity {
 id: string;
 tool: string;
 outcome?: string; // set when the matching tool_result arrives
}

export type RunState =
 | { status: "idle" }
 | { status: "running"; task: string; streamedText: string; activity: ToolActivity[] }
 | { status: "done"; task: string; result: TaskResultEvent; streamedText: string; activity: ToolActivity[] }
 | { status: "error"; task: string; reason: string }
 | { status: "timeout"; task: string };

export class RunController {
 private readonly client: ApiClient;
 private readonly timeoutMs: number;
 private _state: RunState = { status: "idle" };
 private _streamed = "";
 private _activity: ToolActivity[] = [];
 private timer: ReturnType<typeof setTimeout> | null = null;
 private readonly offEvent: () => void;
 private readonly offState: () => void;
 private readonly listeners = new Set<(s: RunState) => void>();

 constructor(client: ApiClient, opts: { timeoutMs?: number } = {}) {
 this.client = client;
 this.timeoutMs = opts.timeoutMs ?? 120_000;
 this.offEvent = client.onEvent((e) => this.onEvent(e));
 this.offState = client.onState((s) => this.onState(s));
 }

 get state(): RunState {
 return this._state;
 }
 subscribe(fn: (s: RunState) => void): () => void {
 this.listeners.add(fn);
 return () => this.listeners.delete(fn);
 }
 private set(s: RunState): void {
 this._state = s;
 for (const fn of this.listeners) fn(s);
 }

 /** Start a run. Single-flight: ignored (returns false) if one is already running, or task is empty,
 * or the socket isn't connected (→ a clear error state, not a silent drop). */
 start(
 task: string,
 system?: string,
 attachments?: string[],
 history?: HistoryMessage[],
 ): boolean {
 if (this._state.status === "running") return false;
 const t = task.trim();
 if (!t) return false;
 if (!this.client.runTask(t, system, attachments, history)) {
 this.set({ status: "error", task: t, reason: "not connected" });
 return false;
 }
 this._streamed = "";
 this._activity = [];
 this.set({ status: "running", task: t, streamedText: "", activity: [] });
 this.armTimer();
 return true;
 }

 /** Clear a terminal state back to idle (the view does this after rendering the outcome). */
 reset(): void {
 if (this._state.status !== "running") this.set({ status: "idle" });
 }

 dispose(): void {
 this.clearTimer();
 this.offEvent();
 this.offState();
 this.listeners.clear();
 }

 private armTimer(): void {
 this.clearTimer();
 this.timer = setTimeout(() => {
 if (this._state.status === "running") this.set({ status: "timeout", task: this._state.task });
 }, this.timeoutMs);
 }
 private clearTimer(): void {
 if (this.timer) {
 clearTimeout(this.timer);
 this.timer = null;
 }
 }

 private onEvent(e: ServerEvent): void {
 if (this._state.status !== "running") return;
 const task = this._state.task;
 if (e.event === "task_result") {
 this.clearTimer();
 this.set({ status: "done", task, result: e,
 streamedText: this._streamed, activity: this._activity });
 } else if (e.event === "error") {
 this.clearTimer();
 this.set({ status: "error", task, reason: e.reason });
 } else if (e.event === "token") {
 // accumulate the streamed answer-in-progress (display only)
 this._streamed += e.delta;
 this.armTimer();
 this.set({ status: "running", task, streamedText: this._streamed, activity: this._activity });
 } else if (e.event === "tool_call") {
 // a tool the agent proposed (the gate still governs execution server-side)
 this._activity = [...this._activity, { id: e.id, tool: e.tool }];
 this.armTimer();
 this.set({ status: "running", task, streamedText: this._streamed, activity: this._activity });
 } else if (e.event === "tool_result") {
 // the outcome for a prior tool_call (match by id)
 this._activity = this._activity.map((a) =>
 a.id === e.id && a.outcome === undefined ? { ...a, outcome: e.outcome } : a,
 );
 this.armTimer();
 this.set({ status: "running", task, streamedText: this._streamed, activity: this._activity });
 } else {
 // approval_request / approval_timeout / approval_error / capabilities / echo → still alive
 this.armTimer();
 }
 }

 private onState(s: ConnState): void {
 if (
 this._state.status === "running" &&
 (s === "disconnected" || s === "error" || s === "auth_failed")
 ) {
 this.clearTimer();
 this.set({ status: "error", task: this._state.task, reason: "connection lost" });
 }
 }
}
