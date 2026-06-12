// ApprovalsController — the security-critical HITL surface. The agent loop pauses on
// every gated action; the server sends `approval_request {approval_id, tool, args, reason}`; this
// controller tracks each request and relays the operator's EXPLICIT decision as
// `{action:"approve"|"deny", approval_id}`.
//
// ZERO AUTHORITY (the crux): there is NO code path that decides on its own. `approve`/`deny` are sent
// ONLY from `decide()`, which is invoked only by an explicit user action — never by a timer, default,
// or heuristic. The controller has no allowlist and no auto-approve. The wire carries only the
// approval_id + verdict; the server binds/consumes the action it proposed, so a client cannot forge or
// substitute it. A decision is sent at most ONCE per request (only while `pending`); a late/duplicate/
// unknown-id decision is a no-op.

import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";

export type ApprovalStatus = "pending" | "expired" | "resolved" | "error";

export interface ApprovalItem {
 approval_id: string;
 tool: string;
 args: Record<string, unknown>;
 reason: string;
 status: ApprovalStatus;
 decision?: "approve" | "deny";
 error?: string;
}

export class ApprovalsController {
 private readonly client: ApiClient;
 private items: ApprovalItem[] = [];
 private readonly indexById = new Map<string, number>();
 private readonly offEvent: () => void;
 private readonly offState: () => void;
 private readonly listeners = new Set<(items: ApprovalItem[]) => void>();

 constructor(client: ApiClient) {
 this.client = client;
 this.offEvent = client.onEvent((e) => this.onEvent(e));
 this.offState = client.onState((s) => this.onState(s));
 }

 get list(): ApprovalItem[] {
 return this.items;
 }
 subscribe(fn: (items: ApprovalItem[]) => void): () => void {
 this.listeners.add(fn);
 return () => this.listeners.delete(fn);
 }
 private emit(): void {
 this.items = [...this.items]; // new ref so React re-renders
 for (const fn of this.listeners) fn(this.items);
 }
 private patch(id: string, next: Partial<ApprovalItem>): void {
 const idx = this.indexById.get(id);
 if (idx === undefined) return;
 this.items[idx] = { ...this.items[idx], ...next };
 this.emit();
 }

 /** Relay an EXPLICIT user decision. Sends at most once, only while `pending`; otherwise a no-op
 * (unknown / expired / already-resolved id → returns false, NO send). NEVER called automatically. */
 decide(approvalId: string, decision: "approve" | "deny"): boolean {
 const idx = this.indexById.get(approvalId);
 if (idx === undefined) return false; // unknown id → no-op (server is authoritative)
 const item = this.items[idx];
 if (item.status !== "pending") return false; // late/duplicate/expired/resolved → no-op, no send
 const sent = decision === "approve" ? this.client.approve(approvalId) : this.client.deny(approvalId);
 if (!sent) {
 this.patch(approvalId, { status: "error", error: "not connected" });
 return false;
 }
 this.patch(approvalId, { status: "resolved", decision });
 return true;
 }

 dispose(): void {
 this.offEvent();
 this.offState();
 this.listeners.clear();
 }

 private onEvent(e: ServerEvent): void {
 if (e.event === "approval_request") {
 if (this.indexById.has(e.approval_id)) return; // duplicate id → ignore (first wins)
 const item: ApprovalItem = {
 approval_id: e.approval_id,
 tool: typeof e.tool === "string" ? e.tool : "(unknown tool)",
 args: e.args && typeof e.args === "object" ? e.args : {},
 reason: typeof e.reason === "string" ? e.reason : "",
 status: "pending",
 };
 this.indexById.set(item.approval_id, this.items.length);
 this.items.push(item);
 this.emit();
 } else if (e.event === "approval_timeout") {
 // only a still-pending request expires; a resolved one is left as-is
 const idx = this.indexById.get(e.approval_id);
 if (idx !== undefined && this.items[idx].status === "pending") {
 this.patch(e.approval_id, { status: "expired" });
 }
 } else if (e.event === "approval_error") {
 // only a still-pending request flips to error; a resolved/expired one keeps its terminal state
 // (mirrors the approval_timeout guard — a late error can't overwrite a sent decision's display).
 if (typeof e.approval_id === "string") {
 const idx = this.indexById.get(e.approval_id);
 if (idx !== undefined && this.items[idx].status === "pending") {
 this.patch(e.approval_id, { status: "error", error: e.reason });
 }
 }
 // a null approval_id targets no specific prompt → ignored (nothing to act on)
 }
 }

 private onState(s: ConnState): void {
 // a dead socket can't deliver a decision → expire any still-pending prompt (no lingering action)
 if (s === "disconnected" || s === "error" || s === "auth_failed") {
 let changed = false;
 this.items = this.items.map((it) => {
 if (it.status === "pending") {
 changed = true;
 return { ...it, status: "expired" as const };
 }
 return it;
 });
 if (changed) for (const fn of this.listeners) fn(this.items);
 }
 }
}
