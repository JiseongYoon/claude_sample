// CapabilitiesController — derives which capabilities are available (by module health) from the WS
// `capabilities` event (live `available` list) + a REST `/capabilities` ({available, all}) + `/health`
// (overall status). The UI uses this to show availability and disable affordances for absent/degraded
// capabilities (so the user isn't offered a call the core will refuse). Read-only display — it gates
// NOTHING (the server is authoritative); malformed payloads are treated as "unknown → absent", no crash.

import type { ApiClient } from "../api/client";
import type { ServerEvent } from "../api/types";

export interface CapabilityView {
 name: string;
 available: boolean;
}

export interface CapabilitiesState {
 caps: CapabilityView[];
 overall: string | null; // /health status: "ok" | "degraded" | "down" | null (unknown)
}

function asStringArray(v: unknown): string[] {
 return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

export class CapabilitiesController {
 private readonly client: ApiClient;
 private state: CapabilitiesState = { caps: [], overall: null };
 private knownAll: string[] = []; // the full capability set (from REST /capabilities.all)
 private readonly offEvent: () => void;
 private readonly listeners = new Set<(s: CapabilitiesState) => void>();

 constructor(client: ApiClient) {
 this.client = client;
 this.offEvent = client.onEvent((e) => this.onEvent(e));
 }

 get current(): CapabilitiesState {
 return this.state;
 }
 subscribe(fn: (s: CapabilitiesState) => void): () => void {
 this.listeners.add(fn);
 return () => this.listeners.delete(fn);
 }
 private set(next: CapabilitiesState): void {
 this.state = next;
 for (const fn of this.listeners) fn(this.state);
 }

 /** Build the cap list from the full set (`all`) with availability from `available`. If `all` is
 * unknown, fall back to the available list itself (those are the only caps we know about). */
 private rebuild(available: string[], overall: string | null): void {
 const all = this.knownAll.length ? this.knownAll : available;
 const set = new Set(available);
 const caps = [...new Set(all)].sort().map((name) => ({ name, available: set.has(name) }));
 this.set({ caps, overall });
 }

 private onEvent(e: ServerEvent): void {
 if (e.event === "capabilities") {
 // the WS handshake/refresh carries the live `available` list
 this.rebuild(asStringArray((e as { available?: unknown }).available), this.state.overall);
 }
 }

 /** Fetch the full picture from REST (available + all + overall health). Safe on failure. */
 async refresh(): Promise<void> {
 let available: string[] = this.state.caps.filter((c) => c.available).map((c) => c.name);
 let overall = this.state.overall;
 try {
 const caps = (await this.client.capabilitiesRest()) as { available?: unknown; all?: unknown };
 if (caps && typeof caps === "object") {
 this.knownAll = asStringArray(caps.all);
 available = asStringArray(caps.available);
 }
 } catch {
 /* keep WS-derived availability */
 }
 try {
 const health = (await this.client.health()) as { status?: unknown };
 if (health && typeof health === "object" && typeof health.status === "string") {
 overall = health.status;
 }
 } catch {
 /* keep prior overall */
 }
 this.rebuild(available, overall);
 }

 dispose(): void {
 this.offEvent();
 this.listeners.clear();
 }
}
