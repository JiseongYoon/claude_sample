// ApiClient — the ONE module that touches the core wire (the client analogue of the backend's
// `Guarded*` seam). Components never build URLs/frames directly; they use this. REST (/health,
// /model/*) + WS (/ws). Auth token is held IN MEMORY only (never localStorage — limits XSS exfil)
// and never logged. Server frames are untrusted: only KNOWN events are dispatched; unknown/malformed
// frames are ignored (never thrown into the UI). The UI has ZERO authority — this is transport only.

import { KNOWN_EVENTS, type ClientCommand, type HistoryMessage, type ServerEvent } from "./types";

export type ConnState =
 | "disconnected"
 | "connecting"
 | "connected"
 | "auth_failed"
 | "error";

export type AuthScheme = "api-key" | "bearer";

// Minimal WebSocket surface we depend on (injectable so tests use a mock — no real socket needed).
export interface WebSocketLike {
 send(data: string): void;
 close(code?: number, reason?: string): void;
 onopen: ((ev: unknown) => void) | null;
 onclose: ((ev: { code: number; reason?: string }) => void) | null;
 onmessage: ((ev: { data: unknown }) => void) | null;
 onerror: ((ev: unknown) => void) | null;
}
export type WsFactory = (url: string) => WebSocketLike;

export interface ApiClientOptions {
 baseUrl: string; // e.g. "http://localhost:8000" (no trailing slash needed)
 token: string;
 authScheme?: AuthScheme; // default "api-key"
 wsFactory?: WsFactory; // default: global WebSocket
 fetchFn?: typeof fetch; // default: global fetch
 maxBackoffMs?: number; // default 10000
 baseBackoffMs?: number; // default 500
 autoReconnect?: boolean; // default true
}

type EventListener = (ev: ServerEvent) => void;
type StateListener = (state: ConnState) => void;

// -- token bootstrap (): exchange an API key for a scoped JWT via /auth/token. Standalone
// (runs BEFORE a connected client). The API key is used only for this request — the caller discards it. --
export interface MintArgs {
 baseUrl: string;
 apiKey: string;
 scopes: string[];
 subject?: string;
 expiresMinutes?: number;
 fetchFn?: typeof fetch;
}
export interface MintResult {
 access_token: string;
 scopes: string[]; // the scopes the SERVER granted (authoritative)
 expires_in: number;
}

const WS_CLOSE_POLICY_VIOLATION = 1008; // the gateway closes auth failures with 1008
const MAX_HANDSHAKE_FAILURES = 6; // consecutive never-opened closes → settle terminal `error` (no infinite loop)

function defaultWsFactory(url: string): WebSocketLike {
 // eslint-disable-next-line @typescript-eslint/no-explicit-any
 return new WebSocket(url) as unknown as WebSocketLike;
}

export class ApiClient {
 private readonly baseUrl: string;
 private readonly token: string; // in-memory only; never persisted/logged
 private readonly authScheme: AuthScheme;
 private readonly wsFactory: WsFactory;
 private readonly fetchFn: typeof fetch;
 private readonly maxBackoffMs: number;
 private readonly baseBackoffMs: number;
 private readonly autoReconnect: boolean;

 private ws: WebSocketLike | null = null;
 private _state: ConnState = "disconnected";
 private attempts = 0;
 private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
 private closedByUser = false;
 private everOpened = false; // did THIS socket complete its handshake (fire onopen)?
 private handshakeFailures = 0; // consecutive closes that NEVER opened (rejected/unreachable)

 private readonly eventListeners = new Set<EventListener>();
 private readonly stateListeners = new Set<StateListener>();
 capabilities: Record<string, unknown> | null = null;

 constructor(opts: ApiClientOptions) {
 this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
 this.token = opts.token;
 this.authScheme = opts.authScheme ?? "api-key";
 this.wsFactory = opts.wsFactory ?? defaultWsFactory;
 this.fetchFn = opts.fetchFn ?? ((...a) => fetch(...a));
 this.maxBackoffMs = opts.maxBackoffMs ?? 10_000;
 this.baseBackoffMs = opts.baseBackoffMs ?? 500;
 this.autoReconnect = opts.autoReconnect ?? true;
 }

 // -- subscriptions ------------------------------------------------------- //
 get state(): ConnState {
 return this._state;
 }
 onEvent(fn: EventListener): () => void {
 this.eventListeners.add(fn);
 return () => this.eventListeners.delete(fn);
 }
 onState(fn: StateListener): () => void {
 this.stateListeners.add(fn);
 return () => this.stateListeners.delete(fn);
 }
 private setState(s: ConnState): void {
 this._state = s;
 for (const fn of this.stateListeners) fn(s);
 }

 // -- WS lifecycle -------------------------------------------------------- //
 private wsUrl(): string {
 // browsers can't set WS headers → the token rides as ?token= (the gateway accepts this).
 const base = this.baseUrl.replace(/^http/, "ws");
 return `${base}/ws?token=${encodeURIComponent(this.token)}`;
 }

 connect(): void {
 if (this._state === "connecting" || this._state === "connected") return;
 this.closedByUser = false;
 this.setState("connecting");
 let sock: WebSocketLike;
 try {
 sock = this.wsFactory(this.wsUrl());
 } catch {
 this.setState("error");
 this.scheduleReconnect();
 return;
 }
 this.ws = sock;
 this.everOpened = false;
 sock.onopen = () => {
 this.attempts = 0;
 this.everOpened = true;
 this.handshakeFailures = 0; // a completed handshake clears the failure streak
 this.setState("connected");
 };
 sock.onmessage = (ev) => this.handleMessage(ev.data);
 sock.onerror = () => {
 // an error before/after open; the close handler decides reconnect vs auth_failed
 if (this._state !== "auth_failed") this.setState("error");
 };
 sock.onclose = (ev) => {
 this.ws = null;
 if (ev && ev.code === WS_CLOSE_POLICY_VIOLATION) {
 // auth rejected by the gateway (1008 frame; the gateway accepts-then-closes so the browser
 // actually receives this code) → do NOT reconnect (would loop on a bad token)
 this.setState("auth_failed");
 return;
 }
 if (this.closedByUser) {
 this.setState("disconnected");
 return;
 }
 if (!this.everOpened) {
 // the handshake never completed (rejected upgrade → 1006, or server unreachable). A browser
 // can't see the HTTP status, so we can't tell "bad token" from "down" — bound the retries so a
 // persistent failure settles in a terminal `error` state instead of toggling forever.
 this.handshakeFailures += 1;
 if (this.handshakeFailures >= MAX_HANDSHAKE_FAILURES) {
 this.setState("error"); // terminal — stop reconnecting; the operator can re-connect manually
 return;
 }
 this.setState("disconnected");
 this.scheduleReconnect();
 return;
 }
 // the socket had opened and then dropped → a transient disconnect; reconnect with backoff.
 this.setState(this._state === "auth_failed" ? "auth_failed" : "disconnected");
 this.scheduleReconnect();
 };
 }

 disconnect(): void {
 this.closedByUser = true;
 if (this.reconnectTimer) {
 clearTimeout(this.reconnectTimer);
 this.reconnectTimer = null;
 }
 const sock = this.ws;
 this.ws = null;
 if (sock) {
 // detach handlers BEFORE closing so a stale onopen/onclose can never fire a spurious
 // state change after an intentional disconnect (robustness; real WS won't open post-close).
 sock.onopen = sock.onmessage = sock.onerror = sock.onclose = null;
 sock.close();
 }
 this.setState("disconnected");
 }

 private scheduleReconnect(): void {
 if (!this.autoReconnect || this.closedByUser) return;
 this.attempts += 1;
 const backoff = Math.min(this.baseBackoffMs * 2 ** (this.attempts - 1), this.maxBackoffMs);
 if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
 this.reconnectTimer = setTimeout(() => {
 if (!this.closedByUser) this.connect();
 }, backoff);
 }

 // -- inbound: parse-known / ignore-unknown ------------------------------- //
 private handleMessage(data: unknown): void {
 if (typeof data !== "string") return; // binary / odd → ignore
 let parsed: unknown;
 try {
 parsed = JSON.parse(data);
 } catch {
 return; // non-JSON → ignore (never throw into the UI)
 }
 if (
 typeof parsed !== "object" ||
 parsed === null ||
 typeof (parsed as { event?: unknown }).event !== "string" ||
 !KNOWN_EVENTS.has((parsed as { event: string }).event as ServerEvent["event"])
 ) {
 return; // unknown/malformed event → ignore
 }
 const ev = parsed as ServerEvent;
 if (ev.event === "capabilities") this.capabilities = ev as Record<string, unknown>;
 for (const fn of this.eventListeners) fn(ev);
 }

 // -- outbound commands --------------------------------------------------- //
 private send(cmd: ClientCommand): boolean {
 if (!this.ws || this._state !== "connected") return false;
 this.ws.send(JSON.stringify(cmd));
 return true;
 }
 runTask(
 task: string,
 system?: string,
 attachments?: string[],
 history?: HistoryMessage[],
 ): boolean {
 const sys = system?.trim();
 const cmd: ClientCommand = { action: "run_task", task };
 if (sys) cmd.system = sys;
 if (attachments && attachments.length) cmd.attachments = attachments;
 // replay prior turns so the agent remembers context (server bounds + validates).
 if (history && history.length) cmd.history = history;
 return this.send(cmd);
 }
 approve(approvalId: string): boolean {
 return this.send({ action: "approve", approval_id: approvalId });
 }
 deny(approvalId: string): boolean {
 return this.send({ action: "deny", approval_id: approvalId });
 }

 // -- REST ---------------------------------------------------------------- //
 private authHeaders(): Record<string, string> {
 return this.authScheme === "bearer"
 ? { Authorization: `Bearer ${this.token}` }
 : { "x-api-key": this.token };
 }
 private async getJson(path: string): Promise<unknown> {
 const res = await this.fetchFn(`${this.baseUrl}${path}`, {
 headers: { ...this.authHeaders() },
 });
 if (!res.ok) {
 throw new Error(`request failed: ${res.status}`); // status only — no body/token in the message
 }
 return res.json();
 }
 /** Like getJson but for mutating model-admin calls: returns the status + parsed body instead of
 * throwing, so the controller can branch on 200/202/400/403/409/503 (the server is authoritative —
 * the client surfaces the verdict, never bypasses it). The body is parsed best-effort (null on fail). */
 private async reqJson(
 method: string,
 path: string,
 body?: unknown,
 ): Promise<{ status: number; ok: boolean; data: unknown }> {
 const headers: Record<string, string> = { ...this.authHeaders() };
 const init: RequestInit = { method, headers };
 if (body !== undefined) {
 headers["Content-Type"] = "application/json";
 init.body = JSON.stringify(body);
 }
 const res = await this.fetchFn(`${this.baseUrl}${path}`, init);
 let data: unknown = null;
 try {
 data = await res.json();
 } catch {
 data = null; // a body-less / non-JSON response is fine — status carries the meaning
 }
 return { status: res.status, ok: res.ok, data };
 }
 health(): Promise<unknown> {
 return this.getJson("/health");
 }
 modelStatus(): Promise<unknown> {
 return this.getJson("/model/status");
 }
 capabilitiesRest(): Promise<unknown> {
 return this.getJson("/capabilities"); // { available: string[], all: string[] }
 }

 // -- read-only discovery () — all GET, no authority, server returns non-secret data -- //
 listModelFiles(): Promise<unknown> {
 return this.getJson("/model/files"); // { files: string[] }
 }
 listConnectors(): Promise<unknown> {
 return this.getJson("/storage/connectors"); // { connectors: NonSecretConnector[], health }
 }
 listMcpServers(): Promise<unknown> {
 return this.getJson("/mcp/servers"); // { servers: NonSecretMcpServer[], health }
 }
 listTools(): Promise<unknown> {
 return this.getJson("/tools"); // { tools: {name, tier, capability}[] }
 }

 // -- model/API control () — admin actions are model:admin-gated SERVER-side; the client
 // never bypasses, it surfaces the status (403 = insufficient scope). status/params/files are reads. -- //
 modelLoad(ggufFile?: string) {
 return this.reqJson("POST", "/model/load", ggufFile ? { gguf_file: ggufFile } : {});
 }
 modelSwitch(ggufFile: string) {
 return this.reqJson("POST", "/model/switch", { gguf_file: ggufFile });
 }
 modelUnload() {
 return this.reqJson("POST", "/model/unload");
 }
 getParams(): Promise<unknown> {
 return this.getJson("/model/params"); // { params: {...} }
 }
 setParams(params: Record<string, number>) {
 return this.reqJson("POST", "/model/params", params);
 }
 getModule(name: string): Promise<unknown> {
 return this.getJson(`/modules/${encodeURIComponent(name)}`); // { name, status, available, detail? }
 }

 // -- direct toolless chat () — REST /chat, multi-turn, NO tool loop / NO gate (distinct
 // from the gated agent run_task path). Returns {status,ok,data} so the caller surfaces 503/502. -- //
 chat(
 messages: Array<{ role: string; content: string }>,
 params?: Record<string, number>,
 attachments?: string[],
 ) {
 const body: Record<string, unknown> = { messages, ...(params ?? {}) };
 if (attachments && attachments.length) body.attachments = attachments;
 return this.reqJson("POST", "/chat", body);
 }

 /** Upload a file to the gated ingest endpoint (). Zero authority: the UI only POSTs the
 * bytes + relays the opaque id the server returns — all validation/containment is server-side. The
 * multipart Content-Type/boundary is set by the browser (do NOT set it here). Errors are status-only
 * (no filename/body echoed into the message): a 403 = the token lacks the `ingest` scope. */
 async uploadFile(file: File): Promise<{ id: string; filename: string; size: number; ext: string }> {
 const form = new FormData();
 form.append("file", file);
 const res = await this.fetchFn(`${this.baseUrl}/ingest`, {
 method: "POST",
 headers: { ...this.authHeaders() }, // no Content-Type — the browser sets the multipart boundary
 body: form,
 });
 if (!res.ok) throw new Error(`upload failed: ${res.status}`); // status only — no filename/body
 const data = (await res.json()) as {
 id?: unknown;
 filename?: unknown;
 size?: unknown;
 ext?: unknown;
 };
 if (!data || typeof data.id !== "string") throw new Error("upload failed: malformed response");
 return {
 id: data.id,
 filename: typeof data.filename === "string" ? data.filename : file.name,
 size: typeof data.size === "number" ? data.size : 0,
 ext: typeof data.ext === "string" ? data.ext : "",
 };
 }

 /** Mint a scoped JWT from an API key (). Standalone (no connected client needed). The API
 * key goes ONLY in this request's `x-api-key` header — the caller must discard it after. Errors are
 * status-only (the key is never echoed into a thrown message). */
 static async mintToken(args: MintArgs): Promise<MintResult> {
 const fetchFn = args.fetchFn ?? fetch;
 const body: Record<string, unknown> = { scopes: args.scopes };
 if (args.subject) body.subject = args.subject;
 if (args.expiresMinutes !== undefined) body.expires_minutes = args.expiresMinutes;
 const res = await fetchFn(`${args.baseUrl}/auth/token`, {
 method: "POST",
 headers: { "x-api-key": args.apiKey, "Content-Type": "application/json" },
 body: JSON.stringify(body),
 });
 if (!res.ok) throw new Error(`mint failed: ${res.status}`); // status only — no key/body
 const data = (await res.json()) as { access_token?: unknown; scopes?: unknown; expires_in?: unknown };
 if (!data || typeof data.access_token !== "string") throw new Error("mint failed: malformed response");
 return {
 access_token: data.access_token,
 scopes: Array.isArray(data.scopes) ? data.scopes.filter((s): s is string => typeof s === "string") : args.scopes,
 expires_in: typeof data.expires_in === "number" ? data.expires_in : 0,
 };
 }
}
