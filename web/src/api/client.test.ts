import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient, type ConnState, type WebSocketLike } from "./client";
import type { ServerEvent } from "./types";

// A controllable mock WebSocket (no real socket / network). Tests drive open/message/close.
class MockWS implements WebSocketLike {
 static instances: MockWS[] = [];
 url: string;
 sent: string[] = [];
 closedWith: number | null = null;
 onopen: ((ev: unknown) => void) | null = null;
 onclose: ((ev: { code: number; reason?: string }) => void) | null = null;
 onmessage: ((ev: { data: unknown }) => void) | null = null;
 onerror: ((ev: unknown) => void) | null = null;

 constructor(url: string) {
 this.url = url;
 MockWS.instances.push(this);
 }
 send(data: string): void {
 this.sent.push(data);
 }
 close(code?: number): void {
 this.closedWith = code ?? 1000;
 this.onclose?.({ code: code ?? 1000 });
 }
 // test helpers
 fireOpen(): void {
 this.onopen?.({});
 }
 fireMessage(data: unknown): void {
 this.onmessage?.({ data });
 }
 fireServerClose(code: number): void {
 this.onclose?.({ code });
 }
 static last(): MockWS {
 return MockWS.instances[MockWS.instances.length - 1];
 }
 static reset(): void {
 MockWS.instances = [];
 }
}

function makeClient(over: Partial<ConstructorParameters<typeof ApiClient>[0]> = {}) {
 const states: ConnState[] = [];
 const events: ServerEvent[] = [];
 const client = new ApiClient({
 baseUrl: "http://core.local:8000",
 token: "SECRET-TOKEN-123",
 wsFactory: (url) => new MockWS(url),
 baseBackoffMs: 100,
 maxBackoffMs: 1000,
 ...over,
 });
 client.onState((s) => states.push(s));
 client.onEvent((e) => events.push(e));
 return { client, states, events };
}

beforeEach(() => MockWS.reset());
afterEach(() => vi.useRealTimers());

// --------------------------------------------------------------------------- //
// NORMAL
// --------------------------------------------------------------------------- //
describe("ApiClient — connect / handshake", () => {
 it("connecting → connected and exposes capabilities", () => {
 const { client, states, events } = makeClient();
 client.connect();
 expect(client.state).toBe("connecting");
 MockWS.last().fireOpen();
 expect(client.state).toBe("connected");
 MockWS.last().fireMessage(JSON.stringify({ event: "capabilities", agent: true, exec: false }));
 expect(states).toEqual(["connecting", "connected"]);
 expect(events[0]).toMatchObject({ event: "capabilities", agent: true });
 expect(client.capabilities).toMatchObject({ agent: true, exec: false });
 });

 it("WS url carries the token via ?token= and uses ws:// scheme", () => {
 const { client } = makeClient();
 client.connect();
 expect(MockWS.last().url).toBe("ws://core.local:8000/ws?token=SECRET-TOKEN-123");
 });

 it("sends run_task / approve / deny only when connected; binds approval_id", () => {
 const { client } = makeClient();
 expect(client.runTask("hello")).toBe(false); // not connected yet
 client.connect();
 MockWS.last().fireOpen();
 expect(client.runTask("hello")).toBe(true);
 expect(client.approve("ap-1")).toBe(true);
 expect(client.deny("ap-2")).toBe(true);
 expect(MockWS.last().sent.map((s) => JSON.parse(s))).toEqual([
 { action: "run_task", task: "hello" },
 { action: "approve", approval_id: "ap-1" },
 { action: "deny", approval_id: "ap-2" },
 ]);
 });

 it("health() returns parsed JSON via an authed REST call", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({ ok: true, status: 200, json: async () => ({ status: "ok", modules: 9 }) }) as unknown as Response,
 );
 const { client } = makeClient({ fetchFn });
 const out = await client.health();
 expect(out).toEqual({ status: "ok", modules: 9 });
 const call = fetchFn.mock.calls[0];
 expect(call[0]).toBe("http://core.local:8000/health");
 expect(call[1]?.headers).toMatchObject({ "x-api-key": "SECRET-TOKEN-123" });
 });

 it("bearer scheme sets Authorization header", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({ ok: true, status: 200, json: async () => ({}) }) as unknown as Response,
 );
 const { client } = makeClient({ authScheme: "bearer", fetchFn });
 await client.modelStatus();
 expect(fetchFn.mock.calls[0][1]?.headers).toMatchObject({
 Authorization: "Bearer SECRET-TOKEN-123",
 });
 });

 it("mintToken POSTs /auth/token with the API key, returns the granted scopes", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({
 ok: true,
 status: 200,
 json: async () => ({ access_token: "JWT-OK", scopes: ["read", "invoke"], expires_in: 3600 }),
 }) as unknown as Response,
 );
 const out = await ApiClient.mintToken({
 baseUrl: "http://core.local:8000",
 apiKey: "APIKEY-SENTINEL",
 scopes: ["read", "invoke", "model:admin"],
 fetchFn,
 });
 expect(out).toEqual({ access_token: "JWT-OK", scopes: ["read", "invoke"], expires_in: 3600 });
 const [url, init] = fetchFn.mock.calls[0];
 expect(url).toBe("http://core.local:8000/auth/token");
 expect(init?.method).toBe("POST");
 expect(init?.headers).toMatchObject({ "x-api-key": "APIKEY-SENTINEL" });
 expect(JSON.parse(String(init?.body))).toMatchObject({ scopes: ["read", "invoke", "model:admin"] });
 });

 it("mintToken throws a status-only error (no API key) on a failed mint", async () => {
 const fetchFn = vi.fn(async () => ({ ok: false, status: 403, json: async () => ({}) }) as unknown as Response);
 const p = ApiClient.mintToken({ baseUrl: "http://core.local:8000", apiKey: "APIKEY-SENTINEL", scopes: ["read"], fetchFn });
 await expect(p).rejects.toThrow(/403/);
 await expect(
 ApiClient.mintToken({ baseUrl: "http://core.local:8000", apiKey: "APIKEY-SENTINEL", scopes: ["read"], fetchFn }),
 ).rejects.not.toThrow(/APIKEY-SENTINEL/);
 });

 it("mintToken rejects a malformed response (no access_token)", async () => {
 const fetchFn = vi.fn(
 async () => ({ ok: true, status: 200, json: async () => ({ nope: true }) }) as unknown as Response,
 );
 await expect(
 ApiClient.mintToken({ baseUrl: "http://core.local:8000", apiKey: "K", scopes: ["read"], fetchFn }),
 ).rejects.toThrow(/malformed/);
 });

 it("discovery methods GET the right paths with auth + return parsed JSON", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({ ok: true, status: 200, json: async () => ({ ok: 1 }) }) as unknown as Response,
 );
 const { client } = makeClient({ fetchFn });
 const cases: [Promise<unknown>, string][] = [
 [client.listModelFiles(), "http://core.local:8000/model/files"],
 [client.listConnectors(), "http://core.local:8000/storage/connectors"],
 [client.listMcpServers(), "http://core.local:8000/mcp/servers"],
 [client.listTools(), "http://core.local:8000/tools"],
 ];
 for (const [p] of cases) expect(await p).toEqual({ ok: 1 });
 const urls = fetchFn.mock.calls.map((c) => c[0]);
 for (const [, url] of cases) expect(urls).toContain(url);
 for (const call of fetchFn.mock.calls)
 expect(call[1]?.headers).toMatchObject({ "x-api-key": "SECRET-TOKEN-123" });
 });
});

// --------------------------------------------------------------------------- //
// ERROR / SECURITY
// --------------------------------------------------------------------------- //
describe("ApiClient — error / security", () => {
 it("close 1008 → auth_failed and does NOT reconnect (no loop on a bad token)", () => {
 vi.useFakeTimers();
 const { client, states } = makeClient();
 client.connect();
 MockWS.last().fireServerClose(1008);
 expect(client.state).toBe("auth_failed");
 vi.advanceTimersByTime(5000);
 expect(MockWS.instances.length).toBe(1); // no reconnect attempt
 expect(states).toContain("auth_failed");
 });

 it("non-1008 close → disconnected + bounded-backoff reconnect", () => {
 vi.useFakeTimers();
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 MockWS.last().fireServerClose(1006); // abnormal drop
 expect(client.state).toBe("disconnected");
 expect(MockWS.instances.length).toBe(1);
 vi.advanceTimersByTime(100); // baseBackoff
 expect(MockWS.instances.length).toBe(2); // reconnected
 });

 it("disconnect() is user-intended → no reconnect", () => {
 vi.useFakeTimers();
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 client.disconnect();
 expect(client.state).toBe("disconnected");
 vi.advanceTimersByTime(5000);
 expect(MockWS.instances.length).toBe(1);
 });

 it("ignores unknown / malformed / non-JSON frames without throwing", () => {
 const { client, events } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 const ws = MockWS.last();
 expect(() => ws.fireMessage("not json{")).not.toThrow();
 expect(() => ws.fireMessage(JSON.stringify({ event: "unknown_evt", x: 1 }))).not.toThrow();
 expect(() => ws.fireMessage(JSON.stringify({ noEvent: true }))).not.toThrow();
 expect(() => ws.fireMessage(JSON.stringify("a string"))).not.toThrow();
 expect(() => ws.fireMessage(new ArrayBuffer(8))).not.toThrow(); // binary → ignored
 expect(events.length).toBe(0); // nothing dispatched
 });

 it("dispatches a known approval_request to listeners", () => {
 const { client, events } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 MockWS.last().fireMessage(
 JSON.stringify({ event: "approval_request", approval_id: "a1", tool: "run_command", args: { cmd: "ls" }, reason: "shell" }),
 );
 expect(events[0]).toMatchObject({ event: "approval_request", approval_id: "a1", tool: "run_command" });
 });

 it("token is NEVER written to localStorage / sessionStorage", () => {
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 client.runTask("x");
 for (let i = 0; i < localStorage.length; i++) {
 const k = localStorage.key(i)!;
 expect(localStorage.getItem(k)).not.toContain("SECRET-TOKEN-123");
 }
 expect(localStorage.length).toBe(0);
 expect(sessionStorage.length).toBe(0);
 });

 it("health() throws a status-only error (no token/body) on a failed request", async () => {
 const fetchFn = vi.fn(async () => ({ ok: false, status: 401, json: async () => ({}) }) as unknown as Response);
 const { client } = makeClient({ fetchFn });
 await expect(client.health()).rejects.toThrow(/401/);
 await expect(client.health()).rejects.not.toThrow(/SECRET-TOKEN-123/);
 });

 it("wsFactory throwing → error state + reconnect scheduled, no crash", () => {
 vi.useFakeTimers();
 let calls = 0;
 const { client } = makeClient({
 wsFactory: (url) => {
 calls += 1;
 if (calls === 1) throw new Error("ws construct failed");
 return new MockWS(url);
 },
 });
 expect(() => client.connect()).not.toThrow();
 expect(client.state).toBe("error");
 vi.advanceTimersByTime(100);
 expect(calls).toBe(2); // retried
 });
});

// --------------------------------------------------------------------------- //
// WS HANDSHAKE-FAILURE BOUNDING (auth-enabled deploy fix)
// --------------------------------------------------------------------------- //
describe("ApiClient — rejected/never-opening handshake does not loop forever", () => {
 it("repeated never-opened closes (1006) settle in terminal `error`, bounded reconnects", () => {
 vi.useFakeTimers();
 const { client } = makeClient();
 client.connect();
 for (let i = 0; i < 12; i++) {
 MockWS.last().fireServerClose(1006); // abnormal close, socket NEVER opened
 vi.advanceTimersByTime(1000); // let any scheduled reconnect fire
 }
 expect(client.state).toBe("error"); // settled, not toggling
 expect(MockWS.instances.length).toBeLessThanOrEqual(6); // initial + ≤MAX_HANDSHAKE_FAILURES reconnects
 });

 it("accept-then-1008 (the gateway's auth reject) → auth_failed, NO reconnect", () => {
 vi.useFakeTimers();
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen(); // gateway accepts first…
 MockWS.last().fireServerClose(1008); // …then closes with the 1008 frame
 expect(client.state).toBe("auth_failed");
 const n = MockWS.instances.length;
 vi.advanceTimersByTime(5000);
 expect(MockWS.instances.length).toBe(n); // never reconnects on auth failure
 });

 it("an opened-then-dropped socket reconnects (transient) and resets the failure streak", () => {
 vi.useFakeTimers();
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 expect(client.state).toBe("connected");
 MockWS.last().fireServerClose(1006); // transient drop AFTER a successful open
 expect(client.state).toBe("disconnected");
 vi.advanceTimersByTime(1000);
 expect(MockWS.instances.length).toBeGreaterThanOrEqual(2); // reconnected, not terminal
 });
});

// --------------------------------------------------------------------------- //
//— ingest upload + attachments wiring (zero authority)
// --------------------------------------------------------------------------- //
describe("ApiClient — ingest upload + attachments", () => {
 it("uploadFile POSTs multipart to /ingest with the auth header (no manual Content-Type), returns the id", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({ ok: true, status: 200, json: async () => ({ id: "abc123", filename: "r.txt", size: 3, ext: ".txt" }) }) as unknown as Response,
 );
 const { client } = makeClient({ fetchFn });
 const file = new File([new Uint8Array([1, 2, 3])], "r.txt", { type: "text/plain" });
 const out = await client.uploadFile(file);
 expect(out).toEqual({ id: "abc123", filename: "r.txt", size: 3, ext: ".txt" });
 const [url, init] = fetchFn.mock.calls[0];
 expect(url).toBe("http://core.local:8000/ingest");
 expect(init?.method).toBe("POST");
 expect(init?.body).toBeInstanceOf(FormData);
 const headers = init?.headers as Record<string, string>;
 expect(headers["Content-Type"]).toBeUndefined(); // the browser sets the multipart boundary
 expect(headers).toMatchObject({ "x-api-key": "SECRET-TOKEN-123" });
 });

 it("uploadFile throws a status-only error (no filename in the message) on failure", async () => {
 const fetchFn = vi.fn(async () => ({ ok: false, status: 413, json: async () => ({}) }) as unknown as Response);
 const { client } = makeClient({ fetchFn });
 const file = new File(["x"], "SENSITIVE-NAME.txt", { type: "text/plain" });
 await expect(client.uploadFile(file)).rejects.toThrow(/upload failed: 413/);
 await expect(client.uploadFile(file)).rejects.not.toThrow(/SENSITIVE-NAME/);
 });

 it("uploadFile rejects a malformed response (no id)", async () => {
 const fetchFn = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }) as unknown as Response);
 const { client } = makeClient({ fetchFn });
 await expect(client.uploadFile(new File(["x"], "a.txt"))).rejects.toThrow(/malformed/);
 });

 it("runTask carries attachments on the wire only when present", () => {
 const { client } = makeClient();
 client.connect();
 MockWS.last().fireOpen();
 expect(client.runTask("with", undefined, ["id1", "id2"])).toBe(true);
 expect(client.runTask("without", undefined, [])).toBe(true); // empty → omitted
 expect(MockWS.last().sent.map((s) => JSON.parse(s))).toEqual([
 { action: "run_task", task: "with", attachments: ["id1", "id2"] },
 { action: "run_task", task: "without" },
 ]);
 });

 it("chat includes attachments in the body only when present", async () => {
 const fetchFn = vi.fn(
 async (_input: RequestInfo | URL, _init?: RequestInit) =>
 ({ ok: true, status: 200, json: async () => ({}) }) as unknown as Response,
 );
 const { client } = makeClient({ fetchFn });
 await client.chat([{ role: "user", content: "q" }], undefined, ["a1"]);
 await client.chat([{ role: "user", content: "q2" }]);
 const body0 = JSON.parse(String(fetchFn.mock.calls[0][1]?.body));
 const body1 = JSON.parse(String(fetchFn.mock.calls[1][1]?.body));
 expect(body0).toMatchObject({ messages: [{ role: "user", content: "q" }], attachments: ["a1"] });
 expect("attachments" in body1).toBe(false);
 });
});
