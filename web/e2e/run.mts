// Web-UI real-API e2e. Drives the REAL frontend `ApiClient` against a REAL running
// core (web/e2e/serve_fake_core.py) over a REAL WebSocket — proving the client speaks the live protocol
// end-to-end: connect → capabilities → run_task → approval_request → approve → task_result. Run via tsx
// (Node 22). Exits 0 on PASS, 1 on FAIL. Operator/CI-run (like the backend smokes), not the unit suite.
import WebSocket from "ws";
import { ApiClient, type WebSocketLike } from "../src/api/client.ts";
import type { ServerEvent } from "../src/api/types.ts";

const PORT = Number(process.argv[2] ?? process.env.E2E_PORT ?? 8137);
const BASE = `http://127.0.0.1:${PORT}`;

// Adapt the Node `ws` socket to the browser-like WebSocketLike the ApiClient expects (coerce text
// frames to string; map close event to {code}).
function nodeWsFactory(url: string): WebSocketLike {
 const sock = new WebSocket(url);
 const wrap: WebSocketLike = {
 send: (d) => sock.send(d),
 close: () => sock.close(),
 onopen: null,
 onclose: null,
 onmessage: null,
 onerror: null,
 };
 sock.onopen = () => wrap.onopen?.({});
 sock.onclose = (e) => wrap.onclose?.({ code: e.code });
 sock.onerror = (e) => wrap.onerror?.(e);
 sock.onmessage = (e) => {
 const data = typeof e.data === "string" ? e.data : (e.data as Buffer).toString("utf8");
 wrap.onmessage?.({ data });
 };
 return wrap;
}

const checks: Array<[string, boolean]> = [];
const log = (name: string, ok: boolean) => {
 checks.push([name, ok]);
 console.log(` ${ok ? "PASS" : "FAIL"} ${name}`);
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function main(): Promise<number> {
 const client = new ApiClient({
 baseUrl: BASE,
 token: "e2e", // auth disabled on the fake core; the token rides as ?token= harmlessly
 wsFactory: nodeWsFactory,
 fetchFn: (...a) => fetch(...a as Parameters<typeof fetch>),
 autoReconnect: false,
 });

 const events: ServerEvent[] = [];
 let approvalId: string | null = null;
 client.onEvent((e) => {
 events.push(e);
 if (e.event === "approval_request") approvalId = e.approval_id;
 });

 client.connect();
 // wait for connect + the capabilities handshake
 for (let i = 0; i < 50 && client.state !== "connected"; i++) await wait(100);
 log("connect", client.state === "connected");
 log("capabilities handshake received", events.some((e) => e.event === "capabilities"));

 // REST: /health reachable + authed
 try {
 const h = (await client.health()) as { status?: string };
 log("REST /health", typeof h.status === "string");
 } catch {
 log("REST /health", false);
 }

 // --surfaces over the LIVE core -- //
 // mint a scoped JWT from an API key (auth disabled here → the route still issues a token)
 try {
 const m = await ApiClient.mintToken({
 baseUrl: BASE,
 apiKey: "e2e",
 scopes: ["read", "invoke"],
 fetchFn: (...a) => fetch(...(a as Parameters<typeof fetch>)),
 });
 log("mintToken issues a token", typeof m.access_token === "string" && m.access_token.length > 0);
 } catch {
 log("mintToken issues a token", false);
 }
 // read-only discovery endpoints live
 try {
 const files = (await client.listModelFiles()) as { files?: string[] };
 log("/model/files lists the GGUF", Array.isArray(files.files) && files.files.includes("e2e-model.gguf"));
 } catch {
 log("/model/files lists the GGUF", false);
 }
 try {
 const tools = (await client.listTools()) as { tools?: Array<{ name: string; tier: string }> };
 const del = tools.tools?.find((t) => t.name === "delete_file");
 log("/tools roster (delete_file = needs_confirmation)", del?.tier === "needs_confirmation");
 } catch {
 log("/tools roster (delete_file = needs_confirmation)", false);
 }
 try {
 const conns = (await client.listConnectors()) as { connectors?: unknown[] };
 const servers = (await client.listMcpServers()) as { servers?: unknown[] };
 log("/storage/connectors + /mcp/servers (empty, correct shape)",
 Array.isArray(conns.connectors) && Array.isArray(servers.servers));
 } catch {
 log("/storage/connectors + /mcp/servers (empty, correct shape)", false);
 }
 // direct /chat — no model is loaded on the fake core, so the client must SURFACE 503 (not crash)
 try {
 const r = await client.chat([{ role: "user", content: "hi" }]);
 log("direct /chat surfaces 503 (no model loaded)", r.status === 503 && r.ok === false);
 } catch {
 log("direct /chat surfaces 503 (no model loaded)", false);
 }
 // upload a real file via the REAL ApiClient (multipart) → an opaque id back
 try {
 const f = new File([new TextEncoder().encode("E2E-INGEST-CONTENT")], "e2e.txt", { type: "text/plain" });
 const r = await client.uploadFile(f);
 log("ingest uploadFile returns an opaque id", typeof r.id === "string" && r.id.length > 0 && r.filename === "e2e.txt");
 } catch {
 log("ingest uploadFile returns an opaque id", false);
 }
 // a wrong-type upload is rejected server-side (415) — the client surfaces it as a status-only error
 try {
 const bad = new File([new Uint8Array([1, 2, 3])], "x.exe", { type: "application/octet-stream" });
 await client.uploadFile(bad);
 log("ingest rejects wrong type (415)", false); // should have thrown
 } catch (e) {
 log("ingest rejects wrong type (415)", e instanceof Error && /upload failed: 415/.test(e.message));
 }

 // run a task CARRYING prior-turn history (transport) → the scripted model proposes
 // the gated delete_file → approval_request. The history is accepted end-to-end (not rejected).
 log(
 "send run_task (with multi-turn history)",
 client.runTask("delete the old file", undefined, undefined, [
 { role: "user", content: "earlier question" },
 { role: "assistant", content: "earlier answer" },
 ]),
 );
 for (let i = 0; i < 50 && !approvalId; i++) await wait(100);
 const req = events.find((e) => e.event === "approval_request") as
 | Extract<ServerEvent, { event: "approval_request" }>
 | undefined;
 log("approval_request received", !!req && req.tool === "delete_file");
 // the proposed tool surfaced as a live tool_call event (display only)
 const toolCall = events.find((e) => e.event === "tool_call") as
 | Extract<ServerEvent, { event: "tool_call" }>
 | undefined;
 log("tool_call event streamed", toolCall?.tool === "delete_file");

 // approve → the run completes with a task_result
 if (approvalId) client.approve(approvalId);
 for (let i = 0; i < 50 && !events.some((e) => e.event === "task_result"); i++) await wait(100);
 const result = events.find((e) => e.event === "task_result") as
 | Extract<ServerEvent, { event: "task_result" }>
 | undefined;
 log("task_result completed after approve (history accepted)", result?.status === "completed");
 // /the tool result + the streamed final-answer tokens arrived over the real WS
 const toolResult = events.find((e) => e.event === "tool_result") as
 | Extract<ServerEvent, { event: "tool_result" }>
 | undefined;
 log("tool_result event streamed", toolResult?.tool === "delete_file");
 const tokens = events.filter((e) => e.event === "token") as Array<Extract<ServerEvent, { event: "token" }>>;
 log("token deltas streamed", tokens.length > 0 && tokens.map((t) => t.delta).join("") === "deleted the file");

 client.disconnect();
 await wait(50);

 const ok = checks.every(([, p]) => p);
 console.log(`\n[e2e] ${ok ? "ALL PASS" : "FAILURES"} (${checks.filter(([, p]) => p).length}/${checks.length})`);
 return ok ? 0 : 1;
}

main().then((code) => process.exit(code), (err) => {
 console.error("[e2e] crashed:", err);
 process.exit(1);
});
