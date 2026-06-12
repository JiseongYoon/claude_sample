// — hermetic INTEGRATED-WHOLE regression of thesurfaces composed together.
// One mint → one ApiClient → everycontroller cooperating over ONE (mock) socket + a mock fetch
// that routes all REST endpoints:
// mintToken(/auth/token) → connect+capabilities → ModelController(status/params/files/health + load)
// → DirectChatController(/chat multi-turn w/ shared system) → RunController(run_task w/ system) →
// ApprovalsController(approval_request → approve) → task_result → citations from the answer text.
// Proves the v2 client works as a whole. Network-free. The REAL-socket counterpart is `web/e2e/`.
import { describe, expect, it } from "vitest";
import { ApiClient, type WebSocketLike } from "./api/client";
import { CapabilitiesController } from "./state/capabilities";
import { DirectChatController } from "./state/directChat";
import { ModelController } from "./state/model";
import { ApprovalsController } from "./state/approvals";
import { RunController } from "./state/run";
import { extractSources } from "./lib/citations";

class MockWS implements WebSocketLike {
 sent: string[] = [];
 onopen: ((ev: unknown) => void) | null = null;
 onclose: ((ev: { code: number }) => void) | null = null;
 onmessage: ((ev: { data: unknown }) => void) | null = null;
 onerror: ((ev: unknown) => void) | null = null;
 send(d: string) {
 this.sent.push(d);
 }
 close() {}
 fireOpen() {
 this.onopen?.({});
 }
 recv(obj: unknown) {
 this.onmessage?.({ data: JSON.stringify(obj) });
 }
 sentActions() {
 return this.sent.map((s) => JSON.parse(s));
 }
}

// a mock fetch routing everyREST endpoint (by path + method).
function makeFetch(): typeof fetch {
 return (async (input: RequestInfo | URL, init?: RequestInit) => {
 const url = String(input);
 const method = (init?.method ?? "GET").toUpperCase();
 const j = (body: unknown, status = 200) =>
 ({ ok: status < 400, status, json: async () => body }) as unknown as Response;
 if (url.endsWith("/auth/token") && method === "POST") {
 const sent = JSON.parse(String(init?.body ?? "{}"));
 return j({ access_token: "JWT-mint", scopes: sent.scopes, expires_in: 3600 });
 }
 if (url.endsWith("/capabilities")) return j({ available: ["agent", "docqa"], all: ["agent", "docqa", "storage"] });
 if (url.endsWith("/health"))
 return j({ status: "ok", modules: { "model-manager": "ok", "llm-serving": "ok", docqa: "ok" } });
 if (url.endsWith("/model/status")) return j({ state: "ready", loaded_file: "m.gguf", serving: true });
 if (url.endsWith("/model/params")) return j({ params: { temperature: 0.7, top_k: 40 } });
 if (url.endsWith("/model/files")) return j({ files: ["a.gguf", "b.gguf"] });
 if (url.endsWith("/model/load") && method === "POST") return j({ state: "launching", accepted: true }, 202);
 if (url.endsWith("/storage/connectors"))
 return j({ connectors: [{ name: "nas", kind: "ssh", host: "h", read_only: true }], health: "ok" });
 if (url.endsWith("/mcp/servers")) return j({ servers: [{ name: "fs", command: "srv", connected: true }], health: "ok" });
 if (url.endsWith("/tools")) return j({ tools: [{ name: "delete_file", tier: "needs_confirmation", capability: "exec" }] });
 if (url.endsWith("/chat") && method === "POST") return j({ choices: [{ message: { content: "direct reply" } }] });
 return j({});
 }) as typeof fetch;
}

describe("— composed v2 client integrated round-trip", () => {
 it("mint → connect → model panel → direct + agent chat → approval → result → citations", async () => {
 const fetchFn = makeFetch();

 // 1) token bootstrap: mint a scoped JWT from an API key (the key is the caller's; discarded after)
 const minted = await ApiClient.mintToken({
 baseUrl: "http://core.local:8080",
 apiKey: "API-KEY",
 scopes: ["read", "invoke", "agent:run", "approve", "model:admin"],
 fetchFn,
 });
 expect(minted.access_token).toBe("JWT-mint");
 expect(minted.scopes).toContain("model:admin"); // → App would set canAdmin = true

 let ws!: MockWS;
 const client = new ApiClient({
 baseUrl: "http://core.local:8080",
 token: minted.access_token,
 authScheme: "bearer",
 wsFactory: () => (ws = new MockWS()),
 fetchFn,
 });

 const caps = new CapabilitiesController(client);
 const model = new ModelController(client);
 const direct = new DirectChatController(client);
 const run = new RunController(client, { timeoutMs: 5000 });
 const approvals = new ApprovalsController(client);

 // 2) connect + capabilities handshake
 client.connect();
 ws.fireOpen();
 expect(client.state).toBe("connected");
 ws.recv({ event: "capabilities", available: ["agent", "docqa"] });
 await caps.refresh();
 expect(caps.current.overall).toBe("ok");

 // 3) model panel: refresh populates status/params/files/modules; an admin load returns 202
 await model.refresh();
 expect(model.current.status).toMatchObject({ state: "ready", serving: true });
 expect(model.current.files).toEqual(["a.gguf", "b.gguf"]);
 expect(model.current.modules.map((m) => m.name)).toContain("llm-serving");
 await model.load("a.gguf");
 expect(model.current.notice).toBe("load: launching…");

 // 4) direct (toolless) chat: multi-turn /chat with the shared system prompt
 await direct.send("hello there", "You are concise.");
 expect(direct.current.messages).toEqual([
 { role: "user", content: "hello there" },
 { role: "assistant", content: "direct reply" },
 ]);

 // 5) agent (gated) chat: run_task carries the shared system; the gated tool triggers approval
 expect(run.start("delete the old file", "You are concise.")).toBe(true);
 ws.recv({
 event: "approval_request",
 approval_id: "ap-1",
 tool: "delete_file",
 args: { path: "workspace/old.txt" },
 reason: "file mutation",
 });
 expect(approvals.list[0]).toMatchObject({ approval_id: "ap-1", tool: "delete_file", status: "pending" });
 expect(approvals.decide("ap-1", "approve")).toBe(true);

 // 6) the run completes; the answer carries a source → citations extract it
 ws.recv({
 event: "task_result",
 status: "completed",
 answer: "Done. See https://src.example/ref for details.",
 steps: 2,
 tool_calls_made: 1,
 });
 expect(run.state).toMatchObject({ status: "done" });
 const answer = (run.state as { result: { answer: string } }).result.answer;
 expect(extractSources(answer)).toEqual(["https://src.example/ref"]);

 // the WS wire carried EXACTLY: run_task (with system) then approve (id only — no forged tool/args)
 expect(ws.sentActions()).toEqual([
 { action: "run_task", task: "delete the old file", system: "You are concise." },
 { action: "approve", approval_id: "ap-1" },
 ]);

 caps.dispose();
 model.dispose();
 direct.dispose();
 run.dispose();
 approvals.dispose();
 });

 it("discovery endpoints compose over the one client (non-secret shapes reach the UI)", async () => {
 const client = new ApiClient({ baseUrl: "http://c:8080", token: "K", fetchFn: makeFetch() });
 expect(await client.listModelFiles()).toEqual({ files: ["a.gguf", "b.gguf"] });
 expect(await client.listTools()).toMatchObject({ tools: [{ name: "delete_file", tier: "needs_confirmation" }] });
 const conns = (await client.listConnectors()) as { connectors: Array<Record<string, unknown>> };
 expect(conns.connectors[0]).not.toHaveProperty("password_env"); // non-secret projection
 const servers = (await client.listMcpServers()) as { servers: Array<Record<string, unknown>> };
 expect(servers.servers[0]).not.toHaveProperty("secret_env");
 });
});
