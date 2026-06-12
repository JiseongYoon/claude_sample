// — hermetic INTEGRATED-WHOLE regression of the composed client.
// All four client modules (ApiClient + RunController + ApprovalsController + CapabilitiesController)
// run together on ONE ApiClient over ONE (mock) socket, driven through the REAL server event sequence:
// connect → capabilities → run_task → approval_request → approve → task_result → capabilities-update.
// Proves the client behaves correctly as a whole (not just per-module). Network-free, deterministic.
// The REAL-socket counterpart (live core + real WS) is `web/e2e/` (operator-run / CI).
import { describe, expect, it } from "vitest";
import { ApiClient, type WebSocketLike } from "./api/client";
import { RunController } from "./state/run";
import { ApprovalsController } from "./state/approvals";
import { CapabilitiesController } from "./state/capabilities";

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

describe("— composed client integrated round-trip", () => {
 it("connect → capabilities → run → approval → approve → result, all modules cooperating", async () => {
 let ws!: MockWS;
 const fetchFn = (async (input: RequestInfo | URL) => {
 const url = String(input);
 const body = url.endsWith("/capabilities")
 ? { available: ["agent", "exec"], all: ["agent", "exec", "docqa"] }
 : url.endsWith("/health")
 ? { status: "degraded", modules: {} }
 : {};
 return { ok: true, status: 200, json: async () => body } as unknown as Response;
 }) as typeof fetch;

 const client = new ApiClient({
 baseUrl: "http://core.local:8000",
 token: "K",
 wsFactory: (u) => {
 ws = new MockWS();
 void u;
 return ws;
 },
 fetchFn,
 });

 const run = new RunController(client, { timeoutMs: 5000 });
 const approvals = new ApprovalsController(client);
 const caps = new CapabilitiesController(client);

 // 1) connect + the capabilities handshake
 client.connect();
 ws.fireOpen();
 expect(client.state).toBe("connected");
 ws.recv({ event: "capabilities", available: ["agent", "exec"] });
 expect(client.capabilities).toMatchObject({ available: ["agent", "exec"] });

 // capability view: REST fills the full set; exec available, docqa absent
 await caps.refresh();
 expect(caps.current.overall).toBe("degraded");
 expect(caps.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "docqa", available: false },
 { name: "exec", available: true },
 ]);

 // 2) start a run → run_task on the wire
 expect(run.start("clean the workspace")).toBe(true);
 expect(run.state.status).toBe("running");

 // 3) the agent proposes a gated tool → approval_request; run stays alive (liveness)
 ws.recv({
 event: "approval_request",
 approval_id: "ap-1",
 tool: "run_command",
 args: { cmd: "rm -rf build" },
 reason: "side-effecting shell",
 });
 expect(approvals.list).toHaveLength(1);
 expect(approvals.list[0]).toMatchObject({ approval_id: "ap-1", tool: "run_command", status: "pending" });
 expect(run.state.status).toBe("running");

 // 4) operator approves → approve on the wire (only the id + verdict)
 expect(approvals.decide("ap-1", "approve")).toBe(true);
 expect(approvals.list[0]).toMatchObject({ status: "resolved", decision: "approve" });

 // 5) the run completes → task_result → done with the tool-activity summary fields
 ws.recv({ event: "task_result", status: "completed", answer: "workspace cleaned", steps: 2, tool_calls_made: 1 });
 expect(run.state).toMatchObject({ status: "done", task: "clean the workspace" });
 expect((run.state as { result: { answer: string } }).result).toMatchObject({
 answer: "workspace cleaned",
 steps: 2,
 tool_calls_made: 1,
 });

 // the wire carried exactly: run_task then approve (id + verdict only — no tool/args forged)
 expect(ws.sentActions()).toEqual([
 { action: "run_task", task: "clean the workspace" },
 { action: "approve", approval_id: "ap-1" },
 ]);

 // 6) a live capabilities update flows to the view
 ws.recv({ event: "capabilities", available: ["agent"] });
 expect(caps.current.caps.find((c) => c.name === "exec")!.available).toBe(false);

 run.dispose();
 approvals.dispose();
 caps.dispose();
 });

 it("deny path: operator denies → deny on the wire; a late timeout doesn't override", () => {
 let ws!: MockWS;
 const client = new ApiClient({
 baseUrl: "http://c:8000",
 token: "K",
 wsFactory: () => (ws = new MockWS()),
 });
 const approvals = new ApprovalsController(client);
 client.connect();
 ws.fireOpen();
 ws.recv({ event: "approval_request", approval_id: "ap-9", tool: "open_url", args: { url: "http://x" }, reason: "nav" });
 expect(approvals.decide("ap-9", "deny")).toBe(true);
 ws.recv({ event: "approval_timeout", approval_id: "ap-9" }); // late → must not override
 expect(approvals.list[0]).toMatchObject({ status: "resolved", decision: "deny" });
 expect(ws.sentActions()).toEqual([{ action: "deny", approval_id: "ap-9" }]);
 approvals.dispose();
 });
});
