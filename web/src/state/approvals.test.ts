import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { ApprovalsController } from "./approvals";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 stateL = new Set<(s: ConnState) => void>();
 approveCalls: string[] = [];
 denyCalls: string[] = [];
 sendReturn = true;
 onEvent(fn: (e: ServerEvent) => void) {
 this.eventL.add(fn);
 return () => this.eventL.delete(fn);
 }
 onState(fn: (s: ConnState) => void) {
 this.stateL.add(fn);
 return () => this.stateL.delete(fn);
 }
 approve(id: string) {
 this.approveCalls.push(id);
 return this.sendReturn;
 }
 deny(id: string) {
 this.denyCalls.push(id);
 return this.sendReturn;
 }
 emit(e: ServerEvent) {
 for (const fn of this.eventL) fn(e);
 }
 emitState(s: ConnState) {
 for (const fn of this.stateL) fn(s);
 }
}

function setup() {
 const fake = new FakeClient();
 const ctrl = new ApprovalsController(fake as unknown as ApiClient);
 return { fake, ctrl };
}

const REQ = (id: string, over: Partial<Extract<ServerEvent, { event: "approval_request" }>> = {}): ServerEvent =>
 ({ event: "approval_request", approval_id: id, tool: "run_command", args: { cmd: "ls" }, reason: "shell", ...over }) as ServerEvent;

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("ApprovalsController — normal", () => {
 it("approval_request → a pending item rendered verbatim", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1", { tool: "open_url", args: { url: "http://x" }, reason: "navigation" }));
 expect(ctrl.list).toHaveLength(1);
 expect(ctrl.list[0]).toMatchObject({
 approval_id: "a1",
 tool: "open_url",
 args: { url: "http://x" },
 reason: "navigation",
 status: "pending",
 });
 });

 it("decide approve → sends approve(id) ONCE and marks resolved", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 expect(ctrl.decide("a1", "approve")).toBe(true);
 expect(fake.approveCalls).toEqual(["a1"]);
 expect(fake.denyCalls).toEqual([]);
 expect(ctrl.list[0]).toMatchObject({ status: "resolved", decision: "approve" });
 });

 it("decide deny → sends deny(id) and marks resolved", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 expect(ctrl.decide("a1", "deny")).toBe(true);
 expect(fake.denyCalls).toEqual(["a1"]);
 expect(ctrl.list[0]).toMatchObject({ status: "resolved", decision: "deny" });
 });

 it("two concurrent requests resolve independently with the correct id", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1", { tool: "run_command" }));
 fake.emit(REQ("a2", { tool: "open_url" }));
 expect(ctrl.list).toHaveLength(2);
 ctrl.decide("a2", "approve");
 expect(fake.approveCalls).toEqual(["a2"]); // only a2 sent
 expect(ctrl.list.find((i) => i.approval_id === "a1")!.status).toBe("pending");
 expect(ctrl.list.find((i) => i.approval_id === "a2")!.status).toBe("resolved");
 });
});

describe("ApprovalsController — SECURITY (zero authority)", () => {
 it("NO auto-approve: with no decide() call, approve/deny are NEVER sent (even after time)", () => {
 const { fake } = setup();
 fake.emit(REQ("a1"));
 vi.advanceTimersByTime(10 * 60 * 1000); // 10 minutes pass
 fake.emit(REQ("a2"));
 vi.advanceTimersByTime(10 * 60 * 1000);
 expect(fake.approveCalls).toEqual([]);
 expect(fake.denyCalls).toEqual([]);
 });

 it("decide on an UNKNOWN id → no-op, no send", () => {
 const { fake, ctrl } = setup();
 expect(ctrl.decide("nope", "approve")).toBe(false);
 expect(fake.approveCalls).toEqual([]);
 });

 it("decide TWICE → second is a no-op (no double-send)", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 ctrl.decide("a1", "approve");
 expect(ctrl.decide("a1", "deny")).toBe(false); // already resolved
 expect(ctrl.decide("a1", "approve")).toBe(false);
 expect(fake.approveCalls).toEqual(["a1"]);
 expect(fake.denyCalls).toEqual([]);
 });

 it("decide after timeout (expired) → no-op, no send", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 fake.emit({ event: "approval_timeout", approval_id: "a1" });
 expect(ctrl.list[0].status).toBe("expired");
 expect(ctrl.decide("a1", "approve")).toBe(false);
 expect(fake.approveCalls).toEqual([]);
 });

 it("approval_error → item flagged error, decide no-op", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 fake.emit({ event: "approval_error", approval_id: "a1", reason: "malformed" });
 expect(ctrl.list[0]).toMatchObject({ status: "error", error: "malformed" });
 expect(ctrl.decide("a1", "approve")).toBe(false);
 expect(fake.approveCalls).toEqual([]);
 });

 it("connection lost → all pending expired (no lingering actionable prompt)", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 fake.emit(REQ("a2"));
 fake.emitState("disconnected");
 expect(ctrl.list.every((i) => i.status === "expired")).toBe(true);
 expect(ctrl.decide("a1", "approve")).toBe(false);
 expect(fake.approveCalls).toEqual([]);
 });

 it("send failure (not connected) → marked error, returns false, not resolved", () => {
 const { fake, ctrl } = setup();
 fake.sendReturn = false;
 fake.emit(REQ("a1"));
 expect(ctrl.decide("a1", "approve")).toBe(false);
 expect(ctrl.list[0].status).toBe("error");
 });

 it("duplicate approval_id → ignored (first wins), no spurious item", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1", { tool: "run_command" }));
 fake.emit(REQ("a1", { tool: "EVIL_swap" }));
 expect(ctrl.list).toHaveLength(1);
 expect(ctrl.list[0].tool).toBe("run_command"); // not swapped
 });

 it("approval_timeout for an already-resolved id does NOT override it", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 ctrl.decide("a1", "approve");
 fake.emit({ event: "approval_timeout", approval_id: "a1" });
 expect(ctrl.list[0].status).toBe("resolved");
 });

 it("approval_error for an already-resolved id does NOT override it (terminal state preserved)", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 ctrl.decide("a1", "approve");
 fake.emit({ event: "approval_error", approval_id: "a1", reason: "late error" });
 expect(ctrl.list[0]).toMatchObject({ status: "resolved", decision: "approve" });
 });

 it("approval_error with null id → ignored (no crash, no item change)", () => {
 const { fake, ctrl } = setup();
 fake.emit(REQ("a1"));
 expect(() => fake.emit({ event: "approval_error", approval_id: null, reason: "x" })).not.toThrow();
 expect(ctrl.list[0].status).toBe("pending");
 });
});
