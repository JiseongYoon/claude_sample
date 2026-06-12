import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import type { HistoryMessage } from "../api/types";
import { RunController, type RunState } from "./run";

// A minimal fake of the slice of ApiClient that RunController uses (onEvent/onState/runTask).
class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 stateL = new Set<(s: ConnState) => void>();
 runTaskCalls: string[] = [];
 lastHistory: HistoryMessage[] | undefined;
 runTaskReturn = true;
 onEvent(fn: (e: ServerEvent) => void) {
 this.eventL.add(fn);
 return () => this.eventL.delete(fn);
 }
 onState(fn: (s: ConnState) => void) {
 this.stateL.add(fn);
 return () => this.stateL.delete(fn);
 }
 runTask(task: string, _system?: string, _attachments?: string[], history?: HistoryMessage[]): boolean {
 this.runTaskCalls.push(task);
 this.lastHistory = history;
 return this.runTaskReturn;
 }
 emit(e: ServerEvent) {
 for (const fn of this.eventL) fn(e);
 }
 emitState(s: ConnState) {
 for (const fn of this.stateL) fn(s);
 }
}

function setup(opts?: { timeoutMs?: number }) {
 const fake = new FakeClient();
 const ctrl = new RunController(fake as unknown as ApiClient, opts ?? {});
 const states: RunState[] = [];
 ctrl.subscribe((s) => states.push(s));
 return { fake, ctrl, states };
}

const RESULT = (over: Partial<Extract<ServerEvent, { event: "task_result" }>> = {}) =>
 ({ event: "task_result", status: "completed", answer: "hi", steps: 2, tool_calls_made: 1, ...over }) as ServerEvent;

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("RunController — normal", () => {
 it("start → running → task_result → done (timer cleared)", () => {
 const { fake, ctrl } = setup();
 expect(ctrl.start("do it")).toBe(true);
 expect(ctrl.state).toEqual({ status: "running", task: "do it", streamedText: "", activity: [] });
 expect(fake.runTaskCalls).toEqual(["do it"]);
 fake.emit(RESULT({ answer: "done!" }));
 expect(ctrl.state).toMatchObject({ status: "done", task: "do it" });
 expect((ctrl.state as Extract<RunState, { status: "done" }>).result.answer).toBe("done!");
 });

 it("error event → error state", () => {
 const { fake, ctrl } = setup();
 ctrl.start("x");
 fake.emit({ event: "error", reason: "agent not enabled" });
 expect(ctrl.state).toEqual({ status: "error", task: "x", reason: "agent not enabled" });
 });
});

describe("RunControllerstreaming + multi-turn", () => {
 it("token events accumulate the streamed answer (display only)", () => {
 const { fake, ctrl } = setup();
 ctrl.start("x");
 fake.emit({ event: "token", delta: "Hel" });
 fake.emit({ event: "token", delta: "lo" });
 const s = ctrl.state as Extract<RunState, { status: "running" }>;
 expect(s.streamedText).toBe("Hello");
 });

 it("tool_call then tool_result populate live activity (matched by id)", () => {
 const { fake, ctrl } = setup();
 ctrl.start("x");
 fake.emit({ event: "tool_call", id: "c1", tool: "summarize_document", args: { path: "a" } });
 let s = ctrl.state as Extract<RunState, { status: "running" }>;
 expect(s.activity).toEqual([{ id: "c1", tool: "summarize_document" }]);
 fake.emit({ event: "tool_result", id: "c1", tool: "summarize_document", outcome: "executed", result: "ok" });
 s = ctrl.state as Extract<RunState, { status: "running" }>;
 expect(s.activity).toEqual([{ id: "c1", tool: "summarize_document", outcome: "executed" }]);
 });

 it("streamed text + activity carry into the done state", () => {
 const { fake, ctrl } = setup();
 ctrl.start("x");
 fake.emit({ event: "tool_call", id: "c1", tool: "t", args: {} });
 fake.emit({ event: "token", delta: "answer" });
 fake.emit(RESULT({ answer: "answer" }));
 const s = ctrl.state as Extract<RunState, { status: "done" }>;
 expect(s.streamedText).toBe("answer");
 expect(s.activity).toEqual([{ id: "c1", tool: "t" }]);
 });

 it("a fresh run resets streamed text + activity", () => {
 const { fake, ctrl } = setup();
 ctrl.start("a");
 fake.emit({ event: "token", delta: "x" });
 fake.emit(RESULT());
 ctrl.reset();
 ctrl.start("b");
 const s = ctrl.state as Extract<RunState, { status: "running" }>;
 expect(s.streamedText).toBe("");
 expect(s.activity).toEqual([]);
 });

 it("start forwards prior-turn history to runTask", () => {
 const { fake, ctrl } = setup();
 const history: HistoryMessage[] = [
 { role: "user", content: "q1" },
 { role: "assistant", content: "a1" },
 ];
 ctrl.start("q2", undefined, undefined, history);
 expect(fake.lastHistory).toEqual(history);
 });

 it("token events keep the run alive (reset the timeout)", () => {
 const { fake, ctrl } = setup({ timeoutMs: 1000 });
 ctrl.start("x");
 vi.advanceTimersByTime(900);
 fake.emit({ event: "token", delta: "…" });
 vi.advanceTimersByTime(900); // would have timed out at 1000 without the reset
 expect(ctrl.state.status).toBe("running");
 });
});

describe("RunController — error / security", () => {
 it("single-flight: a second start while running is ignored", () => {
 const { fake, ctrl } = setup();
 ctrl.start("first");
 expect(ctrl.start("second")).toBe(false);
 expect(fake.runTaskCalls).toEqual(["first"]); // second never sent
 });

 it("empty / whitespace task → not started", () => {
 const { fake, ctrl } = setup();
 expect(ctrl.start(" ")).toBe(false);
 expect(fake.runTaskCalls).toEqual([]);
 });

 it("not connected (runTask returns false) → error state", () => {
 const { fake, ctrl } = setup();
 fake.runTaskReturn = false;
 expect(ctrl.start("x")).toBe(false);
 expect(ctrl.state).toEqual({ status: "error", task: "x", reason: "not connected" });
 });

 it("timeout: no server activity within the bound → timeout", () => {
 const { fake, ctrl } = setup({ timeoutMs: 1000 });
 ctrl.start("x");
 vi.advanceTimersByTime(999);
 expect(ctrl.state.status).toBe("running");
 vi.advanceTimersByTime(2);
 expect(ctrl.state).toEqual({ status: "timeout", task: "x" });
 void fake;
 });

 it("liveness: an approval_request resets the timeout (long human waits don't trip it)", () => {
 const { fake, ctrl } = setup({ timeoutMs: 1000 });
 ctrl.start("x");
 vi.advanceTimersByTime(900);
 fake.emit({ event: "approval_request", approval_id: "a1", tool: "run_command", args: {}, reason: "shell" });
 vi.advanceTimersByTime(900); // would have timed out at 1000 without the reset
 expect(ctrl.state.status).toBe("running");
 fake.emit(RESULT());
 expect(ctrl.state.status).toBe("done");
 });

 it("connection lost mid-run → error (no permanent spinner)", () => {
 const { fake, ctrl } = setup();
 ctrl.start("x");
 fake.emitState("disconnected");
 expect(ctrl.state).toEqual({ status: "error", task: "x", reason: "connection lost" });
 });

 it("events while idle are ignored (no spurious transition)", () => {
 const { fake, ctrl } = setup();
 fake.emit(RESULT());
 fake.emit({ event: "error", reason: "x" });
 fake.emitState("disconnected");
 expect(ctrl.state).toEqual({ status: "idle" });
 });

 it("dispose unsubscribes + clears timer", () => {
 const { fake, ctrl } = setup({ timeoutMs: 1000 });
 ctrl.start("x");
 ctrl.dispose();
 vi.advanceTimersByTime(5000);
 // after dispose, no further transitions; a late event is ignored
 fake.emit(RESULT());
 expect(ctrl.state.status).toBe("running"); // frozen at last pre-dispose state, no timeout fired
 });
});
