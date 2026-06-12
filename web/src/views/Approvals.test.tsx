import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { Approvals } from "./Approvals";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 stateL = new Set<(s: ConnState) => void>();
 approveCalls: string[] = [];
 denyCalls: string[] = [];
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
 return true;
 }
 deny(id: string) {
 this.denyCalls.push(id);
 return true;
 }
 emit(e: ServerEvent) {
 act(() => {
 for (const fn of this.eventL) fn(e);
 });
 }
 emitState(s: ConnState) {
 act(() => {
 for (const fn of this.stateL) fn(s);
 });
 }
}

const REQ = (id: string, over: Partial<Extract<ServerEvent, { event: "approval_request" }>> = {}): ServerEvent =>
 ({ event: "approval_request", approval_id: id, tool: "run_command", args: { cmd: "ls -la" }, reason: "shell", ...over }) as ServerEvent;

function renderApprovals() {
 const fake = new FakeClient();
 render(<Approvals client={fake as unknown as ApiClient} />);
 return { fake, user: userEvent.setup() };
}

afterEach(() => cleanup());

describe("Approvals view — normal", () => {
 it("renders a request verbatim (tool · reason · args) and relays Approve", async () => {
 const { fake, user } = renderApprovals();
 fake.emit(REQ("a1", { tool: "open_url", args: { url: "http://example.com" }, reason: "navigation" }));
 const item = screen.getByTestId("approval");
 expect(within(item).getByText("open_url")).toBeInTheDocument();
 expect(within(item).getByText(/navigation/)).toBeInTheDocument();
 expect(screen.getByTestId("approval-args")).toHaveTextContent('"url": "http://example.com"');
 await user.click(within(item).getByRole("button", { name: "Approve" }));
 expect(fake.approveCalls).toEqual(["a1"]);
 expect(screen.getByTestId("approval-resolved")).toHaveTextContent("decision: approve");
 });

 it("Deny relays deny(id)", async () => {
 const { fake, user } = renderApprovals();
 fake.emit(REQ("a1"));
 await user.click(screen.getByRole("button", { name: "Deny" }));
 expect(fake.denyCalls).toEqual(["a1"]);
 });

 it("two concurrent prompts → approving one sends only its id", async () => {
 const { fake, user } = renderApprovals();
 fake.emit(REQ("a1", { tool: "run_command" }));
 fake.emit(REQ("a2", { tool: "open_url" }));
 const items = screen.getAllByTestId("approval");
 expect(items).toHaveLength(2);
 await user.click(within(items[1]).getByRole("button", { name: "Approve" }));
 expect(fake.approveCalls).toEqual(["a2"]);
 });
});

describe("Approvals view — SECURITY", () => {
 it("NO auto-approve: rendering a request never sends a decision on its own", () => {
 const { fake } = renderApprovals();
 fake.emit(REQ("a1"));
 fake.emit(REQ("a2"));
 expect(fake.approveCalls).toEqual([]);
 expect(fake.denyCalls).toEqual([]);
 // and there IS no default-selected action — both buttons are present, neither pre-clicked
 expect(screen.getAllByRole("button", { name: "Approve" })).toHaveLength(2);
 });

 it("expired (timeout) → buttons gone, status shown; a (would-be) late action can't fire", () => {
 const { fake } = renderApprovals();
 fake.emit(REQ("a1"));
 fake.emit({ event: "approval_timeout", approval_id: "a1" });
 expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
 expect(screen.getByTestId("approval-resolved")).toHaveTextContent("expired");
 expect(fake.approveCalls).toEqual([]);
 });

 it("after Approve, the buttons are replaced by the resolved status (no second click possible)", async () => {
 const { fake, user } = renderApprovals();
 fake.emit(REQ("a1"));
 await user.click(screen.getByRole("button", { name: "Approve" }));
 expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
 expect(screen.queryByRole("button", { name: "Deny" })).not.toBeInTheDocument();
 expect(fake.approveCalls).toEqual(["a1"]); // exactly once
 });

 it("hostile args (huge + script-like) render as escaped, bounded text — no injection, no freeze", () => {
 const { fake } = renderApprovals();
 const hostile = {
 payload: "<img src=x onerror=alert(1)>",
 big: "Z".repeat(10000),
 };
 fake.emit(REQ("a1", { tool: "run_command", args: hostile }));
 const pre = screen.getByTestId("approval-args");
 expect(pre.textContent).toContain("onerror"); // present as text
 expect(document.querySelector("img")).toBeNull(); // NOT an element
 expect(pre.textContent!.length).toBeLessThanOrEqual(2100); // bounded (ARGS_CAP + marker)
 expect(pre.textContent).toContain("truncated");
 });

 it("connection lost → pending prompt expired in the UI (buttons gone)", () => {
 const { fake } = renderApprovals();
 fake.emit(REQ("a1"));
 fake.emitState("disconnected");
 expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
 expect(screen.getByTestId("approval-resolved")).toHaveTextContent("expired");
 });

 it("malformed approval_request (missing tool) → safe render, no crash", () => {
 const { fake } = renderApprovals();
 expect(() =>
 fake.emit({ event: "approval_request", approval_id: "a1", reason: "x" } as unknown as ServerEvent),
 ).not.toThrow();
 expect(screen.getByTestId("approval")).toHaveTextContent("(unknown tool)");
 });
});

describe("Approvals view — decision countdown ", () => {
 it("counts down and disables the buttons at 0 without ever sending a decision", () => {
 vi.useFakeTimers();
 try {
 let now = 10_000;
 const fake = new FakeClient();
 render(
 <Approvals
 client={fake as unknown as ApiClient}
 decisionTimeoutSec={5}
 nowFn={() => now}
 />,
 );
 fake.emit(REQ("c1"));
 expect(screen.getByTestId("approval-countdown")).toHaveTextContent("5s");
 // 3s later → 2s, buttons still enabled
 now = 13_000;
 act(() => vi.advanceTimersByTime(3000));
 expect(screen.getByTestId("approval-countdown")).toHaveTextContent("2s");
 expect(screen.getByRole("button", { name: "Approve" })).toBeEnabled();
 // past the timeout → 0s, buttons disabled (display-only)
 now = 16_000;
 act(() => vi.advanceTimersByTime(3000));
 expect(screen.getByTestId("approval-countdown")).toHaveTextContent("0s");
 expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
 expect(screen.getByRole("button", { name: "Deny" })).toBeDisabled();
 // THE CRUX: the countdown reaching 0 sent NO decision (zero-authority preserved)
 expect(fake.approveCalls).toEqual([]);
 expect(fake.denyCalls).toEqual([]);
 // the item is still "pending" (the SERVER's approval_timeout is authoritative, not the client clock)
 expect(screen.getByTestId("approval").getAttribute("data-status")).toBe("pending");
 } finally {
 vi.useRealTimers();
 }
 });

 it("an explicit click before timeout still relays the decision", async () => {
 let now = 0;
 const fake = new FakeClient();
 const user = userEvent.setup();
 render(
 <Approvals client={fake as unknown as ApiClient} decisionTimeoutSec={120} nowFn={() => now} />,
 );
 fake.emit(REQ("c2"));
 await user.click(screen.getByRole("button", { name: "Approve" }));
 expect(fake.approveCalls).toEqual(["c2"]);
 });
});
