import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { cleanup } from "@testing-library/react";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { Chat } from "./Chat";

import type { HistoryMessage } from "../api/types";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 stateL = new Set<(s: ConnState) => void>();
 runTaskCalls: string[] = [];
 historyCalls: (HistoryMessage[] | undefined)[] = [];
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
 this.historyCalls.push(history);
 return this.runTaskReturn;
 }
 emit(e: ServerEvent) {
 act(() => {
 for (const fn of this.eventL) fn(e);
 });
 }
}

const RESULT = (over: Partial<Extract<ServerEvent, { event: "task_result" }>> = {}): ServerEvent =>
 ({ event: "task_result", status: "completed", answer: "the answer", steps: 3, tool_calls_made: 2, ...over }) as ServerEvent;

function renderChat() {
 const fake = new FakeClient();
 render(<Chat client={fake as unknown as ApiClient} />);
 return { fake, user: userEvent.setup() };
}

afterEach(() => cleanup());

describe("Chat — normal", () => {
 it("send a task → user message + running, then task_result → agent answer + tool-activity summary", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "summarize x");
 await user.click(screen.getByRole("button", { name: "Send" }));

 const list = screen.getByTestId("messages");
 expect(within(list).getByText("summarize x")).toBeInTheDocument();
 expect(screen.getByTestId("running")).toBeInTheDocument();
 expect(fake.runTaskCalls).toEqual(["summarize x"]);

 fake.emit(RESULT({ answer: "done summary" }));
 expect(within(list).getByText("done summary")).toBeInTheDocument();
 expect(screen.getByTestId("tool-activity")).toHaveTextContent("status: completed");
 expect(screen.getByTestId("tool-activity")).toHaveTextContent("steps: 3");
 expect(screen.getByTestId("tool-activity")).toHaveTextContent("tools: 2");
 expect(screen.queryByTestId("running")).not.toBeInTheDocument();
 });
});

describe("Chatstreaming + multi-turn", () => {
 it("token events render the live streamed answer during the run", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit({ event: "token", delta: "Hel" });
 fake.emit({ event: "token", delta: "lo" });
 expect(screen.getByTestId("streaming-text")).toHaveTextContent("Hello");
 });

 it("tool_call / tool_result render live tool activity", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit({ event: "tool_call", id: "c1", tool: "summarize_document", args: {} });
 expect(screen.getByTestId("live-activity")).toHaveTextContent("summarize_document");
 fake.emit({ event: "tool_result", id: "c1", tool: "summarize_document", outcome: "executed", result: "ok" });
 expect(screen.getByTestId("live-activity")).toHaveTextContent("executed");
 });

 it("a streamed token containing HTML is rendered as escaped TEXT (no injection)", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit({ event: "token", delta: '<img src=x onerror="alert(1)">' });
 expect(screen.getByTestId("streaming-text")).toBeInTheDocument();
 expect(document.querySelector("img")).toBeNull();
 });

 it("the done meta includes the live tool-activity outcomes", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit({ event: "tool_call", id: "c1", tool: "summarize_document", args: {} });
 fake.emit({ event: "tool_result", id: "c1", tool: "summarize_document", outcome: "executed", result: "ok" });
 fake.emit(RESULT({ answer: "done" }));
 expect(screen.getByTestId("tool-activity")).toHaveTextContent("summarize_document(executed)");
 });

 it("multi-turn: a second task replays the prior user+agent turns as history", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "first question");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit(RESULT({ answer: "first answer" }));
 await user.type(screen.getByLabelText("task"), "second question");
 await user.click(screen.getByRole("button", { name: "Send" }));
 // the first run carried empty history; the second carries the prior turn pair
 expect(fake.historyCalls[0]).toEqual([]);
 expect(fake.historyCalls[1]).toEqual([
 { role: "user", content: "first question" },
 { role: "assistant", content: "first answer" },
 ]);
 });
});

describe("Chat — error / security", () => {
 it("input + button are disabled while running (single-flight)", async () => {
 const { user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(screen.getByLabelText("task")).toBeDisabled();
 expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
 });

 it("error event → a system error message, running cleared", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 fake.emit({ event: "error", reason: "agent not enabled" });
 expect(screen.getByText(/error: agent not enabled/)).toBeInTheDocument();
 expect(screen.queryByTestId("running")).not.toBeInTheDocument();
 });

 it("an answer containing HTML/script is rendered as escaped TEXT (no injection)", async () => {
 const { fake, user } = renderChat();
 await user.type(screen.getByLabelText("task"), "x");
 await user.click(screen.getByRole("button", { name: "Send" }));
 const payload = '<img src=x onerror="alert(1)">';
 fake.emit(RESULT({ answer: payload }));
 // the literal text is present...
 expect(screen.getByText(payload)).toBeInTheDocument();
 // ...and NO actual <img> element was created from the answer
 expect(document.querySelector("img")).toBeNull();
 });

 it("Send is disabled for empty/whitespace input", async () => {
 const { user } = renderChat();
 const send = screen.getByRole("button", { name: "Send" });
 expect(send).toBeDisabled();
 await user.type(screen.getByLabelText("task"), " ");
 expect(send).toBeDisabled();
 });
});
