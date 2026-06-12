import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { ChatTabs } from "./ChatTabs";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 stateL = new Set<(s: ConnState) => void>();
 runTaskCalls: Array<[string, string | undefined, string[] | undefined]> = [];
 chatCalls: Array<[Array<{ role: string; content: string }>, string[] | undefined]> = [];
 uploadResult: { id: string; filename: string; size: number; ext: string } | Error = {
 id: "id-1",
 filename: "doc.txt",
 size: 3,
 ext: ".txt",
 };
 onEvent(fn: (e: ServerEvent) => void) {
 this.eventL.add(fn);
 return () => this.eventL.delete(fn);
 }
 onState(fn: (s: ConnState) => void) {
 this.stateL.add(fn);
 return () => this.stateL.delete(fn);
 }
 runTask(task: string, system?: string, attachments?: string[]) {
 this.runTaskCalls.push([task, system, attachments]);
 return true;
 }
 async chat(
 messages: Array<{ role: string; content: string }>,
 _params?: Record<string, number>,
 attachments?: string[],
 ) {
 this.chatCalls.push([messages, attachments]);
 return { status: 200, ok: true, data: { choices: [{ message: { content: "r" } }] } };
 }
 async uploadFile() {
 if (this.uploadResult instanceof Error) throw this.uploadResult;
 return this.uploadResult;
 }
 // CapabilitiesController.refresh() reads these (multimodal availability)
 capsRest: { available: string[]; all: string[] } = { available: [], all: [] };
 async capabilitiesRest() {
 return this.capsRest;
 }
 async health() {
 return { status: "ok" };
 }
}
const asClient = (f: FakeClient) => f as unknown as ApiClient;
afterEach(cleanup);

describe("ChatTabs — shared system prompt across distinct Agent/Direct tabs", () => {
 it("Agent tab (default): the shared system prompt reaches run_task", async () => {
 const fake = new FakeClient();
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("system-prompt"), "You are terse.");
 await user.type(screen.getByLabelText("task"), "do a thing");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(fake.runTaskCalls).toEqual([["do a thing", "You are terse.", []]]); // no attachments → empty
 });

 it("Direct tab is toolless + not gated, and the system prompt is prepended to /chat", async () => {
 const fake = new FakeClient();
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("system-prompt"), "Be brief.");
 await user.click(screen.getByRole("tab", { name: "Direct" }));
 expect(screen.getByTestId("direct-toolless-note")).toHaveTextContent("no tools");
 await user.type(screen.getByLabelText("direct-task"), "hi there");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(fake.chatCalls[0][0]).toEqual([
 { role: "system", content: "Be brief." },
 { role: "user", content: "hi there" },
 ]);
 expect(fake.chatCalls[0][1]).toEqual([]); // no attachments
 // the gated agent path was NOT used by the direct tab
 expect(fake.runTaskCalls).toEqual([]);
 });

 it("tabs are mutually exclusive (only one chat surface VISIBLE)", async () => {
 //(DH6/ISSUE-001): both panels stay MOUNTED, but only the active one is visible/
 // accessible — role queries exclude the `hidden` panel, so this user-facing assertion still holds.
 const fake = new FakeClient();
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 expect(screen.getByRole("region", { name: "chat" })).toBeInTheDocument(); // agent
 expect(screen.queryByRole("region", { name: "direct-chat" })).toBeNull();
 await user.click(screen.getByRole("tab", { name: "Direct" }));
 expect(screen.getByRole("region", { name: "direct-chat" })).toBeInTheDocument();
 expect(screen.queryByRole("region", { name: "chat" })).toBeNull();
 });

 it("ISSUE-001 fix (DH6): switching tabs PRESERVES the Agent chat history", async () => {
 const fake = new FakeClient();
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("task"), "remember me");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(screen.getByTestId("messages")).toHaveTextContent("remember me");
 // away to Direct and back to Agent — pre-fix this unmounted Chat and WIPED the transcript
 await user.click(screen.getByRole("tab", { name: "Direct" }));
 await user.click(screen.getByRole("tab", { name: "Agent" }));
 expect(screen.getByTestId("messages")).toHaveTextContent("remember me"); // survived
 });
});

describe("ChatTabs — file attachment upload (zero authority)", () => {
 it("uploads a file → chip appears → the id is attached to run_task", async () => {
 const fake = new FakeClient();
 fake.uploadResult = { id: "ing-7", filename: "report.txt", size: 5, ext: ".txt" };
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.upload(screen.getByLabelText("attach-file"), new File(["hello"], "report.txt", { type: "text/plain" }));
 expect(screen.getByTestId("attachment-chips")).toHaveTextContent("report.txt");
 await user.type(screen.getByLabelText("task"), "summarize it");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(fake.runTaskCalls).toEqual([["summarize it", "", ["ing-7"]]]);
 });

 it("the attachment is shared with the Direct tab", async () => {
 const fake = new FakeClient();
 fake.uploadResult = { id: "ing-9", filename: "n.txt", size: 1, ext: ".txt" };
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.upload(screen.getByLabelText("attach-file"), new File(["x"], "n.txt"));
 await user.click(screen.getByRole("tab", { name: "Direct" }));
 await user.type(screen.getByLabelText("direct-task"), "what is it");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(fake.chatCalls[0][1]).toEqual(["ing-9"]);
 });

 it("removing a chip drops the id from the next send", async () => {
 const fake = new FakeClient();
 fake.uploadResult = { id: "ing-1", filename: "a.txt", size: 1, ext: ".txt" };
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.upload(screen.getByLabelText("attach-file"), new File(["x"], "a.txt"));
 await user.click(screen.getByLabelText("remove-a.txt"));
 expect(screen.queryByTestId("attachment-chips")).toBeNull();
 await user.type(screen.getByLabelText("task"), "go");
 await user.click(screen.getByRole("button", { name: "Send" }));
 expect(fake.runTaskCalls).toEqual([["go", "", []]]);
 });

 it("an upload error shows a friendly notice and no chip (server is authoritative)", async () => {
 const fake = new FakeClient();
 fake.uploadResult = new Error("upload failed: 415");
 render(<ChatTabs client={asClient(fake)} />);
 // bypass the client `accept` hint (non-authoritative) to exercise the SERVER's 415 rejection
 const user = userEvent.setup({ applyAccept: false });
 await user.upload(screen.getByLabelText("attach-file"), new File(["x"], "bad.exe"));
 expect(screen.getByTestId("upload-error")).toHaveTextContent("unsupported file type");
 expect(screen.queryByTestId("attachment-chips")).toBeNull();
 });

 it("a 403 upload error tells the operator to re-mint with the ingest scope", async () => {
 const fake = new FakeClient();
 fake.uploadResult = new Error("upload failed: 403");
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.upload(screen.getByLabelText("attach-file"), new File(["x"], "x.txt"));
 expect(screen.getByTestId("upload-error")).toHaveTextContent("ingest");
 });

 it("when multimodal is available, the picker suggests images + shows the images hint", async () => {
 const fake = new FakeClient();
 fake.capsRest = { available: ["multimodal"], all: ["chat", "multimodal"] };
 render(<ChatTabs client={asClient(fake)} />);
 const hint = await screen.findByTestId("attach-hint");
 expect(hint).toHaveTextContent(/images/);
 expect(screen.getByLabelText("attach-file").getAttribute("accept")).toContain(".png");
 });

 it("text-only (no multimodal) → the hint says text documents and the picker omits images", async () => {
 const fake = new FakeClient();
 fake.capsRest = { available: ["chat"], all: ["chat"] };
 render(<ChatTabs client={asClient(fake)} />);
 // the hint defaults to text-only and stays so after refresh
 const hint = await screen.findByTestId("attach-hint");
 expect(hint).toHaveTextContent(/text documents/);
 expect(screen.getByLabelText("attach-file").getAttribute("accept")).not.toContain(".png");
 });

 it("a hostile filename is rendered as escaped text (no markup injection)", async () => {
 const fake = new FakeClient();
 const hostile = "<img src=x onerror=alert(1)>.txt";
 fake.uploadResult = { id: "ing-x", filename: hostile, size: 1, ext: ".txt" };
 render(<ChatTabs client={asClient(fake)} />);
 const user = userEvent.setup();
 await user.upload(screen.getByLabelText("attach-file"), new File(["x"], "x.txt"));
 const chips = screen.getByTestId("attachment-chips");
 expect(chips).toHaveTextContent(hostile); // shown literally
 expect(chips.querySelector("img")).toBeNull(); // NOT parsed into a DOM element
 });
});
