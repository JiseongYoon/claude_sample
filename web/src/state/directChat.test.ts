import { describe, expect, it } from "vitest";
import type { ApiClient } from "../api/client";
import { DirectChatController } from "./directChat";

class FakeClient {
 chatCalls: Array<Array<{ role: string; content: string }>> = [];
 status = 200;
 reply: string | null = "the model reply";
 throwOnce = false;
 async chat(messages: Array<{ role: string; content: string }>) {
 this.chatCalls.push(messages);
 if (this.throwOnce) {
 this.throwOnce = false;
 throw new Error("network");
 }
 const data =
 this.reply === null ? {} : { choices: [{ message: { role: "assistant", content: this.reply } }] };
 return { status: this.status, ok: this.status < 400, data };
 }
}

function setup() {
 const fake = new FakeClient();
 return { fake, ctrl: new DirectChatController(fake as unknown as ApiClient) };
}

describe("DirectChatController ", () => {
 it("a 200 turn appends user then assistant; the user message is sent", async () => {
 const { fake, ctrl } = setup();
 await ctrl.send("hello");
 expect(ctrl.current.messages).toEqual([
 { role: "user", content: "hello" },
 { role: "assistant", content: "the model reply" },
 ]);
 expect(fake.chatCalls[0]).toEqual([{ role: "user", content: "hello" }]);
 expect(ctrl.current.status).toBe("idle");
 });

 it("multi-turn: the second send carries the full prior transcript", async () => {
 const { fake, ctrl } = setup();
 await ctrl.send("one");
 await ctrl.send("two");
 expect(fake.chatCalls[1]).toEqual([
 { role: "user", content: "one" },
 { role: "assistant", content: "the model reply" },
 { role: "user", content: "two" },
 ]);
 });

 it("a system prompt is prepended per call (not stored in the transcript)", async () => {
 const { fake, ctrl } = setup();
 await ctrl.send("hi", "You are terse.");
 expect(fake.chatCalls[0][0]).toEqual({ role: "system", content: "You are terse." });
 expect(fake.chatCalls[0][1]).toEqual({ role: "user", content: "hi" });
 // the transcript itself holds only user/assistant (system is per-call)
 expect(ctrl.current.messages.every((m) => m.role !== ("system" as unknown))).toBe(true);
 });

 it("503 → error notice 'no model loaded', user message retained", async () => {
 const { fake, ctrl } = setup();
 fake.status = 503;
 await ctrl.send("hi");
 expect(ctrl.current.status).toBe("error");
 expect(ctrl.current.notice).toBe("no model loaded");
 expect(ctrl.current.messages).toEqual([{ role: "user", content: "hi" }]);
 });

 it("502 → 'upstream engine error'", async () => {
 const { fake, ctrl } = setup();
 fake.status = 502;
 await ctrl.send("hi");
 expect(ctrl.current.notice).toBe("upstream engine error");
 });

 it("malformed 200 (no choices) → assistant '(no reply)' + a notice, no crash", async () => {
 const { fake, ctrl } = setup();
 fake.reply = null; // → {} response
 await ctrl.send("hi");
 expect(ctrl.current.messages[1]).toEqual({ role: "assistant", content: "(no reply)" });
 expect(ctrl.current.notice).toBe("malformed response");
 });

 it("single-flight: a second send while sending is dropped", async () => {
 const { fake, ctrl } = setup();
 const p1 = ctrl.send("a");
 const p2 = ctrl.send("b"); // sending → dropped
 await Promise.all([p1, p2]);
 expect(fake.chatCalls.length).toBe(1);
 });

 it("bounds the SENT transcript (≤40 messages) even when the conversation is long", async () => {
 const { fake, ctrl } = setup();
 for (let i = 0; i < 30; i++) await ctrl.send(`m${i}`); // 30 user + 30 assistant = 60 in history
 expect(ctrl.current.messages.length).toBe(60); // full transcript kept for display
 const lastWire = fake.chatCalls[fake.chatCalls.length - 1];
 expect(lastWire.length).toBeLessThanOrEqual(40); // payload bounded
 });

 it("a thrown request → 'request failed', no crash", async () => {
 const { fake, ctrl } = setup();
 fake.throwOnce = true;
 await ctrl.send("hi");
 expect(ctrl.current.status).toBe("error");
 expect(ctrl.current.notice).toBe("request failed");
 });
});
