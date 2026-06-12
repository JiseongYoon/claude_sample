// DirectChatController — a TOOLLESS, multi-turn chat over REST /chat (). This is NOT the
// gated agent path: /chat runs the model directly with no tool loop and no safety gate (the agent's
// tools remain reachable only via the gated run_task loop). It keeps a local user/assistant transcript;
// an optional shared system prompt is prepended per call. Single-flight; malformed/again-safe. The
// server is authoritative (a 503/502/403 is surfaced, never bypassed).

import type { ApiClient } from "../api/client";

export interface DirectMsg {
 role: "user" | "assistant";
 content: string;
}

export interface DirectState {
 messages: DirectMsg[];
 status: "idle" | "sending" | "error";
 notice: string | null;
}

const EMPTY: DirectState = { messages: [], status: "idle", notice: null };

// bound the transcript SENT per call (the full history is kept for display, but only the most recent
// turns are wired) — avoids an unbounded request payload as a conversation grows (client-side DoS guard;
// the server independently clamps max_tokens). The system prompt is always prepended on top of this.
const MAX_SENT_MESSAGES = 40;

function noticeForChat(status: number): string {
 if (status === 503) return "no model loaded";
 if (status === 502) return "upstream engine error";
 if (status === 403) return "requires invoke scope";
 return `chat failed (${status})`;
}

function parseReply(data: unknown): string | null {
 if (!data || typeof data !== "object") return null;
 const choices = (data as { choices?: unknown }).choices;
 if (!Array.isArray(choices) || choices.length === 0) return null;
 const msg = (choices[0] as { message?: unknown }).message;
 if (!msg || typeof msg !== "object") return null;
 const content = (msg as { content?: unknown }).content;
 return typeof content === "string" ? content : null;
}

export class DirectChatController {
 private readonly client: ApiClient;
 private state: DirectState = EMPTY;
 private readonly listeners = new Set<(s: DirectState) => void>();

 constructor(client: ApiClient) {
 this.client = client;
 }

 get current(): DirectState {
 return this.state;
 }
 subscribe(fn: (s: DirectState) => void): () => void {
 this.listeners.add(fn);
 fn(this.state);
 return () => this.listeners.delete(fn);
 }
 private set(patch: Partial<DirectState>): void {
 this.state = { ...this.state, ...patch };
 for (const fn of this.listeners) fn(this.state);
 }

 /** Send a turn. Single-flight; appends the user message optimistically, then the assistant reply on a
 * 200, or surfaces the server's status (503/502/403/…) without crashing. `system` is prepended per call. */
 async send(text: string, system?: string, attachments?: string[]): Promise<void> {
 if (this.state.status === "sending") return;
 const t = text.trim();
 if (!t) return;
 const history: DirectMsg[] = [...this.state.messages, { role: "user", content: t }];
 this.set({ messages: history, status: "sending", notice: null });

 const sys = system?.trim();
 const sent = history.slice(-MAX_SENT_MESSAGES); // bound the payload (client DoS guard)
 const wire = [
 ...(sys ? [{ role: "system", content: sys }] : []),
 ...sent.map((m) => ({ role: m.role, content: m.content })),
 ];
 let res: { status: number; ok: boolean; data: unknown };
 try {
 res = await this.client.chat(wire, undefined, attachments);
 } catch {
 this.set({ status: "error", notice: "request failed" });
 return;
 }
 if (!res.ok) {
 this.set({ status: "error", notice: noticeForChat(res.status) });
 return;
 }
 const reply = parseReply(res.data);
 this.set({
 messages: [...history, { role: "assistant", content: reply ?? "(no reply)" }],
 status: "idle",
 notice: reply === null ? "malformed response" : null,
 });
 }

 dispose(): void {
 this.listeners.clear();
 }
}
