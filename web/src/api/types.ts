// Typed mirror of the core's WS protocol (orchestrator/session.py + gateway /ws).
// Server frames are UNTRUSTED input: we parse only the known shapes and ignore everything else.

export interface CapabilitiesEvent {
 event: "capabilities";
 // the core sends a capabilities snapshot; shape is intentionally open (rendered as data,)
 [key: string]: unknown;
}

export interface ApprovalRequestEvent {
 event: "approval_request";
 approval_id: string;
 tool: string;
 args: Record<string, unknown>;
 reason: string;
}

export interface ApprovalTimeoutEvent {
 event: "approval_timeout";
 approval_id: string;
}

export interface ApprovalErrorEvent {
 event: "approval_error";
 approval_id: string | null;
 reason: string;
}

export interface TaskResultEvent {
 event: "task_result";
 status: string;
 answer: string | null;
 steps: number;
 tool_calls_made: number;
}

// /additive streaming-observation frames (display-only; zero authority).
export interface TokenEvent {
 event: "token";
 delta: string;
}

export interface ToolCallEvent {
 event: "tool_call";
 id: string;
 tool: string;
 args: Record<string, unknown>;
}

export interface ToolResultEvent {
 event: "tool_result";
 id: string;
 tool: string;
 outcome: string;
 result: string;
}

export interface ErrorEvent {
 event: "error";
 reason: string;
}

export interface EchoEvent {
 event: "echo";
 data: unknown;
}

export type ServerEvent =
 | CapabilitiesEvent
 | ApprovalRequestEvent
 | ApprovalTimeoutEvent
 | ApprovalErrorEvent
 | TaskResultEvent
 | TokenEvent
 | ToolCallEvent
 | ToolResultEvent
 | ErrorEvent
 | EchoEvent;

export const KNOWN_EVENTS = new Set<ServerEvent["event"]>([
 "capabilities",
 "approval_request",
 "approval_timeout",
 "approval_error",
 "task_result",
 "token",
 "tool_call",
 "tool_result",
 "error",
 "echo",
]);

// A prior conversation turn the client replays so the agent remembers context ().
// Text-only; the server rejects/strips anything else (no authority injection).
export interface HistoryMessage {
 role: "user" | "assistant";
 content: string;
}

// UI → server commands (the wire carries only these; the server binds the action server-side).
export type ClientCommand =
 | { action: "run_task"; task: string; system?: string; attachments?: string[]; history?: HistoryMessage[] }
 | { action: "approve"; approval_id: string }
 | { action: "deny"; approval_id: string };
