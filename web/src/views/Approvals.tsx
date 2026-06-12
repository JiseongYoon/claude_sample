// Approvals view — renders each `approval_request` VERBATIM (tool · reason · args) so the operator has
// full context, and relays the explicit Approve/Deny. ZERO authority: the UI never auto-decides; only a
// `pending` request shows active buttons; expired/resolved/error requests are visibly disabled. Args are
// rendered as escaped, length-BOUNDED text (no dangerouslySetInnerHTML, no execution) — an MCP tool's
// args or a shell command can be large/hostile.
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import { ApprovalsController, type ApprovalItem } from "../state/approvals";
import styles from "./Approvals.module.css";

const ARGS_CAP = 2000; // chars; args beyond this are truncated (display bound, not a security control)

function renderArgs(args: Record<string, unknown>): { text: string; truncated: boolean } {
 let text: string;
 try {
 text = JSON.stringify(args, null, 2);
 } catch {
 return { text: "[unserializable args]", truncated: false };
 }
 if (text.length > ARGS_CAP) return { text: text.slice(0, ARGS_CAP), truncated: true };
 return { text, truncated: false };
}

export function Approvals({
 client,
 decisionTimeoutSec = 120,
 nowFn = () => Date.now(),
}: {
 client: ApiClient;
 decisionTimeoutSec?: number; // the server's APPROVAL_DECISION_TIMEOUT_SECONDS (display only)
 nowFn?: () => number;
}) {
 const [items, setItems] = useState<ApprovalItem[]>([]);
 const ctrlRef = useRef<ApprovalsController | null>(null);
 const seenAt = useRef<Map<string, number>>(new Map());
 const [, setTick] = useState(0);

 useEffect(() => {
 const ctrl = new ApprovalsController(client);
 ctrlRef.current = ctrl;
 setItems(ctrl.list);
 const off = ctrl.subscribe(setItems);
 return () => {
 off();
 ctrl.dispose();
 ctrlRef.current = null;
 };
 }, [client]);

 const anyPending = items.some((it) => it.status === "pending");

 // stamp first-seen time per pending request (display-only — the security-critical controller is
 // untouched). The countdown is a HINT; the server's approval_timeout remains authoritative.
 useEffect(() => {
 for (const it of items) {
 if (it.status === "pending" && !seenAt.current.has(it.approval_id)) {
 seenAt.current.set(it.approval_id, nowFn());
 }
 }
 }, [items, nowFn]);

 // re-render once a second while something is pending so the countdown advances
 useEffect(() => {
 if (!anyPending) return;
 const id = setInterval(() => setTick((t) => t + 1), 1000);
 return () => clearInterval(id);
 }, [anyPending]);

 function secondsLeft(id: string): number {
 const start = seenAt.current.get(id);
 if (start === undefined) return decisionTimeoutSec;
 return Math.max(0, decisionTimeoutSec - Math.floor((nowFn() - start) / 1000));
 }

 function decide(id: string, decision: "approve" | "deny") {
 ctrlRef.current?.decide(id, decision);
 }

 if (items.length === 0) return null;

 return (
 <section aria-label="approvals" data-testid="approvals" className={styles.panel}>
 <h2 className={styles.heading}>Approvals</h2>
 <ul className={styles.list}>
 {items.map((it) => {
 const { text, truncated } = renderArgs(it.args);
 const pending = it.status === "pending";
 const left = pending ? secondsLeft(it.approval_id) : null;
 const clientTimedOut = left === 0; // display-only: grey the buttons; server expiry is authoritative
 return (
 <li
 key={it.approval_id}
 className={styles.item}
 data-testid="approval"
 data-status={it.status}
 >
 <div>
 <strong className={styles.tool}>{it.tool}</strong>
 {it.reason && <span className={styles.reason}> — {it.reason}</span>}
 </div>
 <details className={styles.args}>
 <summary>args</summary>
 <pre className={styles.argsPre} data-testid="approval-args">
 {text}
 {truncated ? "\n… (truncated)" : ""}
 </pre>
 </details>
 {pending ? (
 <div className={styles.actions}>
 <span className={styles.countdown} data-testid="approval-countdown">
 {left}s
 </span>
 <button
 type="button"
 className={styles.approve}
 disabled={clientTimedOut}
 onClick={() => decide(it.approval_id, "approve")}
 >
 Approve
 </button>
 <button
 type="button"
 className={styles.deny}
 disabled={clientTimedOut}
 onClick={() => decide(it.approval_id, "deny")}
 >
 Deny
 </button>
 </div>
 ) : (
 <div className={styles.resolved} data-testid="approval-resolved">
 {it.status === "resolved" && `decision: ${it.decision}`}
 {it.status === "expired" && "expired (timed out)"}
 {it.status === "error" && `error: ${it.error ?? "unknown"}`}
 </div>
 )}
 </li>
 );
 })}
 </ul>
 </section>
 );
}
