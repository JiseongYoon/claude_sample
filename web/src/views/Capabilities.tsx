// Capabilities view — shows which capabilities are available by module health (the plan's "capability
// visibility by module health"). Read-only/informational: it gates nothing (the server does). An
// unavailable capability is annotated so the operator isn't offered a call the core would refuse.
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import { CapabilitiesController, type CapabilitiesState } from "../state/capabilities";
import styles from "./Capabilities.module.css";

export function Capabilities({ client }: { client: ApiClient }) {
 const [state, setState] = useState<CapabilitiesState>({ caps: [], overall: null });
 const ctrlRef = useRef<CapabilitiesController | null>(null);

 useEffect(() => {
 const ctrl = new CapabilitiesController(client);
 ctrlRef.current = ctrl;
 const off = ctrl.subscribe(setState);
 void ctrl.refresh();
 return () => {
 off();
 ctrl.dispose();
 ctrlRef.current = null;
 };
 }, [client]);

 return (
 <section aria-label="capabilities" className={styles.panel}>
 <h2 className={styles.heading}>
 Capabilities{" "}
 {state.overall && (
 <span className={styles.overall} data-overall={state.overall} data-testid="overall">
 {state.overall}
 </span>
 )}
 </h2>
 {state.caps.length === 0 ? (
 <p className={styles.empty}>no capabilities reported</p>
 ) : (
 <ul className={styles.list}>
 {state.caps.map((c) => (
 <li
 key={c.name}
 className={styles.item}
 data-testid="cap"
 data-cap={c.name}
 data-available={c.available}
 >
 <span className={styles.name}>{c.name}</span>
 <span className={styles.badge} data-available={c.available}>
 {c.available ? "available" : "unavailable"}
 </span>
 </li>
 ))}
 </ul>
 )}
 </section>
 );
}
