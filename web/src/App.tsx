// App shell — orchestrates auth → connect → the views. The operator enters a token (Auth), which
// (re)builds the ApiClient and connects; sign-out disconnects + clears the in-memory token. When
// connected, the capability view + approvals + chat mount (all share the one client). The UI holds
// zero authority — the server gate is authoritative.
import { useState } from "react";
import { ApiClient, type AuthScheme, type ConnState } from "./api/client";
import { clearToken, setConfig, type RuntimeConfig } from "./config";
import { Approvals } from "./views/Approvals";
import { Auth } from "./views/Auth";
import { Capabilities } from "./views/Capabilities";
import { ChatTabs } from "./views/ChatTabs";
import { ModelPanel } from "./views/ModelPanel";
import styles from "./App.module.css";

export function App() {
 const [client, setClient] = useState<ApiClient | null>(null);
 const [state, setState] = useState<ConnState>("disconnected");
 const [authScheme, setAuthScheme] = useState<AuthScheme>("api-key");
 const [grantedScopes, setGrantedScopes] = useState<string[] | null>(null);

 function connect(cfg: RuntimeConfig, scopes?: string[]) {
 client?.disconnect();
 setConfig(cfg);
 setAuthScheme(cfg.authScheme);
 setGrantedScopes(scopes ?? null); // known only via the mint flow
 const c = new ApiClient({ baseUrl: cfg.baseUrl, token: cfg.token, authScheme: cfg.authScheme });
 c.onState(setState);
 c.connect();
 setClient(c);
 }

 // model-admin gating hint for the panel. If minted a token we KNOW the granted scopes → exact
 // answer (model:admin or `*` ⇒ true, else false → controls locked). Otherwise an API key carries full
 // scope (true); a pasted bearer token's scopes are unknown (undefined ⇒ enabled, server surfaces 403).
 const canAdmin: boolean | undefined = grantedScopes
 ? grantedScopes.includes("model:admin") || grantedScopes.includes("*")
 : authScheme === "api-key"
 ? true
 : undefined;

 function signOut() {
 client?.disconnect();
 clearToken();
 setClient(null);
 setState("disconnected");
 setGrantedScopes(null);
 }

 const connected = !!client && state === "connected";

 return (
 <main className={styles.shell}>
 <header className={styles.header}>
 <h1 className={styles.title}>Local AI Agent</h1>
 <span className={styles.conn}>
 <span className={styles.dot} data-state={state} aria-hidden="true" />
 connection: <strong className={styles.stateLabel} data-testid="conn-state">{state}</strong>
 </span>
 {client && (
 <button type="button" onClick={signOut} className={styles.signOut}>
 Sign out
 </button>
 )}
 </header>
 <div className={styles.content}>
 {!client && <Auth onConnect={connect} />}
 {client && state === "auth_failed" && (
 <p className={styles.authFailed} data-testid="auth-failed">
 authentication failed — check the token and reconnect.
 </p>
 )}
 {connected && (
 <div className={styles.workspace}>
 {/* ISSUE-004: the dense Model/API control panel was crammed into the narrow 300px sidebar
 (which also drove ISSUE-003's input overflow). It now sits in its own full-width row at
 the top of the workspace — room to breathe, and the params grid stops overflowing. It
 stays mounted here (its ModelController/polling persists), and authority is unchanged:
 model-admin is still server-gated (`canAdmin` is a UX hint only). */}
 <ModelPanel client={client} canAdmin={canAdmin} />
 <div className={styles.split}>
 <div className={styles.sidebar}>
 <Capabilities client={client} />
 </div>
 <div className={styles.mainCol}>
 <Approvals client={client} />
 <ChatTabs client={client} />
 </div>
 </div>
 </div>
 )}
 </div>
 </main>
 );
}
