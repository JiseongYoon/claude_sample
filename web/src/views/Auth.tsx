// Auth view — connect to the core. Two modes:
// • "paste" — enter a token directly (API-Key or JWT). (Unchanged from.)
// • "mint" — enter an API key, pick scopes, and mint a scoped JWT via /auth/token ().
// Secrets are held in MEMORY only (never localStorage/disk, never logged). DE3: after a mint (success
// OR failure) the API key is DISCARDED from state — it limits the XSS-exfil window and forces re-entry to
// re-mint. The minted JWT is handed to onConnect (held in the client's memory), never stored.
import { useState } from "react";
import { ApiClient, type AuthScheme, type MintArgs, type MintResult } from "../api/client";
import { getConfig, type RuntimeConfig } from "../config";

type Mode = "paste" | "mint";

// sensible operator default — model:admin is opt-in (off), scope minimization (DE3). `ingest`
// ( upload files) is on by default for an interactive operator; model:admin stays off.
const SCOPES = ["read", "invoke", "agent:run", "approve", "ingest", "model:admin"] as const;
const DEFAULT_ON = new Set<string>(["read", "invoke", "agent:run", "approve", "ingest"]);

export function Auth({
 onConnect,
 mint = ApiClient.mintToken,
}: {
 onConnect: (cfg: RuntimeConfig, scopes?: string[]) => void;
 mint?: (a: MintArgs) => Promise<MintResult>;
}) {
 const cfg = getConfig();
 const [mode, setMode] = useState<Mode>("paste");
 const [baseUrl, setBaseUrl] = useState(cfg.baseUrl);
 // paste mode
 const [token, setToken] = useState(""); // never seeded from storage
 const [authScheme, setAuthScheme] = useState<AuthScheme>(cfg.authScheme);
 // mint mode
 const [apiKey, setApiKey] = useState("");
 const [scopes, setScopes] = useState<Set<string>>(new Set(DEFAULT_ON));
 const [expiry, setExpiry] = useState("");
 const [minting, setMinting] = useState(false);
 const [error, setError] = useState<string | null>(null);

 function submitPaste(e: React.FormEvent) {
 e.preventDefault();
 const t = token.trim();
 const b = baseUrl.trim();
 if (!t || !b) return;
 onConnect({ baseUrl: b, token: t, authScheme }); // EXACTLY one arg (paste path)
 }

 async function submitMint(e: React.FormEvent) {
 e.preventDefault();
 const key = apiKey.trim();
 const b = baseUrl.trim();
 if (!key || !b || minting) return;
 const chosen = SCOPES.filter((s) => scopes.has(s));
 const expiresMinutes = expiry.trim() && Number.isFinite(Number(expiry)) ? Number(expiry) : undefined;
 setMinting(true);
 setError(null);
 try {
 const res = await mint({ baseUrl: b, apiKey: key, scopes: chosen, expiresMinutes });
 onConnect({ baseUrl: b, token: res.access_token, authScheme: "bearer" }, res.scopes);
 } catch (err) {
 setError(err instanceof Error ? err.message : "mint failed");
 } finally {
 setApiKey(""); // DE3: discard the API key after the attempt (success OR failure)
 setMinting(false);
 }
 }

 function toggleScope(s: string) {
 setScopes((prev) => {
 const next = new Set(prev);
 if (next.has(s)) next.delete(s);
 else next.add(s);
 return next;
 });
 }

 return (
 <div>
 <label>
 Mode
 <select aria-label="auth-mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
 <option value="paste">Paste a token</option>
 <option value="mint">Mint from API key</option>
 </select>
 </label>

 {mode === "paste" ? (
 <form aria-label="auth" onSubmit={submitPaste}>
 <label>
 Core API URL
 <input
 aria-label="base-url"
 value={baseUrl}
 onChange={(e) => setBaseUrl(e.target.value)}
 placeholder="http://127.0.0.1:8080"
 />
 </label>
 <label>
 Token
 <input
 aria-label="token"
 type="password"
 value={token}
 onChange={(e) => setToken(e.target.value)}
 autoComplete="off"
 />
 </label>
 <label>
 Auth
 <select
 aria-label="auth-scheme"
 value={authScheme}
 onChange={(e) => setAuthScheme(e.target.value as AuthScheme)}
 >
 <option value="api-key">API Key</option>
 <option value="bearer">JWT (Bearer)</option>
 </select>
 </label>
 <button type="submit" disabled={!token.trim() || !baseUrl.trim()}>
 Connect
 </button>
 <p>
 <small>The token is held in memory only — never stored to disk/localStorage.</small>
 </p>
 </form>
 ) : (
 <form aria-label="auth-mint" onSubmit={submitMint}>
 <label>
 Core API URL
 <input
 aria-label="base-url"
 value={baseUrl}
 onChange={(e) => setBaseUrl(e.target.value)}
 placeholder="http://127.0.0.1:8080"
 />
 </label>
 <label>
 API Key
 <input
 aria-label="api-key"
 type="password"
 value={apiKey}
 onChange={(e) => setApiKey(e.target.value)}
 autoComplete="off"
 />
 </label>
 <fieldset>
 <legend>Scopes</legend>
 {SCOPES.map((s) => (
 <label key={s}>
 <input
 type="checkbox"
 aria-label={`scope-${s}`}
 checked={scopes.has(s)}
 onChange={() => toggleScope(s)}
 />
 {s}
 </label>
 ))}
 </fieldset>
 <label>
 Expiry (minutes, optional)
 <input
 aria-label="expires-minutes"
 inputMode="numeric"
 value={expiry}
 onChange={(e) => setExpiry(e.target.value)}
 />
 </label>
 <button type="submit" disabled={!apiKey.trim() || !baseUrl.trim() || minting}>
 Mint &amp; connect
 </button>
 {error && <p data-testid="mint-error">{error}</p>}
 <p>
 <small>
 The API key is used only to mint the token, then discarded. Both are held in memory only.
 </small>
 </p>
 </form>
 )}
 </div>
 );
}
