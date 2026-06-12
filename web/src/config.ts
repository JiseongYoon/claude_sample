// Runtime config — the core API origin + the auth token. Held IN MEMORY only (never localStorage /
// sessionStorage / disk), so a token can't be exfiltrated from storage by XSS. The operator supplies
// both at runtime (entered in the UI,); `VITE_*` env vars are convenience defaults for dev only.

import type { AuthScheme } from "./api/client";

export interface RuntimeConfig {
 baseUrl: string;
 token: string;
 authScheme: AuthScheme;
}

let current: RuntimeConfig = {
 // default = the API GATEWAY origin (api_port 8080), NOT the model server (serve_port 8000).
 baseUrl: (import.meta.env?.VITE_CORE_BASE_URL as string | undefined) ?? "http://127.0.0.1:8080",
 token: "", // never seeded from storage; operator enters it
 authScheme: "api-key",
};

export function getConfig(): RuntimeConfig {
 return current;
}

export function setConfig(next: Partial<RuntimeConfig>): RuntimeConfig {
 current = { ...current, ...next };
 return current;
}

export function clearToken(): void {
 current = { ...current, token: "" };
}
