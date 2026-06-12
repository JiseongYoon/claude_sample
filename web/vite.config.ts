/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standalone frontend (DC) — its own dev server / build; NOT served by the FastAPI core.
// The core API origin is operator-configured at runtime (see src/config.ts) and must be in the
// gateway's CORS allowlist (deny-by-default) — see README.
export default defineConfig({
 plugins: [react()],
 server: { port: 5173 },
 test: {
 environment: "jsdom",
 globals: true,
 setupFiles: ["./src/test/setup.ts"],
 include: ["src/**/*.test.{ts,tsx}"],
 },
});
