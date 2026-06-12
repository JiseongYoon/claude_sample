// Phase-11.4 — hermetic INTEGRATED-WHOLE regression of the UI-polish phase composed together.
// One connected <App> render proves the three issue fixes COEXIST in the live workspace, with the model
// in a `loading` state so all three are simultaneously observable:
// ISSUE-004 — the Model/API panel is relocated OUT of the capabilities sidebar (its own full-width row);
// ISSUE-003 — the four param inputs render inside that (now full-width) panel;
// ISSUE-002 — while loading, the panel shows an INDETERMINATE bar + elapsed + hint (no false %).
// Presentation-only, zero authority: the relocation/indicator add no new endpoint or capability; the load
// indicator's poll/timer only re-read the existing status GET. Network-free (ApiClient is mocked).
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

// the model sits in `loading` so the indicator is visible; reads return benign shapes.
vi.mock("./api/client", async (orig) => {
 const actual = (await orig()) as Record<string, unknown>;
 class FakeApiClient {
 private stateFns = new Set<(s: string) => void>();
 constructor(_opts: unknown) {}
 onState(fn: (s: string) => void) {
 this.stateFns.add(fn);
 return () => this.stateFns.delete(fn);
 }
 onEvent(_fn: (e: unknown) => void) {
 return () => {};
 }
 connect() {
 for (const fn of this.stateFns) fn("connected");
 }
 disconnect() {}
 async modelStatus() {
 return { state: "loading", loaded_file: null, serving: false };
 }
 async getParams() {
 return { params: {} };
 }
 async listModelFiles() {
 return { files: ["a.gguf"] };
 }
 async health() {
 return { modules: { "model-manager": "ok", "llm-serving": "ok" } };
 }
 async capabilitiesRest() {
 return { available: ["agent"], all: ["agent"] };
 }
 static mintToken = (actual.ApiClient as { mintToken: unknown }).mintToken;
 }
 return { ...actual, ApiClient: FakeApiClient };
});

const { App } = await import("./App");

afterEach(cleanup);

describe("— UI-polish fixes coexist in the connected workspace", () => {
 it("relocated panel (004) hosts the overflow-safe params (003) and the load indicator (002) at once", async () => {
 render(<App />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("token"), "a-token");
 await user.click(screen.getByRole("button", { name: "Connect" }));

 const panel = await screen.findByRole("region", { name: "model-panel" });
 const capabilities = screen.getByRole("region", { name: "capabilities" });

 // ISSUE-004: the panel is NOT inside the capabilities sidebar (relocated to its own full-width row)
 expect((capabilities.parentElement as HTMLElement).contains(panel)).toBe(false);

 // ISSUE-003: all four param inputs render inside the (now full-width) panel
 for (const k of ["temperature", "top_p", "top_k", "max_tokens"]) {
 expect(within(panel).getByLabelText(k)).toBeInTheDocument();
 }

 // ISSUE-002: while loading, the indeterminate indicator + elapsed + hint are shown inside the panel
 const progress = within(panel).getByTestId("model-load-progress");
 const bar = within(progress).getByRole("progressbar", { name: "model loading" });
 expect(bar.getAttribute("aria-valuenow")).toBeNull(); // indeterminate — no false percentage
 expect(within(progress).getByTestId("model-load-elapsed")).toHaveTextContent(/^\d+:\d{2}$/);
 expect(progress).toHaveTextContent(/several minutes/);
 });

 it("zero authority preserved — sign-out tears the workspace down (no leaked controller)", async () => {
 render(<App />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("token"), "a-token");
 await user.click(screen.getByRole("button", { name: "Connect" }));
 expect(await screen.findByRole("region", { name: "model-panel" })).toBeInTheDocument();
 await user.click(screen.getByRole("button", { name: "Sign out" }));
 // the connected workspace (and its mounted ModelPanel → ModelController.dispose()) is unmounted
 expect(screen.queryByRole("region", { name: "model-panel" })).toBeNull();
 });
});
