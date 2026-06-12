import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

// Mock the API client so App can reach the `connected` workspace without a real WebSocket: connect()
// synchronously drives the stored onState listener to "connected"; the read methods the mounted
// controllers call on mount (status/params/files/health/capabilities) return benign shapes.
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
 return { state: "ready", loaded_file: "gemma.gguf", serving: true };
 }
 async getParams() {
 return { params: {} };
 }
 async listModelFiles() {
 return { files: [] };
 }
 async health() {
 return { modules: {} };
 }
 async capabilitiesRest() {
 return { available: [], all: [] };
 }
 static mintToken = (actual.ApiClient as { mintToken: unknown }).mintToken;
 }
 return { ...actual, ApiClient: FakeApiClient };
});

// imported AFTER the mock is registered
const { App } = await import("./App");

afterEach(cleanup);

async function connect() {
 render(<App />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("token"), "a-token");
 await user.click(screen.getByRole("button", { name: "Connect" }));
}

describe("App layout (/ ISSUE-004) — ModelPanel relocated out of the sidebar", () => {
 it("renders the model panel and the capabilities sidebar once connected", async () => {
 await connect();
 expect(screen.getByRole("region", { name: "model-panel" })).toBeInTheDocument();
 expect(screen.getByRole("region", { name: "capabilities" })).toBeInTheDocument();
 });

 it("the model panel is NOT stacked inside the capabilities sidebar (relocated to its own full-width row)", async () => {
 await connect();
 const capabilities = screen.getByRole("region", { name: "capabilities" });
 const modelPanel = screen.getByRole("region", { name: "model-panel" });
 // pre-fix the panel was a sibling of Capabilities INSIDE the .sidebar column → sidebar.contains(panel)
 // was true. After the relocation the panel lives in the workspace's own row, outside the sidebar.
 const sidebar = capabilities.parentElement as HTMLElement;
 expect(sidebar.contains(modelPanel)).toBe(false);
 });

 it("the model panel precedes the chat workspace (top control row)", async () => {
 await connect();
 const modelPanel = screen.getByRole("region", { name: "model-panel" });
 const chatTabs = screen.getByRole("region", { name: "chat-tabs" });
 // document order: model panel comes before the chat tabs in the DOM
 expect(modelPanel.compareDocumentPosition(chatTabs) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
 });

 it("relocation grants no authority — the panel still surfaces its server-gated nature (zero authority)", async () => {
 // canAdmin resolves to true for an API-key paste connect; the panel mounts with its admin controls.
 // The relocation is presentation-only: the model-admin actions remain server-gated regardless.
 await connect();
 const modelPanel = screen.getByRole("region", { name: "model-panel" });
 // the load/switch/unload controls exist in the relocated panel (behaviour intact)
 expect(within(modelPanel).getByRole("button", { name: "Load" })).toBeInTheDocument();
 });
});
