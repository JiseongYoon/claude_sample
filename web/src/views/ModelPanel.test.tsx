import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../api/client";
import { ModelPanel } from "./ModelPanel";

class FakeClient {
 statusResp: unknown = { state: "ready", loaded_file: "gemma-q8.gguf", serving: true };
 paramsResp: unknown = { params: { temperature: 0.7 } };
 filesResp: unknown = { files: ["a.gguf", "b.gguf"] };
 healthResp: unknown = { status: "ok", modules: { "model-manager": "ok", exec: "down" } };
 loadStatus = 202;
 loadCalls: (string | undefined)[] = [];
 async modelStatus() {
 return this.statusResp;
 }
 async getParams() {
 return this.paramsResp;
 }
 async listModelFiles() {
 return this.filesResp;
 }
 async health() {
 return this.healthResp;
 }
 async modelLoad(f?: string) {
 this.loadCalls.push(f);
 return { status: this.loadStatus, ok: this.loadStatus < 400, data: {} };
 }
 async modelSwitch() {
 return { status: 200, ok: true, data: {} };
 }
 async modelUnload() {
 return { status: 200, ok: true, data: {} };
 }
 async setParams() {
 return { status: 200, ok: true, data: {} };
 }
}

const asClient = (f: FakeClient) => f as unknown as ApiClient;
afterEach(cleanup);

describe("ModelPanel", () => {
 it("renders status, the file picker, and per-module health after refresh", async () => {
 const fake = new FakeClient();
 render(<ModelPanel client={asClient(fake)} />);
 expect(await screen.findByTestId("model-state")).toHaveTextContent("ready");
 // file picker options
 expect(await screen.findByRole("option", { name: "a.gguf" })).toBeInTheDocument();
 expect(screen.getByRole("option", { name: "b.gguf" })).toBeInTheDocument();
 // module health
 const mods = await screen.findAllByTestId("module-health");
 const byName = (n: string) => mods.find((m) => (m as HTMLElement).dataset.module === n)!;
 expect(byName("exec").dataset.status).toBe("down");
 expect(within(byName("model-manager")).getByText("ok")).toBeInTheDocument();
 });

 it("canAdmin=false → admin-locked message + admin controls disabled", async () => {
 const fake = new FakeClient();
 render(<ModelPanel client={asClient(fake)} canAdmin={false} />);
 expect(await screen.findByTestId("admin-locked")).toBeInTheDocument();
 expect(screen.getByRole("button", { name: "Load" })).toBeDisabled();
 expect(screen.getByRole("button", { name: "Unload" })).toBeDisabled();
 expect(screen.getByLabelText("temperature")).toBeDisabled();
 });

 it("canAdmin=true → controls enabled; Load calls the client once", async () => {
 const fake = new FakeClient();
 render(<ModelPanel client={asClient(fake)} canAdmin={true} />);
 const load = await screen.findByRole("button", { name: "Load" });
 expect(load).toBeEnabled();
 expect(screen.queryByTestId("admin-locked")).toBeNull();
 await userEvent.click(load);
 expect(fake.loadCalls.length).toBe(1);
 });

 it("canAdmin undefined → controls enabled and a 403 is SURFACED (no client bypass)", async () => {
 const fake = new FakeClient();
 fake.loadStatus = 403;
 render(<ModelPanel client={asClient(fake)} />);
 const load = await screen.findByRole("button", { name: "Load" });
 expect(load).toBeEnabled(); // unknown scope → attempt allowed; server decides
 await userEvent.click(load);
 expect(await screen.findByTestId("model-notice")).toHaveTextContent("requires model:admin scope");
 });
});

describe("ModelPanel — load progress indicator (ISSUE-002, frontend, display-only)", () => {
 it("while loading: shows an INDETERMINATE progress bar + an elapsed counter + a hint", async () => {
 const fake = new FakeClient();
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 render(<ModelPanel client={asClient(fake)} />);
 const progress = await screen.findByTestId("model-load-progress");
 const bar = within(progress).getByRole("progressbar", { name: "model loading" });
 expect(bar).toBeInTheDocument();
 expect(bar.getAttribute("aria-valuenow")).toBeNull(); // indeterminate — no false percentage
 expect(screen.getByTestId("model-load-elapsed")).toHaveTextContent(/^\d+:\d{2}$/);
 expect(progress).toHaveTextContent(/several minutes/);
 });

 it("ready (non-loading) → no progress indicator", async () => {
 const fake = new FakeClient(); // default state is "ready"
 render(<ModelPanel client={asClient(fake)} />);
 expect(await screen.findByTestId("model-state")).toHaveTextContent("ready");
 expect(screen.queryByTestId("model-load-progress")).toBeNull();
 });

 it("the elapsed counter ticks up while loading (fake timers)", async () => {
 vi.useFakeTimers();
 try {
 vi.setSystemTime(0);
 const fake = new FakeClient();
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 render(<ModelPanel client={asClient(fake)} />);
 // flush the async mount refresh (the loadStartedAt is stamped at t=0)
 await act(async () => {
 await vi.advanceTimersByTimeAsync(0);
 });
 expect(screen.getByTestId("model-load-elapsed")).toHaveTextContent("0:00");
 // advancing the fake clock by 65s also advances Date.now() (fake timers tie them); the 1s display
 // interval fires repeatedly and the counter reads 1:05 (minute rollover) — proves it ticks.
 await act(async () => {
 await vi.advanceTimersByTimeAsync(65_000);
 });
 expect(screen.getByTestId("model-load-elapsed")).toHaveTextContent("1:05");
 } finally {
 vi.useRealTimers();
 }
 });
});
