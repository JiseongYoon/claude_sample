import { describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../api/client";
import { ModelController } from "./model";

class FakeClient {
 statusResp: unknown = { state: "unloaded", loaded_file: null, serving: false };
 paramsResp: unknown = { params: { temperature: 0.7, top_k: 40 } };
 filesResp: unknown = { files: ["a.gguf", "b.gguf"] };
 healthResp: unknown = { status: "ok", modules: { "model-manager": "ok", "llm-serving": "degraded" } };
 loadStatus = 202;
 switchStatus = 202;
 unloadStatus = 200;
 paramsStatus = 200;
 loadCalls: (string | undefined)[] = [];
 switchCalls: string[] = [];
 unloadCalls = 0;
 setParamsCalls: Record<string, number>[] = [];
 throwReads = false;

 async modelStatus() {
 if (this.throwReads) throw new Error("x");
 return this.statusResp;
 }
 async getParams() {
 if (this.throwReads) throw new Error("x");
 return this.paramsResp;
 }
 async listModelFiles() {
 if (this.throwReads) throw new Error("x");
 return this.filesResp;
 }
 async health() {
 if (this.throwReads) throw new Error("x");
 return this.healthResp;
 }
 async modelLoad(f?: string) {
 this.loadCalls.push(f);
 return { status: this.loadStatus, ok: this.loadStatus < 400, data: {} };
 }
 async modelSwitch(f: string) {
 this.switchCalls.push(f);
 return { status: this.switchStatus, ok: this.switchStatus < 400, data: {} };
 }
 async modelUnload() {
 this.unloadCalls += 1;
 return { status: this.unloadStatus, ok: true, data: {} };
 }
 async setParams(p: Record<string, number>) {
 this.setParamsCalls.push(p);
 return { status: this.paramsStatus, ok: this.paramsStatus < 400, data: {} };
 }
}

function setup() {
 const fake = new FakeClient();
 return { fake, ctrl: new ModelController(fake as unknown as ApiClient) };
}

describe("ModelController — reads", () => {
 it("refresh populates status, params, files, sorted per-module health", async () => {
 const { ctrl } = setup();
 await ctrl.refresh();
 const s = ctrl.current;
 expect(s.status).toEqual({ state: "unloaded", loaded_file: null, serving: false });
 expect(s.params).toEqual({ temperature: 0.7, top_k: 40 });
 expect(s.files).toEqual(["a.gguf", "b.gguf"]);
 expect(s.modules).toEqual([
 { name: "llm-serving", status: "degraded" },
 { name: "model-manager", status: "ok" },
 ]);
 });

 it("malformed reads → safe (unknown), no crash", async () => {
 const { fake, ctrl } = setup();
 fake.statusResp = "garbage";
 fake.paramsResp = {};
 fake.filesResp = null;
 fake.healthResp = { status: "ok" }; // no modules key
 await ctrl.refresh();
 const s = ctrl.current;
 expect(s.status).toBeNull();
 expect(s.params).toEqual({});
 expect(s.files).toEqual([]);
 expect(s.modules).toEqual([]);
 });

 it("read failures are swallowed (each call independent)", async () => {
 const { fake, ctrl } = setup();
 fake.throwReads = true;
 await ctrl.refresh();
 expect(ctrl.current.status).toBeNull();
 expect(ctrl.current.files).toEqual([]);
 });

 it("/health module map accepts the detailed shape {status,detail}", async () => {
 const { fake, ctrl } = setup();
 fake.healthResp = { status: "ok", modules: { exec: { status: "down", detail: "no docker" } } };
 await ctrl.refresh();
 expect(ctrl.current.modules).toEqual([{ name: "exec", status: "down" }]);
 });
});

describe("ModelController — admin actions (server-authoritative; surfaces the verdict)", () => {
 it("load: 202 → 'launching' notice + the file is sent + refresh runs", async () => {
 const { fake, ctrl } = setup();
 await ctrl.load("a.gguf");
 expect(fake.loadCalls).toEqual(["a.gguf"]);
 expect(ctrl.current.notice).toBe("load: launching…");
 expect(ctrl.current.busy).toBe(false);
 });

 it("403 → 'requires model:admin scope' notice (no bypass)", async () => {
 const { fake, ctrl } = setup();
 fake.loadStatus = 403;
 await ctrl.load("a.gguf");
 expect(ctrl.current.notice).toBe("load: requires model:admin scope");
 });

 it("switch 409 (busy) and params 409 (no model) surface a clear notice", async () => {
 const { fake, ctrl } = setup();
 fake.switchStatus = 409;
 await ctrl.switchModel("b.gguf");
 expect(fake.switchCalls).toEqual(["b.gguf"]);
 expect(ctrl.current.notice).toContain("model busy");
 fake.paramsStatus = 409;
 await ctrl.setParams({ temperature: 0.5 });
 expect(fake.setParamsCalls).toEqual([{ temperature: 0.5 }]);
 expect(ctrl.current.notice).toContain("model busy / no model loaded");
 });

 it("unload 200 → 'ready' notice", async () => {
 const { fake, ctrl } = setup();
 await ctrl.unload();
 expect(fake.unloadCalls).toBe(1);
 expect(ctrl.current.notice).toBe("unload: ready");
 });

 it("single-flight: a second action while busy is dropped", async () => {
 const { fake, ctrl } = setup();
 const p1 = ctrl.load("a.gguf");
 const p2 = ctrl.load("b.gguf"); // busy → dropped
 await Promise.all([p1, p2]);
 expect(fake.loadCalls).toEqual(["a.gguf"]);
 });

 it("an action whose request throws → 'request failed', not a crash", async () => {
 const { fake, ctrl } = setup();
 fake.modelUnload = async () => {
 throw new Error("network");
 };
 await ctrl.unload();
 expect(ctrl.current.notice).toBe("unload: request failed");
 expect(ctrl.current.busy).toBe(false);
 });
});

describe("ModelController — load progress (ISSUE-002, display-only)", () => {
 // injected clock + a huge poll interval so the real timer never fires during these synchronous traces.
 const make = (fake: FakeClient, now: () => number) =>
 new ModelController(fake as unknown as ApiClient, now, 1e9);

 it("stamps loadStartedAt (injected clock) on entering a loading state", async () => {
 const fake = new FakeClient();
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 const ctrl = make(fake, () => 1000);
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(1000);
 ctrl.dispose();
 });

 it("keeps the original loadStartedAt across refreshes while still loading", async () => {
 const fake = new FakeClient();
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 let t = 5000;
 const ctrl = make(fake, () => t);
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(5000);
 t = 9000; // clock advances, but still loading → stamp unchanged
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(5000);
 ctrl.dispose();
 });

 it("clears loadStartedAt when the engine reaches ready / error / unloaded", async () => {
 const fake = new FakeClient();
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 const ctrl = make(fake, () => 1);
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(1);
 for (const s of ["ready", "error", "unloaded"]) {
 fake.statusResp = { state: s, loaded_file: null, serving: false };
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBeNull();
 }
 ctrl.dispose();
 });

 it("re-stamps on a fresh loading transition after it cleared (loading→ready→loading)", async () => {
 const fake = new FakeClient();
 let t = 100;
 const ctrl = make(fake, () => t);
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(100);
 fake.statusResp = { state: "ready", loaded_file: null, serving: false };
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBeNull();
 t = 777;
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBe(777);
 ctrl.dispose();
 });

 it("malformed / non-loading status → loadStartedAt null (no spurious progress)", async () => {
 const fake = new FakeClient();
 fake.statusResp = "garbage";
 const ctrl = make(fake, () => 1);
 await ctrl.refresh();
 expect(ctrl.current.loadStartedAt).toBeNull();
 ctrl.dispose();
 });

 it("polls /model/status WHILE loading and stops the moment it reaches ready", async () => {
 vi.useFakeTimers();
 try {
 const fake = new FakeClient();
 let calls = 0;
 fake.modelStatus = async () => {
 calls += 1;
 return fake.statusResp;
 };
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 const ctrl = new ModelController(fake as unknown as ApiClient, () => 1, 1000);
 await ctrl.refresh(); // initial read → loading → poll scheduled
 expect(calls).toBe(1);
 await vi.advanceTimersByTimeAsync(1000); // poll fires → refresh
 expect(calls).toBe(2);
 fake.statusResp = { state: "ready", loaded_file: null, serving: false };
 await vi.advanceTimersByTimeAsync(1000); // poll fires → sees ready → stops + clears stamp
 expect(calls).toBe(3);
 expect(ctrl.current.loadStartedAt).toBeNull();
 await vi.advanceTimersByTimeAsync(5000); // no further polling
 expect(calls).toBe(3);
 ctrl.dispose();
 } finally {
 vi.useRealTimers();
 }
 });

 it("dispose() stops the loading poll (no leaked interval)", async () => {
 vi.useFakeTimers();
 try {
 const fake = new FakeClient();
 let calls = 0;
 fake.modelStatus = async () => {
 calls += 1;
 return fake.statusResp;
 };
 fake.statusResp = { state: "loading", loaded_file: null, serving: false };
 const ctrl = new ModelController(fake as unknown as ApiClient, () => 1, 1000);
 await ctrl.refresh();
 expect(calls).toBe(1);
 ctrl.dispose();
 await vi.advanceTimersByTimeAsync(5000);
 expect(calls).toBe(1); // disposed → no poll
 } finally {
 vi.useRealTimers();
 }
 });
});
