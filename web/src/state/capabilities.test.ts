import { describe, expect, it } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { CapabilitiesController } from "./capabilities";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 capsRest: unknown = { available: ["agent", "docqa"], all: ["agent", "docqa", "exec"] };
 healthResp: unknown = { status: "degraded", modules: {} };
 capsThrows = false;
 healthThrows = false;
 onEvent(fn: (e: ServerEvent) => void) {
 this.eventL.add(fn);
 return () => this.eventL.delete(fn);
 }
 onState(_fn: (s: ConnState) => void) {
 return () => {};
 }
 async capabilitiesRest() {
 if (this.capsThrows) throw new Error("rest down");
 return this.capsRest;
 }
 async health() {
 if (this.healthThrows) throw new Error("health down");
 return this.healthResp;
 }
 emit(e: ServerEvent) {
 for (const fn of this.eventL) fn(e);
 }
}

function setup() {
 const fake = new FakeClient();
 const ctrl = new CapabilitiesController(fake as unknown as ApiClient);
 return { fake, ctrl };
}

const CAPS = (available: string[]): ServerEvent =>
 ({ event: "capabilities", available }) as unknown as ServerEvent;

describe("CapabilitiesController — normal", () => {
 it("WS capabilities event (no REST yet) → all listed are available", () => {
 const { fake, ctrl } = setup();
 fake.emit(CAPS(["agent", "exec"]));
 expect(ctrl.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "exec", available: true },
 ]);
 });

 it("refresh → caps from `all` with availability from `available`; overall from /health", async () => {
 const { ctrl } = setup();
 await ctrl.refresh();
 expect(ctrl.current.overall).toBe("degraded");
 expect(ctrl.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "docqa", available: true },
 { name: "exec", available: false }, // in `all` but not `available` → unavailable
 ]);
 });

 it("a WS event after refresh updates availability against the known full set", async () => {
 const { fake, ctrl } = setup();
 await ctrl.refresh();
 fake.emit(CAPS(["agent"])); // only agent now available
 expect(ctrl.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "docqa", available: false },
 { name: "exec", available: false },
 ]);
 });
});

describe("CapabilitiesController — error / robustness", () => {
 it("REST /capabilities throwing → keeps WS-derived availability (no crash)", async () => {
 const { fake, ctrl } = setup();
 fake.emit(CAPS(["agent", "browser"]));
 fake.capsThrows = true;
 fake.healthThrows = true;
 await expect(ctrl.refresh()).resolves.toBeUndefined();
 expect(ctrl.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "browser", available: true },
 ]);
 });

 it("malformed /capabilities (available not an array) → treated as none, no crash", async () => {
 const { fake, ctrl } = setup();
 fake.capsRest = { available: "nope", all: 123 };
 await ctrl.refresh();
 expect(ctrl.current.caps).toEqual([]); // unknown → absent
 });

 it("malformed /health (no status) → overall stays null, no crash", async () => {
 const { fake, ctrl } = setup();
 fake.healthResp = { modules: {} };
 await ctrl.refresh();
 expect(ctrl.current.overall).toBeNull();
 });

 it("non-string entries in the WS available list are dropped (no crash)", () => {
 const { fake, ctrl } = setup();
 fake.emit({ event: "capabilities", available: ["agent", 42, null, "exec"] } as unknown as ServerEvent);
 expect(ctrl.current.caps).toEqual([
 { name: "agent", available: true },
 { name: "exec", available: true },
 ]);
 });
});
