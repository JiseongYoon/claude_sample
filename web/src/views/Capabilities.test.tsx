import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ApiClient, ConnState } from "../api/client";
import type { ServerEvent } from "../api/types";
import { Capabilities } from "./Capabilities";

class FakeClient {
 eventL = new Set<(e: ServerEvent) => void>();
 capsRest: unknown = { available: ["agent", "docqa"], all: ["agent", "docqa", "exec"] };
 healthResp: unknown = { status: "degraded", modules: {} };
 onEvent(fn: (e: ServerEvent) => void) {
 this.eventL.add(fn);
 return () => this.eventL.delete(fn);
 }
 onState(_fn: (s: ConnState) => void) {
 return () => {};
 }
 async capabilitiesRest() {
 return this.capsRest;
 }
 async health() {
 return this.healthResp;
 }
 emit(e: ServerEvent) {
 act(() => {
 for (const fn of this.eventL) fn(e);
 });
 }
}

afterEach(() => cleanup());

describe("Capabilities view", () => {
 it("renders availability from refresh: an absent capability shows 'unavailable'", async () => {
 const fake = new FakeClient();
 render(<Capabilities client={fake as unknown as ApiClient} />);
 await waitFor(() => expect(screen.getAllByTestId("cap").length).toBe(3));
 const items = screen.getAllByTestId("cap");
 const byName = (name: string) => items.find((n) => (n as HTMLElement).dataset.cap === name)!;
 expect(byName("agent").dataset.available).toBe("true");
 expect(byName("docqa").dataset.available).toBe("true");
 expect(byName("exec").dataset.available).toBe("false");
 expect(within(byName("exec")).getByText(/unavailable/)).toBeInTheDocument();
 expect(screen.getByTestId("overall")).toHaveTextContent("degraded");
 });

 it("a WS capabilities event updates availability live", async () => {
 const fake = new FakeClient();
 render(<Capabilities client={fake as unknown as ApiClient} />);
 await waitFor(() => expect(screen.getAllByTestId("cap").length).toBe(3));
 fake.emit({ event: "capabilities", available: ["agent"] } as unknown as ServerEvent);
 const items = screen.getAllByTestId("cap");
 const byName = (name: string) => items.find((n) => (n as HTMLElement).dataset.cap === name)!;
 expect(byName("docqa").dataset.available).toBe("false");
 expect(byName("exec").dataset.available).toBe("false");
 });
});
