import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Citations } from "./Citations";

afterEach(cleanup);

describe("Citations", () => {
 it("renders extracted sources as escaped text", () => {
 render(<Citations text="answer grounded in https://docs.example/a and [source#1]" />);
 expect(screen.getByTestId("citations")).toBeInTheDocument();
 const cites = screen.getAllByTestId("citation").map((n) => n.textContent);
 expect(cites).toContain("https://docs.example/a");
 expect(cites).toContain("[source#1]");
 });

 it("renders nothing when the answer has no sources", () => {
 render(<Citations text="a plain answer" />);
 expect(screen.queryByTestId("citations")).toBeNull();
 });

 it("a hostile source string is rendered as inert text, not a live link/element", () => {
 render(<Citations text={'see https://evil.example/"><img src=x onerror=alert(1)>'} />);
 // no <img> element injected; the URL fragment is plain text inside the citation <li>
 expect(document.querySelector("img")).toBeNull();
 expect(screen.getByTestId("citations").querySelector("a")).toBeNull();
 });
});
