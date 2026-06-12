import { describe, expect, it } from "vitest";
import { extractSources } from "./citations";

describe("extractSources", () => {
 it("extracts http(s) URLs, trimming trailing punctuation", () => {
 expect(extractSources("See https://example.com/a and http://b.org/x.")).toEqual([
 "https://example.com/a",
 "http://b.org/x",
 ]);
 });

 it("extracts DocQA-style [source…] / […#N] reference tokens", () => {
 const out = extractSources("Per [source#2] and [source: notes.txt], the answer is 42.");
 expect(out).toContain("[source#2]");
 expect(out).toContain("[source: notes.txt]");
 });

 it("dedups repeated sources", () => {
 expect(extractSources("https://x.com then https://x.com again")).toEqual(["https://x.com"]);
 });

 it("returns [] when there are no sources (graceful absence)", () => {
 expect(extractSources("just a plain answer with no links")).toEqual([]);
 expect(extractSources("")).toEqual([]);
 expect(extractSources(undefined as unknown as string)).toEqual([]);
 });

 it("does not treat arbitrary brackets as citations", () => {
 expect(extractSources("an array like [1, 2, 3] is not a source")).toEqual([]);
 });
});
