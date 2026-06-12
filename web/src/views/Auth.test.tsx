import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Auth } from "./Auth";

afterEach(() => cleanup());

describe("Auth view", () => {
 it("submitting calls onConnect with the entered base-url, token, and scheme", async () => {
 const onConnect = vi.fn();
 render(<Auth onConnect={onConnect} />);
 const user = userEvent.setup();
 const url = screen.getByLabelText("base-url");
 await user.clear(url);
 await user.type(url, "http://core:9000");
 await user.type(screen.getByLabelText("token"), "tok-abc");
 await user.click(screen.getByRole("button", { name: "Connect" }));
 expect(onConnect).toHaveBeenCalledWith({
 baseUrl: "http://core:9000",
 token: "tok-abc",
 authScheme: "api-key",
 });
 });

 it("can select the bearer (JWT) scheme", async () => {
 const onConnect = vi.fn();
 render(<Auth onConnect={onConnect} />);
 const user = userEvent.setup();
 await user.type(screen.getByLabelText("token"), "jwt-xyz");
 await user.selectOptions(screen.getByLabelText("auth-scheme"), "bearer");
 await user.click(screen.getByRole("button", { name: "Connect" }));
 expect(onConnect).toHaveBeenCalledWith(expect.objectContaining({ authScheme: "bearer" }));
 });

 it("Connect is disabled until a token is entered", async () => {
 render(<Auth onConnect={vi.fn()} />);
 const user = userEvent.setup();
 const connect = screen.getByRole("button", { name: "Connect" });
 expect(connect).toBeDisabled();
 await user.type(screen.getByLabelText("token"), "t");
 expect(connect).toBeEnabled();
 });

 it("the token field is a password input and the token is NEVER written to storage", async () => {
 render(<Auth onConnect={vi.fn()} />);
 const user = userEvent.setup();
 const token = screen.getByLabelText("token");
 expect(token).toHaveAttribute("type", "password");
 await user.type(token, "super-secret-token");
 await user.click(screen.getByRole("button", { name: "Connect" }));
 expect(localStorage.length).toBe(0);
 expect(sessionStorage.length).toBe(0);
 });
});

describe("Auth view — mint mode ", () => {
 async function intoMint() {
 const user = userEvent.setup();
 await user.selectOptions(screen.getByLabelText("auth-mode"), "mint");
 return user;
 }

 it("mints with default scopes (model:admin OFF) → connects bearer + passes granted scopes", async () => {
 const onConnect = vi.fn();
 const mint = vi.fn(async (a) => ({ access_token: "JWT-123", scopes: a.scopes, expires_in: 3600 }));
 render(<Auth onConnect={onConnect} mint={mint} />);
 const user = await intoMint();
 await user.type(screen.getByLabelText("api-key"), "APIKEY-SECRET");
 await user.click(screen.getByRole("button", { name: /Mint/ }));
 expect(mint).toHaveBeenCalledWith(
 expect.objectContaining({
 apiKey: "APIKEY-SECRET",
 scopes: ["read", "invoke", "agent:run", "approve", "ingest"], // `ingest` on by default
 }),
 );
 expect(onConnect).toHaveBeenCalledWith(
 expect.objectContaining({ token: "JWT-123", authScheme: "bearer" }),
 ["read", "invoke", "agent:run", "approve", "ingest"],
 );
 });

 it("model:admin is opt-in (off by default; can be enabled)", async () => {
 const onConnect = vi.fn();
 const mint = vi.fn(async (a) => ({ access_token: "JWT-X", scopes: a.scopes, expires_in: 60 }));
 render(<Auth onConnect={onConnect} mint={mint} />);
 const user = await intoMint();
 expect(screen.getByLabelText("scope-model:admin")).not.toBeChecked();
 await user.click(screen.getByLabelText("scope-model:admin"));
 await user.type(screen.getByLabelText("api-key"), "K");
 await user.click(screen.getByRole("button", { name: /Mint/ }));
 expect(mint.mock.calls[0][0].scopes).toContain("model:admin");
 });

 it("the API key is DISCARDED after a mint (success), and never written to storage", async () => {
 const mint = vi.fn(async (a) => ({ access_token: "JWT", scopes: a.scopes, expires_in: 60 }));
 render(<Auth onConnect={vi.fn()} mint={mint} />);
 const user = await intoMint();
 const key = screen.getByLabelText("api-key") as HTMLInputElement;
 expect(key).toHaveAttribute("type", "password");
 await user.type(key, "APIKEY-SENSITIVE");
 await user.click(screen.getByRole("button", { name: /Mint/ }));
 expect(key.value).toBe(""); // discarded (DE3)
 expect(localStorage.length).toBe(0);
 expect(sessionStorage.length).toBe(0);
 });

 it("a mint failure surfaces an error, does NOT connect, and still discards the API key", async () => {
 const onConnect = vi.fn();
 const mint = vi.fn(async () => {
 throw new Error("mint failed: 403");
 });
 render(<Auth onConnect={onConnect} mint={mint} />);
 const user = await intoMint();
 await user.type(screen.getByLabelText("api-key"), "BAD-KEY");
 await user.click(screen.getByRole("button", { name: /Mint/ }));
 expect(await screen.findByTestId("mint-error")).toHaveTextContent("403");
 expect(onConnect).not.toHaveBeenCalled();
 expect((screen.getByLabelText("api-key") as HTMLInputElement).value).toBe("");
 });
});
