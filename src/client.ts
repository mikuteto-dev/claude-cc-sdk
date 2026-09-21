import Anthropic from "@anthropic-ai/sdk";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { AuthProvider } from "./auth.js";

export type Msg = Anthropic.Messages.MessageParam;
export type Tool = Anthropic.Messages.Tool;
export type Response = Anthropic.Messages.Message;
export type StreamEvent = Anthropic.Messages.RawMessageStreamEvent;

export interface Opts {
  auth?: { token?: string; apiKey?: string; credsPath?: string };
  model?: string;
  maxRetries?: number;
  baseURL?: string;
  betas?: string[];
  // Match the interactive `claude` entrypoint by default. Set false for `--print`/sdk-cli.
  interactive?: boolean;
}

export interface Send {
  model?: string;
  system?: string | Anthropic.Messages.TextBlockParam[];
  messages: Msg[];
  maxTokens?: number;
  thinking?: boolean | "adaptive" | { type: "enabled"; budget_tokens: number };
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
  tools?: Tool[];
  toolChoice?: Anthropic.Messages.ToolChoice;
  temperature?: number;
  stop?: string[];
  speed?: "fast";
  ctx1m?: boolean;
  redact?: boolean;
  cache?: boolean;
  betas?: string[];
  signal?: AbortSignal;
  timeout?: number;
}

// Keep these values aligned with the installed Claude Code binary.
const VER = "2.1.278";
const B = {
  cc: "claude-code-20250219",     oauth: "oauth-2025-04-20",
  isp: "interleaved-thinking-2025-05-14", redact: "redact-thinking-2026-02-12",
  ttc: "thinking-token-count-2026-05-13", ctx: "context-management-2025-06-27",
  cache: "prompt-caching-scope-2026-01-05", midSys: "mid-conversation-system-2026-04-07",
  advisor: "advisor-tool-2026-03-01", effort: "effort-2025-11-24",
  extTtl: "extended-cache-ttl-2025-04-11",
  fast: "fast-mode-2026-02-01",   ctx1m: "context-1m-2025-08-07",
} as const;

// Model capability gates, mirroring CC's bundled catalog. Unknown names fall
// through to the permissive branch, same as CC's first-party defaults.
const NON_ADAPTIVE = ["claude-3-", "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5",
  "claude-sonnet-4-0", "claude-sonnet-4-5", "claude-haiku-4-5"];
const MIDCONV = ["claude-sonnet-5", "claude-opus-4-8", "claude-opus-5", "claude-fable-5",
  "claude-mythos-5-1"];
const EFFORT = ["claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-4-6", "claude-opus-4-7",
  "claude-opus-4-8", "claude-opus-5", "claude-fable-5", "claude-mythos-5-1"];
const OUT64K = ["claude-sonnet-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
  "claude-opus-5", "claude-fable-5", "claude-mythos-5"];
const OUT32K = ["claude-haiku-4-5", "claude-sonnet-4-0", "claude-sonnet-4-5", "claude-sonnet-4-6",
  "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5", "claude-3-7-sonnet"];

const has = (m: string, list: string[]) => list.some(p => m.includes(p));
const adaptiveCapable = (m: string) => !has(m, NON_ADAPTIVE);
const thinkCapable = (m: string) => !m.includes("claude-3-");
const midConv = (m: string) => has(m, MIDCONV);
const effortCapable = (m: string) => has(m, EFFORT);
const effortDefault = (m: string) => m.includes("claude-opus-4-7") ? "xhigh" : "high";

function maxOut(m: string): number {
  if (has(m, OUT64K)) return 64000;
  if (has(m, OUT32K)) return 32000;
  if (m.includes("claude-3-opus") || m.includes("claude-3-haiku")) return 4096;
  if (m.includes("claude-3-")) return 8192;
  return 64000;
}

interface CcConfig {
  userID?: string;
  oauthAccount?: { accountUuid?: string };
}

function readCcConfig(): CcConfig {
  const customDir = process.env.CLAUDE_CONFIG_DIR;
  const paths = [
    ...(customDir ? [join(customDir, ".claude.json")] : []),
    join(homedir(), ".claude.json"),
  ];
  for (const path of [...new Set(paths)]) {
    try {
      const value: unknown = JSON.parse(readFileSync(path, "utf-8"));
      if (value && typeof value === "object") return value as CcConfig;
    } catch { /* try the next Claude Code state location */ }
  }
  return {};
}

// CC reads ANTHROPIC_CUSTOM_HEADERS as "key:val,key:val"
function envHeaders(): Record<string, string> {
  const raw = process.env.ANTHROPIC_CUSTOM_HEADERS;
  if (!raw) return {};
  const h: Record<string, string> = {};
  for (const p of raw.split(",")) { const i = p.indexOf(":"); if (i > 0) h[p.slice(0, i).trim()] = p.slice(i + 1).trim(); }
  return h;
}

// CC puts an ephemeral cache breakpoint on its own system blocks (never on
// user message content). Subscribed OAuth accounts get the 1h TTL allowlist;
// API-key requests get the default 5m TTL.
const ephemeral = (ttl: string) => ({ type: "ephemeral" as const, ttl: ttl as "1h" | "5m" });

function cacheSystem(sys: any, ttl: string) {
  if (!sys) return undefined;
  if (typeof sys === "string") return [{ type: "text", text: sys, cache_control: ephemeral(ttl) }];
  const a = [...sys];
  if (a.length) a[a.length - 1] = { ...a[a.length - 1], cache_control: ephemeral(ttl) };
  return a;
}

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const block = content.find(
    (candidate): candidate is { type: "text"; text?: string } =>
      Boolean(candidate && typeof candidate === "object" && (candidate as any).type === "text"),
  );
  return typeof block?.text === "string" ? block.text : "";
}

// The binary derives a short billing suffix from the first non-meta user text.
function firstUserText(msgs: Msg[]): string {
  const candidate = msgs.find(message => {
    const value = message as Msg & { isMeta?: boolean };
    return value.role === "user" && value.isMeta !== true;
  });
  return candidate ? contentText(candidate.content) : "";
}

function billingSuffix(messages: Msg[]): string {
  const prompt = firstUserText(messages);
  const selected = [4, 7, 20].map(index => prompt[index] || "0").join("");
  return createHash("sha256")
    .update(`59cf53e54c78${selected}${VER}`)
    .digest("hex")
    .slice(0, 3);
}

function systemBlocks(sys: string | Anthropic.Messages.TextBlockParam[]): any[] {
  return typeof sys === "string" ? [{ type: "text", text: sys }] : [...sys];
}

function resolveThinking(t: Send["thinking"], interactive: boolean, model: string, maxTokens: number): { on: boolean; param: any } {
  if (!t) return { on: false, param: undefined };
  if (t === true || t === "adaptive") {
    const display = interactive ? {} : { display: "omitted" as const };
    if (adaptiveCapable(model)) return { on: true, param: { type: "adaptive", ...display } };
    // CC falls back to budget thinking on models without adaptive_thinking
    return { on: true, param: { type: "enabled", budget_tokens: Math.max(1024, maxTokens - 1), ...display } };
  }
  return { on: true, param: t };
}

// CLAUDE_CODE_EXTRA_BODY merges arbitrary fields into the request body.
function extraBody(): Record<string, unknown> {
  const raw = process.env.CLAUDE_CODE_EXTRA_BODY;
  if (!raw) return {};
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === "object" && !Array.isArray(v)) return v;
  } catch { /* malformed env is ignored by CC too */ }
  return {};
}

// CLAUDE_CODE_ATTRIBUTION_HEADER=<falsy> suppresses the billing system block.
function attributionOn(): boolean {
  const v = process.env.CLAUDE_CODE_ATTRIBUTION_HEADER;
  return v === undefined || /^(1|true|yes|on)$/i.test(v.trim());
}

let generatedDeviceId: string | undefined;

function deviceId(config: CcConfig): string {
  if (typeof config.userID === "string" && /^[0-9a-f]{64}$/i.test(config.userID)) return config.userID;
  return generatedDeviceId ??= randomBytes(32).toString("hex");
}

export class Claude {
  private auth: AuthProvider;
  private model: string;
  private xbetas: string[];
  private retries: number;
  private base: string | undefined;
  private sid = randomUUID();
  private did: string;
  private ccCfg: CcConfig;
  private interactive: boolean;

  constructor(o: Opts = {}) {
    this.auth = new AuthProvider(o.auth ?? {});
    this.model = o.model ?? "claude-sonnet-5";
    this.xbetas = o.betas ?? [];
    this.retries = o.maxRetries ?? 2;
    this.base = o.baseURL;
    this.interactive = o.interactive ?? true;
    this.ccCfg = readCcConfig();
    this.did = deviceId(this.ccCfg);
  }

  // CC sends metadata.user_id as JSON with device/account/session for rate limit routing
  private meta() {
    return { user_id: JSON.stringify({ device_id: this.did, account_uuid: this.ccCfg.oauthAccount?.accountUuid ?? "", session_id: this.sid }) };
  }

  private entrypoint() {
    return process.env.CLAUDE_CODE_ENTRYPOINT ?? (this.interactive ? "cli" : "sdk-cli");
  }

  private ttl() { return this.auth.oauth ? "1h" : "5m"; }

  // Betas are model-capability driven in CC, not driven by the request's
  // thinking flag: thinking-capable models always carry isp/ttc (and redact on
  // the interactive entrypoint).
  private betas(o: Send, model: string) {
    const b: string[] = [B.cc];
    if (this.auth.oauth) b.push(B.oauth);
    if (thinkCapable(model)) {
      b.push(B.isp);
      if (o.redact ?? this.interactive) b.push(B.redact);
      b.push(B.ttc, B.ctx);
    }
    b.push(B.cache);
    if (midConv(model)) b.push(B.midSys);
    b.push(B.advisor);
    if (effortCapable(model)) b.push(B.effort);
    if (this.ttl() === "1h") b.push(B.extTtl);
    if (o.ctx1m) b.push(B.ctx1m);
    if (o.speed === "fast") b.push(B.fast);
    for (const x of [...this.xbetas, ...(o.betas ?? [])]) if (!b.includes(x)) b.push(x);
    return b;
  }

  private params(o: Send, stream = false) {
    const model = o.model ?? this.model;
    const maxTokens = o.maxTokens ?? maxOut(model);
    const { on: think, param: thinkParam } = resolveThinking(o.thinking, this.interactive, model, maxTokens);
    const useCache = o.cache !== false;
    const oc: Record<string, unknown> = {};
    // CC sends the model's default effort; unsupported values are dropped.
    if (effortCapable(model)) oc.effort = o.effort ?? effortDefault(model);

    const extra = extraBody();
    let betas = this.betas(o, model);
    const xb = extra.anthropic_beta ?? extra.betas;
    if (xb !== undefined) {
      const list = (Array.isArray(xb) ? xb : String(xb).split(",")).map(v => String(v).trim()).filter(Boolean);
      betas = [...list, ...betas.filter(v => !list.includes(v))];
      delete extra.anthropic_beta;
      delete extra.betas;
    }

    return {
      model,
      messages: o.messages,
      max_tokens: maxTokens,
      metadata: this.meta(),
      betas,
      stream,
      ...(this.auth.oauth ? { system: this.firstPartySystem(o, useCache) } :
        o.system !== undefined ? { system: useCache ? cacheSystem(o.system, this.ttl()) : o.system } : {}),
      ...(o.tools && { tools: o.tools }),
      ...(o.toolChoice && { tool_choice: o.toolChoice }),
      ...(o.stop && { stop_sequences: o.stop }),
      // CC only sends temperature when explicitly set, and never with thinking
      ...(!think && o.temperature !== undefined && { temperature: o.temperature }),
      ...(think && { thinking: thinkParam }),
      ...(think && { context_management: { edits: [{ type: "clear_thinking_20251015", keep: "all" }] } }),
      ...(Object.keys(oc).length && { output_config: oc }),
      ...(o.speed && { speed: o.speed }),
      ...extra,
    };
  }

  private firstPartySystem(o: Send, useCache: boolean): any[] {
    const ttl = this.ttl();
    const blocks: any[] = [
      attributionOn() && {
        type: "text",
        text: `x-anthropic-billing-header: cc_version=${VER}.${billingSuffix(o.messages)}; cc_entrypoint=${this.entrypoint()};`,
      },
      {
        type: "text",
        text: this.interactive
          ? "You are Claude Code, Anthropic's official CLI for Claude."
          : "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
        ...(useCache ? { cache_control: ephemeral(ttl) } : {}),
      },
    ].filter(Boolean);
    if (o.system !== undefined) {
      const supplied = systemBlocks(o.system);
      if (useCache && supplied.length) {
        supplied[supplied.length - 1] = { ...supplied[supplied.length - 1], cache_control: ephemeral(ttl) };
      }
      blocks.push(...supplied);
    }
    return blocks;
  }

  // new Anthropic client per request — auth token may change between calls
  private async sdk(timeout: number): Promise<Anthropic> {
    await this.auth.refresh();
    const h: Record<string, string> = {
      "x-app": "cli",
      "User-Agent": `claude-cli/${VER} (external, ${this.entrypoint()})`,
      "X-Claude-Code-Session-Id": this.sid,
      // CC constructs the SDK client with dangerouslyAllowBrowser: true
      "anthropic-dangerous-direct-browser-access": "true",
      ...envHeaders(),
    };
    const c: ConstructorParameters<typeof Anthropic>[0] = {
      maxRetries: this.retries,
      timeout,
      defaultHeaders: h,
    };
    if (this.base) c.baseURL = this.base;
    if (this.auth.oauth) {
      const t = await this.auth.token();
      if (!t) throw new Error(
        "OAuth token not found. Run `claude /login` first, or pass auth.token directly.",
      );
      c.apiKey = null; c.authToken = t;
    } else {
      const k = this.auth.key();
      if (!k) throw new Error(
        "No authentication found. Either:\n" +
        "  1. Run `claude /login` to set up OAuth\n" +
        "  2. Set ANTHROPIC_API_KEY environment variable\n" +
        "  3. Pass auth: { apiKey: '...' } to constructor",
      );
      c.apiKey = k;
    }
    return new Anthropic(c);
  }

  private norm(o: string | Send): Send {
    return typeof o === "string" ? { messages: [{ role: "user", content: o }] } : o;
  }

  private reqOpts(o: Send, stream: boolean) {
    return {
      ...(o.signal && { signal: o.signal }),
      timeout: o.timeout ?? (stream ? 600_000 : 300_000),
    };
  }

  async send(o: string | Send): Promise<Response> {
    const s = this.norm(o);
    // capture token before request — if 401, re-read from keychain (another CC may have refreshed)
    let failed: string | undefined;
    try {
      failed = (await this.auth.token()) ?? undefined;
      const c = await this.sdk(s.timeout ?? 300_000);
      return await c.beta.messages.create(this.params(s) as any, this.reqOpts(s, false)) as Response;
    } catch (e: any) {
      if (e.status === 401 && this.auth.oauth && failed && await this.auth.onAuthError(failed)) {
        const c = await this.sdk(s.timeout ?? 300_000);
        return await c.beta.messages.create(this.params(s) as any, this.reqOpts(s, false)) as Response;
      }
      throw e;
    }
  }

  async ask(o: string | Send): Promise<string> {
    const r = await this.send(o);
    return r.content.filter((b): b is Anthropic.Messages.TextBlock => b.type === "text").map(b => b.text).join("");
  }

  async *stream(o: string | Send): AsyncGenerator<StreamEvent> {
    const s = this.norm(o);
    let failed: string | undefined;
    try {
      failed = (await this.auth.token()) ?? undefined;
      const c = await this.sdk(s.timeout ?? 600_000);
      const st = c.beta.messages.stream(this.params(s, true) as any, this.reqOpts(s, true));
      for await (const e of st) yield e as unknown as StreamEvent;
    } catch (e: any) {
      if (e.status === 401 && this.auth.oauth && failed && await this.auth.onAuthError(failed)) {
        const c = await this.sdk(s.timeout ?? 600_000);
        const st = c.beta.messages.stream(this.params(s, true) as any, this.reqOpts(s, true));
        for await (const e2 of st) yield e2 as unknown as StreamEvent;
        return;
      }
      throw e;
    }
  }

  async *streamText(o: string | Send): AsyncGenerator<string> {
    for await (const e of this.stream(o)) {
      if (e.type === "content_block_delta" && "delta" in e && (e.delta as any).type === "text_delta") {
        yield (e.delta as any).text;
      }
    }
  }
}
