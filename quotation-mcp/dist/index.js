import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import { callPythonTool } from "./python-spawner";
const server = new Server({ name: "quotation-server", version: "1.0.0" }, { capabilities: { tools: {} } });
const customerLevelSchema = {
    type: "string",
    enum: ["A", "B", "C", "D", "E"],
    default: "B",
    description: "Customer price level. Claude Code uses returned candidates and selection context to choose.",
};
server.setRequestHandler(ListToolsRequestSchema, () => ({
    tools: [
        {
            name: "match_quotation",
            description: "Match quotation items using migrated agent-jk Python logic. Default behavior is Claude Code auto-selection: the tool returns candidates plus wanding business knowledge as selection context; Claude Code must choose one result and not show candidates unless requested. No internal selector model is called.",
            inputSchema: {
                type: "object",
                properties: {
                    keywords: { type: "string", description: "Product name/spec, e.g. PVC-U pipe DN25." },
                    customer_level: customerLevelSchema,
                    product_type: { type: "string", description: "Optional product type hint for filtering." },
                    price_library_path: { type: "string", description: "Optional price library path. Defaults to data/wanding_price_lib.xlsx." },
                    show_candidates: { type: "boolean", description: "Set true only when the user explicitly asks to see the candidate list." },
                },
                required: ["keywords"],
            },
        },
        {
            name: "match_quotation_batch",
            description: "Match up to 50 quotation queries. Default behavior is Claude Code auto-selection per item using the returned selection context; do not show candidate lists unless requested. No internal selector model is called.",
            inputSchema: {
                type: "object",
                properties: {
                    keywords_list: { type: "array", items: { type: "string" }, maxItems: 50 },
                    customer_level: customerLevelSchema,
                    product_type: { type: "string" },
                    price_library_path: { type: "string" },
                    show_candidates: { type: "boolean", description: "Set true only when the user explicitly asks to see candidate lists." },
                },
                required: ["keywords_list"],
            },
        },
        {
            name: "get_inventory_by_code",
            description: "Query inventory by product code. In standalone MCP mode this returns null unless Accurate API credentials are configured.",
            inputSchema: {
                type: "object",
                properties: { code: { type: "string", description: "Product code / Item Code." } },
                required: ["code"],
            },
        },
        {
            name: "get_inventory_by_code_batch",
            description: "Query inventory for up to 50 product codes.",
            inputSchema: {
                type: "object",
                properties: { codes: { type: "array", items: { type: "string" }, maxItems: 50 } },
                required: ["codes"],
            },
        },
        {
            name: "fill_quotation_sheet",
            description: "Fill a quotation Excel using migrated Python logic. Ambiguous rows are not auto-selected; they are returned with candidates for Claude Code/user review.",
            inputSchema: {
                type: "object",
                properties: {
                    file_path: { type: "string", description: "Input quotation Excel absolute path." },
                    output_path: { type: "string", description: "Optional output path." },
                    sheet_name: { type: "string", description: "Optional worksheet name." },
                    customer_level: customerLevelSchema,
                    price_library_path: { type: "string" },
                },
                required: ["file_path"],
            },
        },
        {
            name: "parse_excel_smart",
            description: "Parse an Excel file using the migrated agent-jk quote_tools.parse_excel_smart logic.",
            inputSchema: {
                type: "object",
                properties: {
                    file_path: { type: "string", description: "Excel absolute path." },
                    sheet_name: { type: "string", description: "Optional worksheet name." },
                    max_rows: { type: "number", description: "Maximum rows to return." },
                },
                required: ["file_path"],
            },
        },
        {
            name: "ask_clarification",
            description: "Return a clarification prompt/options object for quotation ambiguity.",
            inputSchema: {
                type: "object",
                properties: {
                    question: { type: "string" },
                    reason: { type: "string" },
                    options: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: { id: { type: "string" }, name: { type: "string" } },
                            required: ["id", "name"],
                        },
                    },
                },
            },
        },
    ],
}));
function asRecord(value) {
    return value && typeof value === "object" ? value : {};
}
function normalizeArgs(args) {
    const normalized = { ...args };
    if (typeof normalized.customerLevel === "string" && typeof normalized.customer_level !== "string") {
        normalized.customer_level = normalized.customerLevel;
    }
    return normalized;
}
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: rawArgs } = request.params;
    const args = normalizeArgs(asRecord(rawArgs));
    try {
        const result = await callPythonTool(name, args);
        if (!result.success) {
            return {
                content: [{ type: "text", text: JSON.stringify({ error: result.error }, null, 2) }],
                isError: true,
            };
        }
        return {
            content: [{ type: "text", text: JSON.stringify(result.result, null, 2) }],
        };
    }
    catch (error) {
        return {
            content: [{ type: "text", text: JSON.stringify({ error: String(error) }, null, 2) }],
            isError: true,
        };
    }
});
const transport = new StdioServerTransport();
server.connect(transport).catch((error) => {
    console.error(error);
    process.exit(1);
});
