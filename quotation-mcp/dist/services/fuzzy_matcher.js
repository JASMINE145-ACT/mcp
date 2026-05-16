import * as XLSX from "xlsx";
import { config } from "../config";
const PRICE_LEVEL_COLS = {
    A: ["A级别", "报单价格"],
    B: ["B级别", "报单价格"],
    C: ["C级别", "报单价格"],
    D: ["D级别", "报单价格"],
    D_LOW: ["D级别", "报单价格"],
    E: ["E级别", "报单价格"],
    FACTORY_INC_TAX: ["出厂价", "含税"],
    FACTORY_EXC_TAX: ["出厂价", "不含税"],
    PURCHASE_EXC_TAX: ["采购不含税"],
};
// Spec equivalence mapping (same as agent-jk)
const SPEC_EQUIVALENTS = {
    "50": ["50mm", "DN50", "50mm(2寸)", "2寸"],
    "40": ["40mm", "DN40", "40mm(1.5寸)", "1.5寸"],
    "25": ["25mm", "DN25", "25mm(1寸)", "1寸"],
    "32": ["32mm", "DN32"],
    "75": ["75mm", "DN75", "3寸"],
    "110": ["110mm", "DN110", "4寸"],
};
function tokenize(text) {
    // Split by non-alphanumeric characters, keep Chinese and alphanumeric tokens
    return (text || "")
        .toLowerCase()
        .split(/[^a-z0-9\u4e00-\u9fa5]+/)
        .filter((t) => t.length > 0);
}
function scoreTokens(keywords, description) {
    const kwTokens = tokenize(keywords);
    const descTokens = tokenize(description);
    if (!kwTokens.length)
        return 0;
    // Filter out meaningless tokens (pure numbers only)
    const meaningfulKwTokens = kwTokens.filter((t) => !/^\d+$/.test(t));
    if (meaningfulKwTokens.length === 0)
        return 0;
    let hit = 0;
    for (const t of meaningfulKwTokens) {
        // Check if keyword token is contained in description token (supports Chinese compound words)
        // Only check d.includes(t) - description should contain the keyword
        const contained = descTokens.some((d) => d.includes(t));
        if (contained) {
            // If it's a complete match (exact), give higher weight
            if (descTokens.includes(t)) {
                hit += 3;
            }
            else {
                hit += 2; // Partial/substring match
            }
        }
        else {
            // For compound tokens like "直接50", also check if any desc token contains the first part
            // This handles cases like "直接50" matching "直接头"
            if (t.length > 3) {
                // Extract Chinese prefix if present (handles cases like "直接50" -> "直接")
                const chinesePart = t.match(/[\u4e00-\u9fa5]+/)?.[0];
                if (chinesePart && chinesePart.length >= 2) {
                    const hasChineseMatch = descTokens.some((d) => d.includes(chinesePart));
                    if (hasChineseMatch) {
                        hit += 1; // Partial Chinese match
                    }
                }
            }
        }
        // spec equivalence
        const eqs = SPEC_EQUIVALENTS[t] ?? [];
        for (const e of eqs) {
            if (descTokens.some((d) => d.includes(e))) {
                hit += 1;
                break;
            }
        }
    }
    return Math.round((hit / meaningfulKwTokens.length) * 100);
}
function findPriceCol(headers, level) {
    const keywords = PRICE_LEVEL_COLS[level] ?? PRICE_LEVEL_COLS["B"];
    // Find column containing ALL keywords (e.g., "（一级代理）B级别 报单价格" contains both "B级别" and "报单价格")
    return headers.findIndex((h) => keywords.every((kw) => h.includes(kw)));
}
export async function matchWandingPriceCandidates(keywords, customerLevel = "B", maxCandidates = 20) {
    try {
        const wb = XLSX.readFile(config.wandingPriceLib, { sheetRows: 5000 });
        const sheet = wb.Sheets[wb.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(sheet, { header: 1 });
        if (!data.length)
            return [];
        const headers = data[0].map((h) => String(h ?? "").trim());
        const priceColIdx = findPriceCol(headers, customerLevel);
        const codeIdx = headers.findIndex((h) => h.includes("Material") || h.includes("产品编号"));
        const descIdx = headers.findIndex((h) => h.includes("Describrition") || h.includes("描述"));
        const rows = data.slice(1);
        const scored = rows
            .map((row) => {
            const desc = (row[descIdx] ?? "").toLowerCase();
            const code = String(row[codeIdx] ?? "").trim();
            if (!code || !desc)
                return null;
            const score = scoreTokens(keywords, desc);
            // Require at least one meaningful token match
            return score >= 60
                ? {
                    code,
                    matched_name: (row[descIdx] ?? "").trim(),
                    unit_price: parseFloat(row[priceColIdx] ?? "0") || 0,
                    source: "字段匹配",
                    score,
                }
                : null;
        })
            .filter(Boolean);
        return scored.sort((a, b) => b.score - a.score).slice(0, maxCandidates);
    }
    catch {
        return [];
    }
}
