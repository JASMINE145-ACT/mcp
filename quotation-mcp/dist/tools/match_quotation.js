import { config } from "../config";
import { matchWandingPriceCandidates } from "../services/fuzzy_matcher";
import { matchMappingTopCandidates } from "../services/mapping_matcher";
import { llmSelectBest } from "../services/llm_selector";
const SOURCE_PRIORITY = { "共同": 0, "历史报价": 1, "字段匹配": 2 };
function mergeCandidates(mapping, wanding) {
    const byCode = new Map();
    for (const c of mapping) {
        byCode.set(c.code, { ...c, source: c.code in byCode ? "共同" : "历史报价" });
    }
    for (const c of wanding) {
        if (byCode.has(c.code)) {
            const existing = byCode.get(c.code);
            if (c.unit_price !== 0)
                existing.unit_price = c.unit_price;
            existing.source = "共同";
        }
        else {
            byCode.set(c.code, { ...c, source: "字段匹配" });
        }
    }
    return Array.from(byCode.values());
}
export async function executeMatchQuotation(params) {
    const { keywords, customerLevel = "B", lang, showAllCandidates } = params;
    if (!keywords?.trim()) {
        return { success: false, error: "keywords cannot be empty" };
    }
    try {
        // Parallel two-way
        const [mapping, wanding] = await Promise.all([
            lang === "en"
                ? Promise.resolve([])
                : matchMappingTopCandidates(keywords, 5),
            matchWandingPriceCandidates(keywords, customerLevel, 20),
        ]);
        let candidates = mergeCandidates(mapping, wanding);
        // Sort by source priority
        candidates = candidates
            .sort((a, b) => (SOURCE_PRIORITY[a.source] ?? 2) - (SOURCE_PRIORITY[b.source] ?? 2))
            .slice(0, config.maxCandidates);
        if (candidates.length === 0) {
            return {
                success: true,
                result: JSON.stringify({ unmatched: true, keywords }),
            };
        }
        if (showAllCandidates) {
            return {
                success: true,
                result: JSON.stringify({ needs_selection: true, candidates }),
            };
        }
        if (candidates.length === 1) {
            return {
                success: true,
                result: JSON.stringify({
                    single: true,
                    chosen: candidates[0],
                    match_source: candidates[0].source,
                }),
            };
        }
        // >=2 candidates -> LLM selection
        const best = await llmSelectBest(keywords, candidates);
        if (!best) {
            return {
                success: true,
                result: JSON.stringify({ unmatched: true, keywords }),
            };
        }
        return {
            success: true,
            result: JSON.stringify({
                single: true,
                chosen: best,
                match_source: best.source ?? "共同",
            }),
        };
    }
    catch (e) {
        return { success: false, error: String(e) };
    }
}
