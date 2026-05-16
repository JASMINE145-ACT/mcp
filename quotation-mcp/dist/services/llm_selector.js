import { readFileSync, existsSync } from "fs";
import { config } from "../config";
const BUSINESS_KNOWLEDGE = `候选选择业务规则：
1. 选择与关键词最贴近的规格、材质、口径、用途。
2. 口径优先：dn50、50、1-1/2 需对应转换后再比对。
3. 材质优先：PPR/PVC-U/PE 不能混选，除非关键词未指定且候选强相关。
4. 来源是 tie-breaker：共同>历史报价>字段匹配，但语义冲突时语义优先。
5. 若都不匹配可返回 index=0。原因须 >=10 字。
`;
const SYSTEM_PROMPT = `Output ONLY a single JSON object. No prose. No markdown fences.
Reason must be >=10 Chinese characters. Schema: {"index": <0..N>, "reason": "<short zh reason>"}.
Choose exactly one index.`;
// Pre-filter/scoring (reference agent-jk _apply_candidate_pre_filter)
function applyPreFilter(keywords, candidates) {
    const kw = (keywords || "").toLowerCase();
    const scored = candidates.map((c) => {
        let filterScore = 0;
        const name = ((c.matched_name || "") + " " + c.code).toLowerCase();
        const src = c.source || "";
        if (!kw.includes("给水") && !kw.includes("aw")) {
            if (name.includes("aw") && name.includes("给水"))
                filterScore -= 15;
            if (name.includes("排水"))
                filterScore += 8;
        }
        if (kw.includes("国标")) {
            name.includes("印尼") ? (filterScore -= 10) : (filterScore += 6);
        }
        if (kw.includes("日标") || kw.includes("印尼"))
            filterScore += 6;
        const srcScore = src === "共同" ? 9 : src === "历史报价" ? 6 : 3;
        filterScore += srcScore;
        // Preserve original matching score, add filter score as separate field
        return { ...c, filterScore };
    });
    return scored.sort((a, b) => (b.filterScore ?? 0) - (a.filterScore ?? 0));
}
function loadKnowledge() {
    try {
        const path = config.businessKnowledge;
        if (existsSync(path))
            return readFileSync(path, "utf-8").trim();
    }
    catch { /* ignore */ }
    return BUSINESS_KNOWLEDGE;
}
function buildPrompt(keywords, candidates, knowledge) {
    const lines = candidates.map((c, i) => `${i + 1}. [${c.code}] ${c.matched_name} | price=${c.unit_price} | src=${c.source}`).join("\n");
    return `keywords: ${keywords}\nN=${candidates.length}\ncandidates:\n${lines}\n\nbusiness_knowledge:\n${knowledge}\n\ntask: choose exactly one index in 1..${candidates.length}, or 0 if none matches.\noutput JSON only: {"index": number, "reason": "short text"}`;
}
export async function llmSelectBest(keywords, candidates) {
    if (!candidates.length)
        return null;
    const sorted = applyPreFilter(keywords, candidates);
    const top = sorted.slice(0, 8);
    const knowledge = loadKnowledge();
    const apiKey = config.llmSelectorApiKey;
    const model = config.llmSelectorModel;
    if (!apiKey || !model) {
        // No LLM - fallback to first candidate
        return sorted[0] ?? null;
    }
    try {
        const resp = await fetch(`${config.llmSelectorBaseUrl}/chat/completions`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model,
                messages: [
                    { role: "system", content: SYSTEM_PROMPT },
                    { role: "user", content: buildPrompt(keywords, top, knowledge) },
                ],
                max_tokens: 500,
                temperature: 0,
                response_format: { type: "json_object" },
            }),
            signal: AbortSignal.timeout(config.llmSelectorTimeout * 1000),
        });
        if (!resp.ok)
            throw new Error(`LLM API error: ${resp.status}`);
        const json = await resp.json();
        const content = json.choices?.[0]?.message?.content ?? "";
        const parsed = JSON.parse(content);
        const idx = Number(parsed?.index ?? 0);
        if (idx <= 0 || idx > top.length)
            return sorted[0] ?? null;
        return top[idx - 1];
    }
    catch {
        return sorted[0] ?? null;
    }
}
