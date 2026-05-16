import * as XLSX from "xlsx";
import { config } from "../config";
function normalizeText(s) {
    return s.toLowerCase().replace(/\s+/g, "").replace(/[/\\-]/g, "");
}
function fuzzyScore(keywords, text) {
    const kw = normalizeText(keywords);
    const tx = normalizeText(text);
    if (tx === kw)
        return 100;
    if (tx.includes(kw))
        return 80;
    // token overlap scoring (support Chinese)
    const kwTokens = (kw.match(/[a-z0-9\u4e00-\u9fa5]+/g) ?? []).map((t) => t);
    const txTokens = (tx.match(/[a-z0-9\u4e00-\u9fa5]+/g) ?? []).map((t) => t);
    if (kwTokens.length === 0)
        return 0;
    const hit = kwTokens.filter((t) => txTokens.includes(t)).length;
    return Math.round((hit / kwTokens.length) * 60);
}
export async function matchMappingTopCandidates(keywords, topK = 5) {
    try {
        const wb = XLSX.readFile(config.mappingTable, { sheetRows: 2000 });
        const sheet = wb.Sheets[wb.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(sheet, { defval: "" });
        const scored = data
            .map((row) => {
            // Multi-language header support
            const inquiryName = (row["询价货物名称"] ??
                row["Nama Permintaan Barang"] ??
                row["询价货物名称（中文）"] ??
                "").trim();
            const spec = (row["询价规格型号"] ??
                row["Spesifikasi dan Model Permintaan Barang"] ??
                row["规格"] ??
                "").trim();
            const combined = `${inquiryName} ${spec}`.trim();
            const code = (row["产品编号"] ??
                row["Product number"] ??
                row["编号"] ??
                "").trim();
            const quoteName = (row["报价名称"] ??
                row["Nama Penawaran Barang"] ??
                row["产品名称"] ??
                inquiryName).trim();
            // Note: mapping_table.xlsx doesn't have price column, price will be filled from wanding_price_lib
            const price = 0; // Will be filled by mergeCandidates from wanding_price_lib
            if (!code)
                return null;
            const score = fuzzyScore(keywords, combined);
            return score > 0 ? { code, matched_name: quoteName, unit_price: price, source: "历史报价", score } : null;
        })
            .filter(Boolean);
        return scored.sort((a, b) => b.score - a.score).slice(0, topK);
    }
    catch {
        return [];
    }
}
