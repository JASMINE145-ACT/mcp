import { existsSync } from "fs";
import { resolve } from "path";
import * as XLSX from "xlsx";
export function parseExcelSmart(filePath, opts = {}) {
    try {
        const fullPath = resolve(filePath);
        if (!existsSync(fullPath)) {
            return { success: false, error: `ENOENT: file not found: ${fullPath}` };
        }
        const workbook = XLSX.readFile(fullPath, { sheetRows: 0 });
        const sheetName = opts.sheetName ?? workbook.SheetNames[0];
        const sheet = workbook.Sheets[sheetName];
        if (!sheet) {
            return { success: false, error: `Sheet not found: ${sheetName}` };
        }
        const range = XLSX.utils.decode_range(sheet["!ref"] ?? "A1");
        const maxRows = opts.maxRows ?? Infinity;
        const actualMaxRow = Math.min(range.e.r + 1, maxRows + 1); // +1 for header
        const data = XLSX.utils.sheet_to_json(sheet, {
            header: 1,
            defval: "",
            range: 0,
        });
        const truncated = data.length > maxRows;
        const rows = data.slice(1, actualMaxRow);
        const headers = data[0].map((h) => String(h ?? "").trim());
        return {
            success: true,
            result: {
                headers,
                rows,
                total_rows: data.length - 1,
                ...(truncated ? { truncated: true } : {}),
            },
        };
    }
    catch (e) {
        return { success: false, error: String(e) };
    }
}
