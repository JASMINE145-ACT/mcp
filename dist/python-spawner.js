import { spawn } from "child_process";
import { config } from "./config";
const PYTHON_TIMEOUT_MS = Number(process.env.QUOTATION_PYTHON_TIMEOUT_MS ?? 90000);
function parseLastJsonLine(stdout) {
    const lines = stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const last = lines.at(-1);
    if (!last) {
        return { success: false, error: "Python produced no JSON output" };
    }
    return JSON.parse(last);
}
export async function callPythonTool(tool, params) {
    return new Promise((resolveResult) => {
        const pythonCmd = process.env.PYTHON_EXECUTABLE ?? (process.platform === "win32" ? "python" : "python3");
        const proc = spawn(pythonCmd, [config.pythonEntry], {
            cwd: config.projectRoot,
            env: {
                ...process.env,
                PYTHONIOENCODING: process.env.PYTHONIOENCODING ?? "utf-8",
                PYTHONUTF8: process.env.PYTHONUTF8 ?? "1",
                ENABLE_WANDING_VECTOR: process.env.ENABLE_WANDING_VECTOR ?? "0",
                INVENTORY_ENABLE_RESOLVER_VECTOR: process.env.INVENTORY_ENABLE_RESOLVER_VECTOR ?? "0",
                USE_RESOLVER_FALLBACK: process.env.USE_RESOLVER_FALLBACK ?? "0",
                DATA_DIR: process.env.DATA_DIR ?? config.dataDir,
                WANDING_PRICE_LIB_PATH: process.env.WANDING_PRICE_LIB_PATH ?? config.wandingPriceLib,
                PRICE_LIBRARY_PATH: process.env.PRICE_LIBRARY_PATH ?? config.wandingPriceLib,
                MAPPING_TABLE_PATH: process.env.MAPPING_TABLE_PATH ?? config.mappingTable,
                WANDING_BUSINESS_KNOWLEDGE_PATH: process.env.WANDING_BUSINESS_KNOWLEDGE_PATH ?? config.businessKnowledge,
                AOL_ACCESS_TOKEN: process.env.AOL_ACCESS_TOKEN ?? "",
                AOL_DATABASE_ID: process.env.AOL_DATABASE_ID ?? "",
                AOL_SIGNATURE_SECRET: process.env.AOL_SIGNATURE_SECRET ?? "",
                AOL_API_BASE_URL: process.env.AOL_API_BASE_URL ?? "https://account.accurate.id",
            },
            stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        let settled = false;
        const finish = (result) => {
            if (settled)
                return;
            settled = true;
            clearTimeout(timer);
            resolveResult(result);
        };
        const timer = setTimeout(() => {
            proc.kill();
            finish({ success: false, error: `Python call timed out after ${PYTHON_TIMEOUT_MS}ms` });
        }, PYTHON_TIMEOUT_MS);
        proc.stdout.on("data", (data) => {
            stdout += data.toString();
        });
        proc.stderr.on("data", (data) => {
            stderr += data.toString();
        });
        proc.on("error", (err) => {
            finish({
                success: false,
                error: `Failed to spawn Python: ${err.message}. Set PYTHON_EXECUTABLE if Python is not on PATH.`,
            });
        });
        proc.on("close", (code) => {
            if (settled)
                return;
            try {
                const parsed = parseLastJsonLine(stdout);
                if (!parsed.success && stderr) {
                    parsed.error = `${parsed.error ?? "Python tool failed"}\n${stderr.trim()}`;
                }
                finish(parsed);
            }
            catch (err) {
                finish({
                    success: false,
                    error: `Failed to parse Python output (exit ${code ?? "unknown"}): ${String(err)}\nstdout=${stdout}\nstderr=${stderr}`,
                });
            }
        });
        proc.stdin.write(`${JSON.stringify({ tool, params })}\n`);
        proc.stdin.end();
    });
}
