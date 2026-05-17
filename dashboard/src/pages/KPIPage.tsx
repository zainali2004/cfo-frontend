import { useDatasetStore } from '@/stores/useDatasetStore'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'

export function KPIPage() {
    const concepts = useDatasetStore((s) => s.kpiConcepts)
    const loading = useDatasetStore((s) => s.isKpiCalculating)
    const kpiError = useDatasetStore((s) => s.kpiError)
    const agent3Done = useDatasetStore((s) => s.agent3Done)
    const runKpiCalculation = useDatasetStore((s) => s.runKpiCalculation)

    async function handleRunKpi() {
        await runKpiCalculation()
        const state = useDatasetStore.getState()
        if (!state.kpiError) {
            toast.success('KPI calculation completed')
        }
    }

    function toClientSteps(input?: string | string[]): string[] {
        if (!input) return []
        if (Array.isArray(input)) {
            return input.map((p) => p.trim()).filter(Boolean)
        }
        return input
            .split(/\n|\.|;|→|->/g)
            .map((p) => p.trim())
            .filter(Boolean)
    }

    function humanizeFormulaText(text: string): string {
        if (!text) return text

        const tokenToWords = (token: string) => token.replace(/_/g, ' ')

        let out = text.replace(/\b[a-z]+(?:_[a-z0-9]+)+\b/gi, (match) => tokenToWords(match))
        out = out
            .replace(/\s*=\s*/g, ' equals ')
            .replace(/\s*\/\s*/g, ' divided by ')
            .replace(/\s*\*\s*/g, ' multiplied by ')
            .replace(/\s*\+\s*/g, ' plus ')
            .replace(/\s*-\s*/g, ' minus ')
            .replace(/\s+/g, ' ')
            .trim()

        if (!out) return text
        return out.charAt(0).toUpperCase() + out.slice(1)
    }

    function humanizeVariableNames(text: string): string {
        if (!text) return text

        return text.replace(/\b[a-z]+(?:_[a-z0-9]+)+\b/gi, (match) => {
            return match.replace(/_/g, ' ')
        })
    }

    function capitalizeFirstLetter(text: string): string {
        const trimmed = text.trim()
        if (!trimmed) return trimmed
        return trimmed.charAt(0).toUpperCase() + trimmed.slice(1)
    }

    function extractCalculationSteps(kpi: {
        calculation_formula?: string
        calculation_steps?: string[] | string
        calculation_explainer?: string
        data_mapping?: { calculation_method?: string }
    }): string[] {
        const stepsFromList = toClientSteps(kpi.calculation_steps)
        if (stepsFromList.length > 0) return stepsFromList

        const stepsFromExplainer = toClientSteps(kpi.calculation_explainer)
        if (stepsFromExplainer.length > 0) return stepsFromExplainer

        const formula = kpi.calculation_formula?.trim()
        if (formula) {
            return [formula]
        }

        const method = kpi.data_mapping?.calculation_method?.trim()
        if (method) {
            return [method]
        }

        return []
    }

    function extractAssumptions(kpi: Record<string, unknown>): string[] {
        const candidates: unknown[] = []

        candidates.push(kpi.assumptions)
        candidates.push(kpi.assumption)
        candidates.push(kpi.missing_inputs)

        const mapping = typeof kpi.data_mapping === 'object' && kpi.data_mapping !== null
            ? (kpi.data_mapping as Record<string, unknown>)
            : null
        if (mapping) {
            candidates.push(mapping.assumptions)
            candidates.push(mapping.missing_inputs)
        }

        const flattened: string[] = []
        candidates.forEach((candidate) => {
            if (!candidate) return
            if (Array.isArray(candidate)) {
                candidate.forEach((item) => {
                    const t = String(item).trim()
                    if (t) flattened.push(t)
                })
                return
            }
            if (typeof candidate === 'string') {
                const t = candidate.trim()
                if (t) flattened.push(t)
                return
            }
            if (typeof candidate === 'object') {
                Object.values(candidate as Record<string, unknown>).forEach((val) => {
                    const t = String(val).trim()
                    if (t) flattened.push(t)
                })
            }
        })

        // Deduplicate while preserving order.
        const seen = new Set<string>()
        return flattened.filter((item) => {
            const key = item.toLowerCase()
            if (seen.has(key)) return false
            seen.add(key)
            return true
        })
    }

    function formatCalculatedValue(value: unknown): string {
        if (value === null || value === undefined || value === '') return 'Needs data'
        if (typeof value === 'string' && value.toLowerCase() === 'needs_data') return 'Needs data'
        if (typeof value === 'number') {
            if (!Number.isFinite(value)) return 'Needs data'
            return Number.isInteger(value) ? value.toString() : value.toFixed(2)
        }
        return String(value)
    }

    if (loading) {
        return (
            <div className="flex h-[50vh] w-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-deloitte" />
                <span className="ml-3 text-gray-500 font-medium">Calculating KPI metrics...</span>
            </div>
        )
    }

    return (
        <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div className="flex items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">KPI Metrics</h2>
                    <p className="text-gray-500 dark:text-gray-400">
                        Run Agent 3 to generate KPI name, meaning, value, and calculation steps.
                    </p>
                </div>
                <Button
                    onClick={handleRunKpi}
                    disabled={loading}
                    className="bg-deloitte hover:bg-deloitte-dark text-white font-medium h-11"
                >
                    {agent3Done ? 'Re-run KPI Calculator' : 'Run KPI Calculator'}
                </Button>
            </div>

            {kpiError && (
                <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800/40 dark:bg-red-900/20 dark:text-red-300">
                    {kpiError}
                </div>
            )}

            {concepts.length === 0 ? (
                <div className="text-center py-12">
                    <p className="text-gray-500">Run KPI Calculator to load metrics.</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {concepts.map((kpi, idx) => {
                        const steps = extractCalculationSteps(kpi)
                            .map(humanizeFormulaText)
                            .map(capitalizeFirstLetter)
                        const assumptions = extractAssumptions(kpi as Record<string, unknown>)
                            .filter((item) => item && item.toLowerCase() !== 'n/a')
                            .map((item) => humanizeVariableNames(item))
                            .map(capitalizeFirstLetter)

                        return (
                            <div key={`${kpi.concept_phrase ?? 'kpi'}-${idx}`} className="rounded-lg border border-gray-200 bg-white p-5 dark:border-gray-700 dark:bg-gray-800">
                                <div className="grid gap-3 md:grid-cols-2">
                                    <div>
                                        <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">KPI</p>
                                        <p className="text-lg font-semibold text-gray-900 dark:text-white">{kpi.concept_phrase || 'Untitled KPI'}</p>
                                    </div>
                                    <div>
                                        <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">Value</p>
                                        <p className="text-lg font-semibold text-deloitte">{formatCalculatedValue(kpi.calculated_value)}</p>
                                    </div>
                                </div>

                                <div className="mt-4">
                                    <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">Meaning</p>
                                    <p className="text-sm text-gray-700 dark:text-gray-300">{kpi.why_it_matters || kpi.business_relevance || 'No meaning provided.'}</p>
                                </div>

                                <div className="mt-4">
                                    <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">How To Calculate</p>
                                    {steps.length > 0 ? (
                                        <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm text-gray-700 dark:text-gray-300">
                                            {steps.map((step, sidx) => (
                                                <li key={`${idx}-step-${sidx}`}>{step}</li>
                                            ))}
                                        </ol>
                                    ) : (
                                        <p className="text-sm text-gray-500 dark:text-gray-400">No calculation steps provided.</p>
                                    )}
                                </div>

                                <div className="mt-4">
                                    <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">Assumptions / Missing Inputs</p>
                                    {assumptions.length > 0 ? (
                                        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-gray-700 dark:text-gray-300">
                                            {assumptions.map((item, aidx) => (
                                                <li key={`${idx}-assume-${aidx}`}>{item}</li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">none</p>
                                    )}
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
