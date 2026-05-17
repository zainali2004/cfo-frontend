import { useDatasetStore } from '@/stores/useDatasetStore'
import { SectorHeader } from '@/components/sector/SectorHeader'
import { TopicCloud } from '@/components/sector/TopicCloud'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'

export function SectorPage() {
    const profiles = useDatasetStore((s) => s.profiles)
    const benchmarks = useDatasetStore((s) => s.sectorBenchmarks)
    const loading = useDatasetStore((s) => s.isSectorDetecting)
    const sectorError = useDatasetStore((s) => s.sectorError)
    const runSectorDetection = useDatasetStore((s) => s.runSectorDetection)
    const concepts = useDatasetStore((s) => s.kpiConcepts)
    const kpiLoading = useDatasetStore((s) => s.isKpiCalculating)
    const kpiError = useDatasetStore((s) => s.kpiError)
    const agent3Done = useDatasetStore((s) => s.agent3Done)
    const runKpiCalculation = useDatasetStore((s) => s.runKpiCalculation)
    const combinedReady = Boolean(benchmarks) && agent3Done && concepts.length > 0

    async function runSectorAndKpi() {
        if (!profiles.length) {
            toast.error('Run Data Profiler first.')
            return
        }

        try {
            await runSectorDetection()
            const afterSector = useDatasetStore.getState()
            if (afterSector.sectorError || !afterSector.agent2Done || !afterSector.domainInfo) {
                return
            }

            await runKpiCalculation()
            const state = useDatasetStore.getState()
            if (!state.kpiError) {
                toast.success('Sector and KPI pipeline completed')
            }
        } catch (err) {
            console.error('Sector & KPI pipeline failed:', err)
            toast.error('Pipeline failed. Check console for details.')
        }
    }

    const handleRunSectorAndKpi = async () => {
        try {
            await runSectorAndKpi()
        } catch (err) {
            console.error('Failed to run Sector & KPI:', err)
        }
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
                <span className="ml-3 text-gray-500 font-medium">Analysing Industry Trends...</span>
            </div>
        )
    }

    return (
        <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div className="flex items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Sector & KPI</h2>
                    <p className="text-gray-500 dark:text-gray-400">Run Agent 2 to detect the sector, then Agent 3 to calculate KPIs.</p>
                </div>
                <Button
                    onClick={handleRunSectorAndKpi}
                    disabled={loading || kpiLoading || !profiles.length}
                    className="bg-deloitte hover:bg-deloitte-dark text-white font-medium h-11"
                >
                    {loading || kpiLoading ? 'Running Sector & KPI Pipeline...' : 'Run Sector & KPI'}
                </Button>
            </div>

            {sectorError && (
                <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800/40 dark:bg-red-900/20 dark:text-red-300">
                    {sectorError}
                </div>
            )}

            {loading || kpiLoading ? (
                <div className="flex h-[38vh] w-full items-center justify-center rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
                    <div className="flex items-center gap-3 text-gray-500 font-medium">
                        <Loader2 className="h-6 w-6 animate-spin text-deloitte" />
                        <span>Running Sector and KPI pipeline...</span>
                    </div>
                </div>
            ) : !combinedReady ? (
                <div className="text-center py-12">
                    <p className="text-gray-500">Run the combined Sector & KPI pipeline to load the results.</p>
                </div>
            ) : (
                <>
                    <SectorHeader
                        sector={benchmarks?.sector || ''}
                        confidence={benchmarks?.confidence || 0}
                        sentiment={(benchmarks?.market_sentiment as 'Bullish' | 'Bearish' | 'Neutral') || 'Neutral'}
                        wikiUrl={benchmarks?.wiki_url || ''}
                    />
                    <TopicCloud topics={benchmarks?.topics || []} />

                    <div className="mt-12 space-y-6">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <h3 className="text-xl font-bold text-gray-900 dark:text-white">KPI Metrics</h3>
                                <p className="text-gray-500 dark:text-gray-400">Agent 3 runs automatically after sector detection.</p>
                            </div>
                        </div>

                        {kpiError && (
                            <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800/40 dark:bg-red-900/20 dark:text-red-300">
                                {kpiError}
                            </div>
                        )}

                        <div className="space-y-4">
                            {concepts.map((kpi, idx) => (
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
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            )}
        </div>
    )
}
