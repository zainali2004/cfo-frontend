import { useMemo, useState } from 'react'
import { Sparkles, AlertTriangle, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useDatasetStore } from '@/stores/useDatasetStore'
import type { InsightCategoryKey } from '@/services/dataService'

export function InsightsPage() {
    const insights = useDatasetStore((s) => s.insights)
    const isGenerating = useDatasetStore((s) => s.isGeneratingInsights)
    const error = useDatasetStore((s) => s.insightsError)
    const generateInsights = useDatasetStore((s) => s.generateInsights)
    const data = useDatasetStore((s) => s.data)

    const categoryEntries = useMemo(() => {
        if (!insights) return [] as Array<{ key: InsightCategoryKey; label: string; items: unknown[] }>
        const labels: Record<InsightCategoryKey, string> = {
            descriptive: 'Descriptive',
            predictive: 'Predictive',
            domain_related: 'Sector Related',
            novel_patterns: 'Novel Patterns',
            quality_implications: 'Quality Implications',
            recommended_actions: 'Recommended Actions',
            open_questions: 'Open Questions',
        }

        return (Object.keys(insights) as InsightCategoryKey[])
            .filter((key) => Array.isArray(insights[key]) && (insights[key]?.length ?? 0) > 0)
            .map((key) => ({
                key,
                label: labels[key],
                items: (insights[key] ?? []) as unknown[],
            }))
    }, [insights])

    const [activeTab, setActiveTab] = useState<InsightCategoryKey | null>(null)

    const effectiveActiveTab = useMemo(() => {
        if (!categoryEntries.length) return null
        if (activeTab && categoryEntries.some((entry) => entry.key === activeTab)) return activeTab
        return categoryEntries[0].key
    }, [activeTab, categoryEntries])

    if (!data) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[400px] text-center p-8 animate-in fade-in zoom-in duration-500">
                <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-full mb-4">
                    <FileText className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">No Data Available</h3>
                <p className="text-gray-500 dark:text-gray-400 max-w-sm mt-2">
                    Please upload a dataset first to generate AI insights.
                </p>
            </div>
        )
    }

    return (
        <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in duration-500">

            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <Sparkles className="h-6 w-6 text-deloitte" />
                        Strategic Insights
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                        Generate categorized insights from your data and KPI context.
                    </p>
                </div>
                <Button
                    onClick={generateInsights}
                    disabled={isGenerating}
                    className="bg-deloitte hover:bg-deloitte-dark text-white font-medium h-11"
                >
                    {isGenerating ? 'Generating Insights...' : 'Generate AI Insights'}
                </Button>
            </div>

            {/* Main Content Area */}
            <div className="min-h-[460px]">
                {!insights && !isGenerating && !error && (
                    <Card className="flex flex-col items-center justify-center py-16 px-4 text-center border-dashed border-2 bg-gray-50/50 dark:bg-gray-900/20">
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Ready for Insights</h3>
                        <p className="text-gray-500 dark:text-gray-400 max-w-md">
                            Click Generate AI Insights to fetch descriptive, predictive, sector, and recommendation categories.
                        </p>
                    </Card>
                )}

                {isGenerating && (
                    <Card className="p-8 text-sm text-gray-600 dark:text-gray-300">
                        Generating categorized insights...
                    </Card>
                )}

                {error && (
                    <Card className="flex flex-col items-center justify-center py-16 px-4 text-center border-red-100 dark:border-red-900/50 bg-red-50/50 dark:bg-red-900/10">
                        <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-full mb-4">
                            <AlertTriangle className="h-8 w-8 text-red-600 dark:text-red-400" />
                        </div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Insight Generation Failed</h3>
                        <p className="text-red-600 dark:text-red-300 max-w-sm mb-6">{error}</p>
                        <Button
                            onClick={generateInsights}
                            className="bg-deloitte hover:bg-deloitte-dark text-white font-medium h-10"
                        >
                            Retry
                        </Button>
                    </Card>
                )}

                {insights && !isGenerating && (
                    <Card className="p-6 bg-white dark:bg-gray-900 border-gray-100 dark:border-gray-800 ring-1 ring-gray-900/5 dark:ring-white/10">
                        {categoryEntries.length === 0 ? (
                            <p className="text-sm text-gray-500 dark:text-gray-400">No insight categories were returned.</p>
                        ) : (
                            <>
                                <div className="mb-5 flex flex-wrap gap-2">
                                    {categoryEntries.map((entry) => (
                                        <button
                                            key={entry.key}
                                            onClick={() => setActiveTab(entry.key)}
                                            className={`rounded-full px-3 py-1.5 text-sm font-medium border transition-colors ${effectiveActiveTab === entry.key
                                                ? 'bg-deloitte text-white border-deloitte'
                                                : 'bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-700'
                                                }`}
                                        >
                                            {entry.label}
                                        </button>
                                    ))}
                                </div>

                                {categoryEntries
                                    .filter((entry) => entry.key === effectiveActiveTab)
                                    .map((entry) => (
                                        <div key={entry.key} className="space-y-3">
                                            {entry.items.map((item, idx) => (
                                                <div key={`${entry.key}-${idx}`} className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-gray-50/60 dark:bg-gray-800/50">
                                                    <p className="text-sm text-gray-700 dark:text-gray-200 whitespace-pre-wrap">{formatInsightItem(item)}</p>
                                                </div>
                                            ))}
                                        </div>
                                    ))}
                            </>
                        )}
                    </Card>
                )}
            </div>
        </div>
    )
}

function formatInsightItem(item: unknown): string {
    if (typeof item === 'string') {
        return item
    }

    if (item && typeof item === 'object') {
        const obj = item as Record<string, unknown>
        const preferred = ['insight', 'evidence', 'interpretation', 'action', 'question', 'summary']
        for (const key of preferred) {
            const value = obj[key]
            if (typeof value === 'string' && value.trim()) {
                return value
            }
        }

        return Object.entries(obj)
            .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${String(v)}`)
            .join(' | ')
    }

    return String(item)
}
