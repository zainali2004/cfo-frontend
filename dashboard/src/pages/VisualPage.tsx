import { useMemo } from 'react'
import { LineChart, Loader2, AlertCircle, FileText, Image as ImageIcon, Sparkles } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useDatasetStore } from '@/stores/useDatasetStore'
import type { VisualInsight } from '@/services/dataService'
import {
    ResponsiveContainer,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    LineChart as ReLineChart,
    Line,
    BarChart,
    Bar,
    AreaChart,
    Area,
    PieChart,
    Pie,
    Cell,
    ComposedChart,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ScatterChart,
    Scatter,
    RadialBarChart,
    RadialBar,
    Treemap,
    FunnelChart,
    Funnel,
    LabelList,
} from 'recharts'

const SERIES_COLORS = ['#0a6b79', '#19a974', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

function normalizeSeries(visual: VisualInsight): Array<{ key: string; label: string }> {
    if (Array.isArray(visual.series) && visual.series.length > 0) {
        return visual.series
            .filter((s) => typeof s?.key === 'string' && s.key.trim().length > 0)
            .map((s) => ({ key: s.key, label: s.label || s.key }))
    }

    if (Array.isArray(visual.y_keys) && visual.y_keys.length > 0) {
        return visual.y_keys.map((key) => ({ key, label: key }))
    }

    const firstRow = Array.isArray(visual.chart_data) && visual.chart_data.length > 0 ? visual.chart_data[0] : null
    if (!firstRow || typeof firstRow !== 'object') return []

    const xKey = visual.x_key || 'name'
    return Object.keys(firstRow)
        .filter((k) => k !== xKey)
        .map((k) => ({ key: k, label: k }))
}

// Coerce a value to a finite number, tolerating the string formats the LLM often
// returns: "1,234", "$1.2M"-style symbols, "12%", whitespace, and accounting
// negatives like "(500)". Returns null when the value is not numeric.
function toNumber(value: unknown): number | null {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null
    if (typeof value !== 'string') return null

    let s = value.trim()
    if (s === '') return null

    // Strip currency symbols, thousands separators, percent signs, and spaces
    // first, so accounting negatives with a suffix (e.g. "(5)%") still parse.
    s = s.replace(/[,%\s$€£¥]/g, '')

    let sign = 1
    if (/^\(.*\)$/.test(s)) {
        sign = -1 // accounting negative, e.g. "(1,234)"
        s = s.slice(1, -1)
    }
    if (s === '' || s === '-' || s === '.') return null

    const n = Number(s)
    return Number.isFinite(n) ? sign * n : null
}

// Build a render-ready copy of chart_data with value columns coerced to numbers,
// and report which keys actually hold plottable numeric data.
function prepareChartData(
    rawData: Record<string, unknown>[],
    xKey: string,
    candidateKeys: string[],
): { data: Record<string, unknown>[]; numericKeys: string[] } {
    if (!rawData.length) return { data: [], numericKeys: [] }

    // Candidate value keys = declared series keys ∪ every non-x key seen in the data.
    const keySet = new Set<string>(candidateKeys.filter(Boolean))
    rawData.forEach((row) => {
        if (row && typeof row === 'object') {
            Object.keys(row).forEach((k) => {
                if (k !== xKey) keySet.add(k)
            })
        }
    })

    const data = rawData.map((row) => {
        const next: Record<string, unknown> = { ...row }
        keySet.forEach((k) => {
            const num = toNumber(row?.[k])
            if (num !== null) next[k] = num
        })
        return next
    })

    const numericKeys = Array.from(keySet).filter((k) =>
        data.some((row) => typeof row[k] === 'number' && Number.isFinite(row[k] as number)),
    )

    return { data, numericKeys }
}

function ChartEmptyState({ message }: { message: string }) {
    return (
        <div className="h-40 w-full flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/40 text-center px-4">
            <AlertCircle className="h-5 w-5 text-gray-400" />
            <p className="text-sm text-gray-500 dark:text-gray-400">{message}</p>
        </div>
    )
}

function VisualChart({ visual }: { visual: VisualInsight }) {
    const rawData = (Array.isArray(visual.chart_data) ? visual.chart_data : []) as Record<string, unknown>[]
    const chartType = (visual.recharts_type || '').toLowerCase()
    const declaredSeries = normalizeSeries(visual)

    // Resolve the x-axis key: use the declared one if it exists in the data,
    // otherwise fall back to the first non-numeric column, then 'name'.
    const firstRow = rawData.find((r) => r && typeof r === 'object') || null
    const rowKeys = firstRow ? Object.keys(firstRow) : []
    let xKey = visual.x_key || 'name'
    if (firstRow && !(xKey in firstRow)) {
        const nonNumeric = rowKeys.find((k) => toNumber(firstRow[k]) === null)
        xKey = nonNumeric || rowKeys[0] || 'name'
    }

    const { data: chartData, numericKeys } = prepareChartData(
        rawData,
        xKey,
        declaredSeries.map((s) => s.key),
    )

    // Keep only declared series whose key actually carries numeric data; if none
    // survive, fall back to every detected numeric column so the chart isn't empty.
    let series = declaredSeries.filter((s) => numericKeys.includes(s.key))
    if (series.length === 0) {
        series = numericKeys.map((k) => ({ key: k, label: k }))
    }

    if (!chartType) return null
    if (!rawData.length) return <ChartEmptyState message="No data was generated for this chart." />
    if (numericKeys.length === 0) {
        return <ChartEmptyState message="This chart has no plottable numeric values." />
    }

    if (chartType === 'pie') {
        const primary = series[0]?.key || 'value'
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Tooltip />
                        <Legend />
                        <Pie data={chartData} dataKey={primary} nameKey={xKey} outerRadius={110} label>
                            {chartData.map((_, idx) => (
                                <Cell key={`cell-${idx}`} fill={SERIES_COLORS[idx % SERIES_COLORS.length]} />
                            ))}
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'bar') {
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {series.map((s, idx) => (
                            <Bar key={s.key} dataKey={s.key} name={s.label} fill={SERIES_COLORS[idx % SERIES_COLORS.length]} />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'area') {
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {series.map((s, idx) => (
                            <Area
                                key={s.key}
                                type="monotone"
                                dataKey={s.key}
                                name={s.label}
                                stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                                fill={SERIES_COLORS[idx % SERIES_COLORS.length]}
                                fillOpacity={0.28}
                            />
                        ))}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'composed') {
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {series.map((s, idx) =>
                            idx === series.length - 1 ? (
                                <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={SERIES_COLORS[idx % SERIES_COLORS.length]} strokeWidth={2.2} dot={false} />
                            ) : (
                                <Bar key={s.key} dataKey={s.key} name={s.label} fill={SERIES_COLORS[idx % SERIES_COLORS.length]} />
                            ),
                        )}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'radar') {
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={chartData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey={xKey} />
                        <PolarRadiusAxis />
                        <Tooltip />
                        <Legend />
                        {series.map((s, idx) => (
                            <Radar
                                key={s.key}
                                dataKey={s.key}
                                name={s.label}
                                stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                                fill={SERIES_COLORS[idx % SERIES_COLORS.length]}
                                fillOpacity={0.25}
                            />
                        ))}
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'scatter') {
        const xNumericKey = numericKeys[0] || series[0]?.key
        const yNumericKey = numericKeys[1] || series[1]?.key || series[0]?.key

        if (!xNumericKey || !yNumericKey) return null

        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey={xNumericKey} name={xNumericKey} />
                        <YAxis type="number" dataKey={yNumericKey} name={yNumericKey} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        <Scatter name={`${xNumericKey} vs ${yNumericKey}`} data={chartData} fill={SERIES_COLORS[0]} />
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'radialbar') {
        const primary = series[0]?.key || numericKeys[0]
        if (!primary) return null

        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart innerRadius="20%" outerRadius="90%" data={chartData} startAngle={180} endAngle={0}>
                        <Tooltip />
                        <Legend />
                        <RadialBar dataKey={primary} name={primary} label={{ position: 'insideStart', fill: '#ffffff' }}>
                            {chartData.map((_, idx) => (
                                <Cell key={`radial-cell-${idx}`} fill={SERIES_COLORS[idx % SERIES_COLORS.length]} />
                            ))}
                        </RadialBar>
                    </RadialBarChart>
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'treemap') {
        const sizeKey = series[0]?.key || numericKeys[0] || 'value'
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <Treemap data={chartData} dataKey={sizeKey} nameKey={xKey} stroke="#ffffff" fill={SERIES_COLORS[0]} />
                </ResponsiveContainer>
            </div>
        )
    }

    if (chartType === 'funnel') {
        const valueKey = series[0]?.key || numericKeys[0] || 'value'
        return (
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <FunnelChart>
                        <Tooltip />
                        <Legend />
                        <Funnel dataKey={valueKey} data={chartData} nameKey={xKey} isAnimationActive>
                            <LabelList position="right" fill="#374151" stroke="none" dataKey={xKey} />
                        </Funnel>
                    </FunnelChart>
                </ResponsiveContainer>
            </div>
        )
    }

    return (
        <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ReLineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey={xKey} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {series.map((s, idx) => (
                        <Line
                            key={s.key}
                            type="monotone"
                            dataKey={s.key}
                            name={s.label}
                            stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                            strokeWidth={2.2}
                            dot={false}
                        />
                    ))}
                </ReLineChart>
            </ResponsiveContainer>
        </div>
    )
}

export function VisualPage() {
    const data = useDatasetStore((s) => s.data)
    const insights = useDatasetStore((s) => s.insights)
    const visuals = useDatasetStore((s) => s.visuals)
    const isGeneratingVisuals = useDatasetStore((s) => s.isGeneratingVisuals)
    const visualsError = useDatasetStore((s) => s.visualsError)
    const runVisualGeneration = useDatasetStore((s) => s.runVisualGeneration)

    const handleGenerateVisuals = async () => {
        try {
            await runVisualGeneration()
        } catch (err) {
            console.error('Failed to generate visuals:', err)
        }
    }

    const readyVisuals = useMemo(
        () => visuals.filter((v) => Boolean(v.image_b64) || Boolean(v.error) || (Array.isArray(v.chart_data) && v.chart_data.length > 0)),
        [visuals],
    )

    if (!data) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[400px] text-center p-8 animate-in fade-in zoom-in duration-500">
                <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-full mb-4">
                    <FileText className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">No Data Available</h3>
                <p className="text-gray-500 dark:text-gray-400 max-w-sm mt-2">
                    Please upload a dataset first to view visualizations.
                </p>
            </div>
        )
    }

    return (
        <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <LineChart className="h-6 w-6 text-deloitte" />
                        Visual Insights
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                        Generate chart-ready visuals from the latest insights and uploaded data.
                    </p>
                </div>

                <Button
                    onClick={handleGenerateVisuals}
                    disabled={isGeneratingVisuals}
                    className="bg-deloitte hover:bg-deloitte/90 text-white"
                >
                    {isGeneratingVisuals ? (
                        <>
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            Generating Visual Insights...
                        </>
                    ) : (
                        <>
                            <Sparkles className="h-4 w-4 mr-2" />
                            Generate Visual Insights
                        </>
                    )}
                </Button>
            </div>

            {!insights && (
                <Card className="p-6 border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-800">
                    <p className="text-sm text-yellow-800 dark:text-yellow-300">
                        Visual generation will create insights automatically if they are not available yet.
                    </p>
                </Card>
            )}

            {visualsError && (
                <div className="flex items-center gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-300 text-sm">
                    <AlertCircle className="h-5 w-5 flex-shrink-0" />
                    <p>{visualsError}</p>
                </div>
            )}

            {isGeneratingVisuals && readyVisuals.length === 0 && (
                <Card className="p-8 flex flex-col items-center gap-3">
                    <Loader2 className="h-7 w-7 animate-spin text-deloitte" />
                    <p className="text-sm text-gray-600 dark:text-gray-300">Building visual insights from your data...</p>
                </Card>
            )}

            {!isGeneratingVisuals && readyVisuals.length === 0 && insights && (
                <Card className="p-10 text-center">
                    <ImageIcon className="h-8 w-8 text-gray-400 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">No Visual Insights Generated Yet</h3>
                    <p className="text-gray-500 dark:text-gray-400 mt-2">Use the button above to generate visuals.</p>
                </Card>
            )}

            <div className="space-y-6">
                {readyVisuals.map((visual, index) => (
                    <Card key={visual.chart_id || `${visual.insight_text}-${index}`} className="overflow-hidden border-gray-200 dark:border-gray-800">
                        <div className="p-5 space-y-3">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                                {index + 1}. {visual.chart_title || visual.insight_text || 'Insight'}
                            </h3>

                            {visual.derived_signal && (
                                <p className="text-sm text-gray-700 dark:text-gray-300">
                                    <span className="font-semibold">Statement:</span> {visual.derived_signal}
                                </p>
                            )}

                            {visual.key_driver && (
                                <div className="flex items-start gap-2 rounded-lg border border-deloitte/30 bg-deloitte/5 dark:bg-deloitte/10 p-3">
                                    <Sparkles className="h-4 w-4 text-deloitte mt-0.5 flex-shrink-0" />
                                    <p className="text-sm text-gray-800 dark:text-gray-200">
                                        <span className="font-semibold text-deloitte">Key driver:</span> {visual.key_driver}
                                    </p>
                                </div>
                            )}

                            {visual.why_this_chart && (
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                    <span className="font-semibold">Why this visual:</span> {visual.why_this_chart}
                                </p>
                            )}

                            {visual.error && (
                                <div className="text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
                                    {visual.error}
                                </div>
                            )}

                            {visual.image_b64 && (
                                <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 p-2">
                                    <img
                                        src={`data:image/png;base64,${visual.image_b64}`}
                                        alt={visual.insight_text || `Visual ${index + 1}`}
                                        className="w-full h-auto rounded-lg"
                                    />
                                </div>
                            )}

                            {!visual.image_b64 && <VisualChart visual={visual} />}

                            {Array.isArray(visual.assumptions) && visual.assumptions.length > 0 && (
                                <div className="rounded-md border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-3">
                                    <p className="text-xs font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300 mb-2">Assumptions</p>
                                    <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700 dark:text-gray-300">
                                        {visual.assumptions.map((a, i) => (
                                            <li key={`${a}-${i}`}>{a}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </Card>
                ))}
            </div>
        </div>
    )
}
