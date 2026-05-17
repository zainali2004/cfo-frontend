import { useState } from 'react'
import {
    Download, FileText, Presentation,
    CheckCircle2, Loader2, AlertCircle
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useDatasetStore } from '@/stores/useDatasetStore'
import type { ExportFormat } from '@/services/dataService'

// ── Export Option Config ──────────────────────────────

interface ExportOption {
    id: ExportFormat
    name: string
    description: string
    icon: React.ReactNode
    color: string
}

const EXPORT_OPTIONS: ExportOption[] = [
    {
        id: 'pptx',
        name: 'PowerPoint Presentation',
        description: 'Executive-ready slides with charts, insights, and key findings',
        icon: <Presentation className="h-6 w-6" />,
        color: 'text-orange-500',
    },
    {
        id: 'pdf',
        name: 'PDF Report',
        description: 'Full executive summary with charts and insights',
        icon: <FileText className="h-6 w-6" />,
        color: 'text-red-500',
    },
]

// ── Main Page ─────────────────────────────────────────

export function ExportPage() {
    // Store state — pure consumption, no inline business logic
    const data = useDatasetStore((s) => s.data)
    const rowCount = useDatasetStore((s) => s.rowCount)
    const insights = useDatasetStore((s) => s.insights)
    const visuals = useDatasetStore((s) => s.visuals)
    const agent5Done = useDatasetStore((s) => s.agent5Done)
    const exportingFormat = useDatasetStore((s) => s.exportingFormat)
    const lastExport = useDatasetStore((s) => s.lastExport)
    const exportError = useDatasetStore((s) => s.exportError)
    const runExport = useDatasetStore((s) => s.runExport)

    // Local UI state for completed checkmarks (purely visual, not domain state)
    const [completed, setCompleted] = useState<ExportFormat[]>([])

    // ── Empty State ──
    if (!data) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[400px] text-center p-8 animate-in fade-in zoom-in duration-500">
                <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-full mb-4">
                    <FileText className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">No Data Available</h3>
                <p className="text-gray-500 dark:text-gray-400 max-w-sm mt-2">
                    Please upload a dataset first to export reports.
                </p>
            </div>
        )
    }

    // ── Handle Export (delegates to store action) ──
    const handleExport = async (format: ExportFormat) => {
        await runExport(format)

        // Show checkmark for 3s (purely visual feedback, result is in store)
        setCompleted((prev) => [...prev, format])
        setTimeout(() => {
            setCompleted((prev) => prev.filter((f) => f !== format))
        }, 3000)
    }

    const chartCount = visuals.filter((v) => Boolean(v.image_b64) || (Array.isArray(v.chart_data) && v.chart_data.length > 0)).length

    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">

            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <Download className="h-6 w-6 text-deloitte" />
                    Export & Share
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                    Download your visual insights in your preferred format.
                </p>
            </div>

            {/* Report Preview Card */}
            <Card className="relative overflow-hidden bg-white dark:bg-gray-900 border-gray-100 dark:border-gray-800 ring-1 ring-gray-900/5 dark:ring-white/10">
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-deloitte to-green-400" />
                <div className="p-6 lg:p-8">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Report Preview</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatBox label="Data Rows" value={rowCount.toLocaleString()} />
                        <StatBox
                            label="Insights"
                            value={insights ? 'Generated' : 'Not Generated'}
                            valueColor={insights ? 'text-green-600 dark:text-green-400' : 'text-yellow-600 dark:text-yellow-400'}
                        />
                        <StatBox
                            label="Visuals"
                            value={chartCount > 0 ? `${chartCount} Generated` : (agent5Done ? 'No Charts Rendered' : 'Not Generated')}
                        />
                    </div>
                </div>
            </Card>

            {!agent5Done && (
                <div className="flex items-center gap-3 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl text-yellow-800 dark:text-yellow-300 text-sm">
                    <AlertCircle className="h-5 w-5 flex-shrink-0" />
                    <p>Generate Visual Insights first to enable export.</p>
                </div>
            )}

            {/* Error Banner */}
            {exportError && (
                <div className="flex items-center gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-300 text-sm">
                    <AlertCircle className="h-5 w-5 flex-shrink-0" />
                    <p>{exportError}</p>
                </div>
            )}

            {/* Export Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {EXPORT_OPTIONS.map((option) => {
                    const isDownloading = exportingFormat === option.id
                    const isCompleted = completed.includes(option.id)

                    return (
                        <Card
                            key={option.id}
                            className={`relative overflow-hidden p-8 flex flex-col items-center text-center gap-5 transition-all duration-300 hover:shadow-lg cursor-pointer group
                                bg-white dark:bg-gray-900 border-gray-100 dark:border-gray-800 ring-1 ring-gray-900/5 dark:ring-white/10
                                ${isCompleted ? 'ring-2 ring-green-500/50' : ''}
                            `}
                        >
                            <div className={`p-3 rounded-xl bg-gray-50 dark:bg-gray-800 group-hover:scale-110 transition-transform ${option.color}`}>
                                {option.icon}
                            </div>
                            <div>
                                <h4 className="font-semibold text-gray-900 dark:text-gray-100">{option.name}</h4>
                                <p className="text-sm text-gray-500 dark:text-gray-300 mt-1">{option.description}</p>
                            </div>
                            <Button
                                onClick={() => handleExport(option.id)}
                                disabled={isDownloading || exportingFormat !== null || !agent5Done || chartCount === 0}
                                className="mt-auto w-full bg-deloitte hover:bg-deloitte/90 text-white"
                                size="sm"
                            >
                                {isDownloading ? (
                                    <>
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                        Generating...
                                    </>
                                ) : isCompleted ? (
                                    <>
                                        <CheckCircle2 className="h-4 w-4 mr-2" />
                                        Exported
                                    </>
                                ) : (
                                    <>
                                        <Download className="h-4 w-4 mr-2" />
                                        Export
                                    </>
                                )}
                            </Button>
                        </Card>
                    )
                })}
            </div>

            {/* Last Export Info */}
            {lastExport && (
                <div className="text-center text-xs text-gray-400 dark:text-gray-500 pt-2">
                    Last export: <strong>{lastExport.fileName}</strong> ({lastExport.format.toUpperCase()}) at {new Date(lastExport.generatedAt).toLocaleTimeString()}
                </div>
            )}

        </div>
    )
}

// ── Helper ──────────────────────────

function StatBox({ label, value, valueColor }: { label: string; value: string; valueColor?: string }) {
    return (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
            <p className={`text-lg font-bold mt-1 ${valueColor || 'text-gray-900 dark:text-white'}`}>{value}</p>
        </div>
    )
}
