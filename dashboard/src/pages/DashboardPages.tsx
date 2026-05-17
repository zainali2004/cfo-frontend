import { useState } from 'react'
import { Loader2, Paperclip, X } from 'lucide-react'
import { toast } from 'sonner'
import { Dropzone } from '@/components/Dropzone'
import { Button } from '@/components/ui/button'
import { runDataProfiler, uploadFile } from '@/services/dataService'
import { useDatasetStore } from '@/stores/useDatasetStore'
import { DetectedTablesPreview } from '@/components/DetectedTablesPreview'
import { DataMetadataCard } from '@/components/DataMetadataCard'

export function UploadPage() {
    const setData = useDatasetStore((s) => s.setData)
    const setProfileState = useDatasetStore((s) => s.setProfileState)
    const dataset = useDatasetStore((s) => s.uploadPayload)
    const sector = useDatasetStore((s) => s.sector)
    const setSector = useDatasetStore((s) => s.setSector)
    const customKpis = useDatasetStore((s) => s.customKpis)
    const setCustomKpis = useDatasetStore((s) => s.setCustomKpis)
    const file = useDatasetStore((s) => s.file)
    const setFile = useDatasetStore((s) => s.setFile)
    const clearData = useDatasetStore((s) => s.clearData)

    const [sectorInput, setSectorInput] = useState(sector ?? '')
    const [kpiInput, setKpiInput] = useState('')
    // File state moved to store
    const [isUploading, setIsUploading] = useState(false)

    // Wrapper to handle file selection AND clearing data if file is removed
    const handleFileSelect = (newFile: File | null) => {
        setFile(newFile)
        if (!newFile) {
            clearData()
        }
    }

    const handleUpload = async () => {
        if (!file) {
            toast.error("Please select a file first")
            return
        }

        try {
            setIsUploading(true)
            const formData = new FormData()
            formData.append('file', file)
            if (sector) formData.append('sector', sector)
            if (customKpis.length > 0) formData.append('custom_kpis', JSON.stringify(customKpis))

            const response = await uploadFile(formData)

            // Keep Streamlit-like flow: Agent 1 runs once immediately after upload.
            const profileResp = await runDataProfiler({
                raw_preview: response.raw_preview,
                extracted_text: response.extracted_text,
            })

            // Commit data only after upload + profiling are both successful.
            // This keeps next tabs and previews locked until the success toast.
            setData(response)
            setProfileState({
                profiles: profileResp.profiles ?? [],
                isProfiling: false,
                profileError: null,
                agent1Done: true,
            })
            toast.success("Data uploaded successfully")
        } catch (error) {
            toast.error("Upload failed. Please try again.")
            console.error(error)
        } finally {
            setIsUploading(false)
        }
    }

    const handleSectorChange = (value: string) => {
        setSectorInput(value)
        setSector(value || null)
    }

    const handleAddKpi = () => {
        const trimmed = kpiInput.trim()
        if (!trimmed) return
        // Prevent duplicates (case-insensitive)
        if (customKpis.some((k) => k.toLowerCase() === trimmed.toLowerCase())) {
            setKpiInput('')
            return
        }
        setCustomKpis([...customKpis, trimmed])
        setKpiInput('')
    }

    const handleRemoveKpi = (index: number) => {
        setCustomKpis(customKpis.filter((_, i) => i !== index))
    }

    const handleKpiKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault()
            handleAddKpi()
        }
        // Backspace on empty input removes last chip
        if (e.key === 'Backspace' && kpiInput === '' && customKpis.length > 0) {
            handleRemoveKpi(customKpis.length - 1)
        }
    }

    return (
        <div className="py-12 max-w-2xl mx-auto">
            <div className="text-center mb-10">
                <h2 className="text-2xl font-semibold text-gray-800 dark:text-white mb-2">Upload</h2>
                <p className="text-gray-500 dark:text-gray-400">Upload your dataset for analysis.</p>
            </div>

            <div className="grid grid-cols-1 gap-6">
                {/* File Upload Zone */}
                <div className="space-y-4">
                    <Dropzone
                        onFileSelect={handleFileSelect}
                        selectedFile={file}
                        disabled={isUploading}
                    />

                    {file && (
                        <Button
                            onClick={handleUpload}
                            disabled={isUploading}
                            className="w-full bg-deloitte hover:bg-deloitte-dark text-white font-medium h-12 text-base transition-all"
                        >
                            {isUploading ? (
                                <div className="flex items-center gap-2">
                                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                                    Processing File...
                                </div>
                            ) : (
                                "Process Dataset"
                            )}
                        </Button>
                    )}
                </div>
            </div>

            {dataset && (
                <div className="mt-8 space-y-4 rounded-xl border border-gray-200 bg-white p-5 dark:border-gray-700 dark:bg-gray-800">
                    <div>
                        <h3 className="text-base font-semibold text-gray-800 dark:text-gray-100">Quick Preview</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                            Uploaded as <span className="font-medium uppercase">{dataset.file_type}</span>. Review this preview, then continue to the Data Profiling tab.
                        </p>
                    </div>

                    {dataset.extracted_text && (
                        <div className="rounded-md border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-900/40 dark:text-gray-200">
                            <p className="mb-2 font-medium">Extracted Text Preview</p>
                            <pre className="whitespace-pre-wrap break-words text-xs text-gray-600 dark:text-gray-300">
                                {dataset.file_type === 'msg'
                                    ? dataset.extracted_text
                                    : `${dataset.extracted_text.slice(0, 1200)}${dataset.extracted_text.length > 1200 ? '...' : ''}`}
                            </pre>
                        </div>
                    )}

                    {dataset.attachments_preview && dataset.attachments_preview.length > 0 && (
                        <div className="space-y-2 rounded-md border border-gray-200 p-3 dark:border-gray-700">
                            <p className="text-sm font-medium text-gray-800 dark:text-gray-100">Email Attachments</p>
                            {dataset.attachments_preview.map((att, idx) => (
                                <div key={`${att.name}-${idx}`} className="rounded-md bg-gray-50 p-2 text-xs text-gray-700 dark:bg-gray-900/40 dark:text-gray-200">
                                    <p className="font-medium">
                                        <Paperclip className="mr-1 inline h-3.5 w-3.5" />
                                        {att.name} ({att.type})
                                        {att.is_primary ? ' - primary data source' : ''}
                                    </p>
                                    {typeof att.table_count === 'number' && <p>Detected tables: {att.table_count}</p>}
                                    {att.text_preview && <p className="mt-1">{att.text_preview}</p>}
                                </div>
                            ))}
                        </div>
                    )}

                    <DetectedTablesPreview />
                </div>
            )}

            {/* ── Sector Input ─────────────────────────────────── */}
            <div className="mt-10 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 transition-colors">
                <label
                    htmlFor="sector-input"
                    className="block text-sm font-semibold text-gray-700 dark:text-gray-200 mb-1"
                >
                    Sector
                    <span className="ml-2 text-xs font-normal text-gray-400 dark:text-gray-500">Optional</span>
                </label>
                <p className="text-xs text-gray-400 dark:text-gray-500 mb-3">
                    Specify your industry sector, or leave blank to let the system auto-detect it from your data.
                </p>
                <input
                    id="sector-input"
                    type="text"
                    value={sectorInput}
                    onChange={(e) => handleSectorChange(e.target.value)}
                    placeholder="e.g. Banking, Healthcare, Technology"
                    className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900 px-4 py-2.5 text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-deloitte/40 focus:border-deloitte transition-colors"
                />
            </div>

            {/* ── Custom KPI Tags Input ────────────────────────── */}
            <div className="mt-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 transition-colors">
                <label
                    htmlFor="kpi-input"
                    className="block text-sm font-semibold text-gray-700 dark:text-gray-200 mb-1"
                >
                    Additional KPIs
                    <span className="ml-2 text-xs font-normal text-gray-400 dark:text-gray-500">Optional</span>
                </label>
                <p className="text-xs text-gray-400 dark:text-gray-500 mb-3">
                    Add any specific KPIs you'd like calculated. The system will also include default KPIs based on your sector.
                </p>

                {/* Chip container + input */}
                <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900 px-3 py-2 focus-within:ring-2 focus-within:ring-deloitte/40 focus-within:border-deloitte transition-colors">
                    {customKpis.map((kpi, index) => (
                        <span
                            key={index}
                            className="inline-flex items-center gap-1 rounded-full bg-deloitte/10 dark:bg-deloitte/20 text-deloitte text-xs font-medium px-2.5 py-1"
                        >
                            {kpi}
                            <button
                                type="button"
                                onClick={() => handleRemoveKpi(index)}
                                className="rounded-full p-0.5 hover:bg-deloitte/20 dark:hover:bg-deloitte/30 transition-colors cursor-pointer"
                                aria-label={`Remove ${kpi}`}
                            >
                                <X className="h-3 w-3" />
                            </button>
                        </span>
                    ))}
                    <input
                        id="kpi-input"
                        type="text"
                        value={kpiInput}
                        onChange={(e) => setKpiInput(e.target.value)}
                        onKeyDown={handleKpiKeyDown}
                        placeholder={customKpis.length === 0 ? 'Type a KPI and press Enter' : 'Add another...'}
                        className="flex-1 min-w-[140px] bg-transparent text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 outline-none border-none py-0.5"
                    />
                </div>

                {/* Hint */}
                {customKpis.length > 0 && (
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                        {customKpis.length} custom KPI{customKpis.length !== 1 ? 's' : ''} added · Press Backspace to remove last
                    </p>
                )}
            </div>
        </div>
    )
}

export function DataProfilePage() {
    const uploadPayload = useDatasetStore((s) => s.uploadPayload)
    const profiles = useDatasetStore((s) => s.profiles)
    const isProfiling = useDatasetStore((s) => s.isProfiling)
    const profileError = useDatasetStore((s) => s.profileError)
    const agent1Done = useDatasetStore((s) => s.agent1Done)
    const setProfileState = useDatasetStore((s) => s.setProfileState)

    const handleRunDataProfiler = async () => {
        if (!uploadPayload) {
            toast.error('Please upload a dataset first')
            return
        }

        try {
            setProfileState({ isProfiling: true, profileError: null })
            const data = await runDataProfiler({
                raw_preview: uploadPayload.raw_preview,
                extracted_text: uploadPayload.extracted_text,
            })
            setProfileState({
                profiles: data.profiles ?? [],
                isProfiling: false,
                profileError: null,
                agent1Done: true,
            })
            toast.success('Data profiling completed')
        } catch (error) {
            console.error(error)
            setProfileState({
                isProfiling: false,
                profileError: 'Profiling failed. Please try again.',
            })
            toast.error('Profiling failed. Please try again.')
        }
    }

    return (
        <div className="py-8 max-w-5xl mx-auto space-y-8 px-4">
            <div>
                <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">Data Profile</h2>
                <p className="text-gray-500 dark:text-gray-400">Review the structure of your uploaded dataset and run Agent 1 profiling.</p>
            </div>

            <Button
                onClick={handleRunDataProfiler}
                disabled={isProfiling || !uploadPayload}
                className="bg-deloitte hover:bg-deloitte-dark text-white font-medium h-11"
            >
                {isProfiling ? (
                    <span className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Running Data Profiler...
                    </span>
                ) : (
                    agent1Done ? 'Re-run Data Profiler' : 'Run Data Profiler'
                )}
            </Button>

            {agent1Done && (
                <div className="rounded-md border border-green-200 bg-green-50 p-3 text-sm font-medium text-green-800 dark:border-green-800/40 dark:bg-green-900/20 dark:text-green-300">
                    Agent 1 completed.
                </div>
            )}

            {profileError && (
                <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800/40 dark:bg-red-900/20 dark:text-red-300">
                    {profileError}
                </div>
            )}

            <DataMetadataCard />
            <DetectedTablesPreview />

            {profiles.length > 0 && (
                <div className="space-y-4">
                    {profiles.map((profile, index) => {
                        const tableName = typeof profile.table_name === 'string' ? profile.table_name : `Profile ${index + 1}`
                        return (
                            <div key={`${tableName}-${index}`} className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
                                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">{tableName}</h3>
                                <div className="mt-3 grid gap-3 md:grid-cols-2 text-sm text-gray-700 dark:text-gray-300">
                                    <p><span className="font-semibold">Rows:</span> {String(profile.rows ?? 'N/A')}</p>
                                    <p><span className="font-semibold">Columns:</span> {String(profile.cols ?? 'N/A')}</p>
                                </div>
                                {Boolean(profile.business_context_clues) && (
                                    <div className="mt-3 text-sm text-gray-700 dark:text-gray-300">
                                        <p className="font-semibold">Business Context Clues</p>
                                        <p>{String(profile.business_context_clues)}</p>
                                    </div>
                                )}
                                {Boolean(profile.recommended_preprocessing) && (
                                    <div className="mt-3 text-sm text-gray-700 dark:text-gray-300">
                                        <p className="font-semibold">Recommended Preprocessing</p>
                                        <p>{String(profile.recommended_preprocessing)}</p>
                                    </div>
                                )}
                                {Boolean(profile.data_structure_insights) && (
                                    <div className="mt-3 text-sm text-gray-700 dark:text-gray-300">
                                        <p className="font-semibold">Structural Insights</p>
                                        <p>{String(profile.data_structure_insights)}</p>
                                    </div>
                                )}
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

export { SectorPage } from './SectorPage'

export { KPIPage } from './KPIPage'

export { InsightsPage } from '@/pages/InsightsPage'

export { VisualPage } from '@/pages/VisualPage'

export { ExportPage } from '@/pages/ExportPage'
export { ChatPage } from '@/pages/ChatPage'
