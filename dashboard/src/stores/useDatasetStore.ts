import { create } from 'zustand'
import { exportVisualReport, generateInsights, getKPIs, getSectorBenchmarks, getVisualInsights, getVisualizationData, type ChartData, type DatasetResponse, type DetectedTable, type ExportFormat, type ExportResult, type InsightsByCategory, type KPIConcept, type SectorBenchmarks, type VisualInsight } from '@/services/dataService'
import type { ChatAttachment, ChatMessage } from '@/services/chatService'

interface DatasetState {
    // Raw data from backend
    data: Record<string, unknown>[] | null
    // Metadata
    rowCount: number
    columnCount: number
    missingValues: number
    columns: string[]
    // Detected Tables (Multi-table support)
    tables: DetectedTable[]
    // Sector (user-provided or auto-detected)
    sector: string | null
    // Custom KPIs added by user (on top of backend defaults)
    customKpis: string[]
    // Uploaded File (Persisted for session)
    file: File | null
    // Dataset ID (from upload response, used by Insights & Charts APIs)
    datasetId: string | null
    uploadPayload: DatasetResponse | null

    // Agent 1 profile state
    profiles: Record<string, unknown>[]
    isProfiling: boolean
    profileError: string | null
    agent1Done: boolean
    domainInfo: Record<string, unknown> | null
    agent2Done: boolean
    sectorBenchmarks: SectorBenchmarks | null
    isSectorDetecting: boolean
    sectorError: string | null
    sectorRunInFlight: Promise<void> | null

    // Agent 3 KPI state
    kpiConcepts: KPIConcept[]
    isKpiCalculating: boolean
    kpiError: string | null
    agent3Done: boolean
    kpiRunInFlight: Promise<void> | null

    // Agent 4 Insights state
    insights: InsightsByCategory | null
    isGeneratingInsights: boolean
    insightsError: string | null
    agent4Done: boolean
    insightsRunInFlight: Promise<void> | null
    generateInsights: () => Promise<void>

    // Agent 5 Visual state
    visuals: VisualInsight[]
    isGeneratingVisuals: boolean
    visualsError: string | null
    agent5Done: boolean
    visualsRunInFlight: Promise<void> | null
    runVisualGeneration: () => Promise<void>

    // Charts State
    charts: ChartData | null
    isLoadingCharts: boolean
    chartsError: string | null
    loadCharts: () => Promise<void>

    // Export State
    exportingFormat: ExportFormat | null
    lastExport: ExportResult | null
    exportError: string | null
    runExport: (format: ExportFormat) => Promise<void>

    // Chat State
    chatMessages: ChatMessage[]
    chatInputValue: string
    chatAttachments: Record<number, ChatAttachment[]>
    setChatMessages: (messages: ChatMessage[]) => void
    setChatInputValue: (value: string) => void
    setChatAttachments: (attachments: Record<number, ChatAttachment[]>) => void
    clearChatState: () => void

    // Actions
    setData: (dataset: DatasetResponse) => void
    setProfileState: (args: { profiles?: Record<string, unknown>[]; isProfiling?: boolean; profileError?: string | null; agent1Done?: boolean }) => void
    setSectorState: (args: { domainInfo?: Record<string, unknown> | null; agent2Done?: boolean; sector?: string | null; sectorBenchmarks?: SectorBenchmarks | null }) => void
    runSectorDetection: () => Promise<void>
    runKpiCalculation: () => Promise<void>
    setSector: (sector: string | null) => void
    setCustomKpis: (kpis: string[]) => void
    setFile: (file: File | null) => void
    clearData: () => void
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
    data: null,
    rowCount: 0,
    columnCount: 0,
    missingValues: 0,
    columns: [],
    tables: [],
    sector: null,
    customKpis: [],
    file: null,
    datasetId: null,
    uploadPayload: null,

    profiles: [],
    isProfiling: false,
    profileError: null,
    agent1Done: false,
    domainInfo: null,
    agent2Done: false,
    sectorBenchmarks: null,
    isSectorDetecting: false,
    sectorError: null,
    sectorRunInFlight: null,

    kpiConcepts: [],
    isKpiCalculating: false,
    kpiError: null,
    agent3Done: false,
    kpiRunInFlight: null,

    // Agent 4 Insights state
    insights: null,
    isGeneratingInsights: false,
    insightsError: null,
    agent4Done: false,
    insightsRunInFlight: null,

    visuals: [],
    isGeneratingVisuals: false,
    visualsError: null,
    agent5Done: false,
    visualsRunInFlight: null,

    // Charts State
    charts: null,
    isLoadingCharts: false,
    chartsError: null,

    // Export State
    exportingFormat: null,
    lastExport: null,
    exportError: null,

    // Chat State
    chatMessages: [],
    chatInputValue: '',
    chatAttachments: {},

    // Actions
    setData: (dataset) => set({
        data: dataset.data,
        rowCount: dataset.meta.rowCount,
        columnCount: dataset.meta.columnCount,
        missingValues: dataset.meta.missingValues,
        columns: dataset.meta.columns,
        tables: dataset.tables,
        datasetId: dataset.dataset_id,
        uploadPayload: dataset,
        profiles: [],
        isProfiling: false,
        profileError: null,
        agent1Done: false,
        domainInfo: null,
        agent2Done: false,
        sectorBenchmarks: null,
        isSectorDetecting: false,
        sectorError: null,
        sectorRunInFlight: null,
        kpiConcepts: [],
        isKpiCalculating: false,
        kpiError: null,
        agent3Done: false,
        kpiRunInFlight: null,
        insights: null,
        isGeneratingInsights: false,
        insightsError: null,
        agent4Done: false,
        insightsRunInFlight: null,
        visuals: [],
        isGeneratingVisuals: false,
        visualsError: null,
        agent5Done: false,
        visualsRunInFlight: null,
        // Reset charts when new data is loaded so they refresh
        charts: null,
        isLoadingCharts: false,
        chartsError: null,
        chatMessages: [],
        chatInputValue: '',
        chatAttachments: {},
    }),

    setProfileState: ({ profiles, isProfiling, profileError, agent1Done }) => set((state) => ({
        profiles: profiles ?? state.profiles,
        isProfiling: isProfiling ?? state.isProfiling,
        profileError: profileError ?? state.profileError,
        agent1Done: agent1Done ?? state.agent1Done,
    })),

    setSectorState: ({ domainInfo, agent2Done, sector, sectorBenchmarks }) => set((state) => ({
        domainInfo: domainInfo ?? state.domainInfo,
        agent2Done: agent2Done ?? state.agent2Done,
        sector: sector ?? state.sector,
        sectorBenchmarks: sectorBenchmarks ?? state.sectorBenchmarks,
    })),

    runSectorDetection: async () => {
        const existingTask = get().sectorRunInFlight
        if (existingTask) {
            return existingTask
        }

        const task = (async () => {
            const profiles = get().profiles
            if (!profiles.length) {
                set({ sectorError: 'Run Data Profiler first.' })
                return
            }

            set({ isSectorDetecting: true, sectorError: null })

            try {
                const sectorHint = get().sector
                const uploadPayload = get().uploadPayload

                const data = await getSectorBenchmarks({
                    data_profile: profiles,
                    memory: {
                        Agent1: profiles,
                        user_hints: (sectorHint ?? '').trim(),
                    },
                    extracted_text: uploadPayload?.extracted_text ?? null,
                    user_hints: (sectorHint ?? '').trim(),
                })

                set({
                    sector: data.sector,
                    domainInfo: {
                        domain: data.sector,
                        confidence: data.confidence,
                        market_sentiment: data.market_sentiment,
                        topics: data.topics,
                        definition: data.definition,
                        subdomain: data.subdomain,
                        wiki_url: data.wiki_url,
                    },
                    agent2Done: true,
                    sectorBenchmarks: data,
                    isSectorDetecting: false,
                    sectorError: null,
                    sectorRunInFlight: null,
                })
            } catch (error) {
                console.error('Failed to load benchmarks', error)
                set({
                    isSectorDetecting: false,
                    sectorError: 'Sector detection failed. Please try again.',
                    sectorRunInFlight: null,
                })
            }
        })()

        set({ sectorRunInFlight: task })
        return task
    },

    runKpiCalculation: async () => {
        const existingTask = get().kpiRunInFlight
        if (existingTask) {
            return existingTask
        }

        const task = (async () => {
            let domainInfo = get().domainInfo
            let profiles = get().profiles
            let uploadPayload = get().uploadPayload

            if ((!domainInfo || !profiles.length || !uploadPayload) && !get().agent2Done) {
                await get().runSectorDetection()
                domainInfo = get().domainInfo
                profiles = get().profiles
                uploadPayload = get().uploadPayload
            }

            if (!domainInfo || !profiles.length || !uploadPayload) {
                set({ kpiError: 'Run Sector Detector first.' })
                return
            }

            set({ isKpiCalculating: true, kpiError: null })

            try {
                const concepts = await getKPIs({
                    domain_info: domainInfo,
                    data_profile: profiles,
                    memory: {
                        Agent1: profiles,
                        Agent2: domainInfo,
                    },
                    user_metrics: get().customKpis,
                    key_metrics: get().customKpis.join(', '),
                    df_records: uploadPayload.data ?? null,
                    df_columns: uploadPayload.meta?.columns ?? null,
                    pdf_dfs_records: uploadPayload.dfs_records ?? null,
                    extracted_text: uploadPayload.extracted_text ?? null,
                })

                set({
                    kpiConcepts: concepts,
                    isKpiCalculating: false,
                    kpiError: null,
                    agent3Done: true,
                    kpiRunInFlight: null,
                })
            } catch (error) {
                console.error('Failed to calculate KPIs', error)
                set({
                    isKpiCalculating: false,
                    kpiError: 'KPI calculation failed. Please try again.',
                    kpiRunInFlight: null,
                })
            }
        })()

        set({ kpiRunInFlight: task })
        return task
    },

    setSector: (sector) => set({ sector }),
    setCustomKpis: (kpis) => set({ customKpis: kpis }),
    setFile: (file) => set({ file }),
    setChatMessages: (messages) => set({ chatMessages: messages }),
    setChatInputValue: (value) => set({ chatInputValue: value }),
    setChatAttachments: (attachments) => set({ chatAttachments: attachments }),
    clearChatState: () => set({ chatMessages: [], chatInputValue: '', chatAttachments: {} }),
    clearData: () => set({
        data: null,
        rowCount: 0,
        columnCount: 0,
        missingValues: 0,
        columns: [],
        tables: [],
        sector: null,
        customKpis: [],
        uploadPayload: null,
        profiles: [],
        isProfiling: false,
        profileError: null,
        agent1Done: false,
        domainInfo: null,
        agent2Done: false,
        sectorBenchmarks: null,
        isSectorDetecting: false,
        sectorError: null,
        sectorRunInFlight: null,
        kpiConcepts: [],
        isKpiCalculating: false,
        kpiError: null,
        agent3Done: false,
        kpiRunInFlight: null,
        insights: null,
        isGeneratingInsights: false,
        insightsError: null,
        agent4Done: false,
        insightsRunInFlight: null,
        visuals: [],
        isGeneratingVisuals: false,
        visualsError: null,
        agent5Done: false,
        visualsRunInFlight: null,
        charts: null,
        isLoadingCharts: false,
        chartsError: null,
        exportingFormat: null,
        lastExport: null,
        exportError: null,
        datasetId: null,
        file: null,
        chatMessages: [],
        chatInputValue: '',
        chatAttachments: {},
    }),

    generateInsights: async () => {
        const existingTask = get().insightsRunInFlight
        if (existingTask) {
            return existingTask
        }

        const task = (async () => {
            let domainInfo = get().domainInfo
            let profiles = get().profiles
            let kpiConcepts = get().kpiConcepts
            const uploadPayload = get().uploadPayload

            if ((!domainInfo || !profiles.length || !kpiConcepts.length) && !get().agent3Done) {
                await get().runKpiCalculation()
                domainInfo = get().domainInfo
                profiles = get().profiles
                kpiConcepts = get().kpiConcepts
            }

            if (!domainInfo || !profiles.length || !kpiConcepts.length) {
                set({ insightsError: 'Run KPI Calculator first.' })
                return
            }

            set({ isGeneratingInsights: true, insightsError: null })

            try {
                const insightsData = await generateInsights({
                    data_profile: profiles,
                    domain_info: domainInfo,
                    extracted_concepts: kpiConcepts,
                    memory: {
                        Agent1: profiles,
                        Agent2: domainInfo,
                        Agent3: kpiConcepts,
                    },
                    extracted_text: uploadPayload?.extracted_text ?? null,
                    user_hints: get().sector ?? '',
                    key_metrics: get().customKpis.join(', '),
                })

                set({
                    insights: insightsData,
                    isGeneratingInsights: false,
                    insightsError: null,
                    agent4Done: true,
                    insightsRunInFlight: null,
                })
            } catch (error) {
                console.error('Failed to generate insights:', error)
                set({
                    isGeneratingInsights: false,
                    insightsError: 'Failed to generate insights. Please try again.',
                    insightsRunInFlight: null,
                })
            }
        })()

        set({ insightsRunInFlight: task })
        return task
    },

    runVisualGeneration: async () => {
        const existingTask = get().visualsRunInFlight
        if (existingTask) {
            return existingTask
        }

        const task = (async () => {
            let insights = get().insights
            const domainInfo = get().domainInfo
            const uploadPayload = get().uploadPayload
            const kpiConcepts = get().kpiConcepts

            if (!domainInfo || !uploadPayload) {
                set({ visualsError: 'Run Sector & KPI first.' })
                return
            }

            // Show loading immediately — don't wait before setting this
            set({ isGeneratingVisuals: true, visualsError: null })

            if (!insights) {
                await get().generateInsights()
                insights = get().insights
            }

            if (!insights) {
                set({ isGeneratingVisuals: false, visualsError: 'Run Generate AI Insights first.' })
                return
            }

            try {
                const visuals = await getVisualInsights({
                    extracted_text: uploadPayload.extracted_text ?? null,
                    insights_json: { analysis: insights },
                    domain_info: domainInfo,
                    kpis: kpiConcepts,
                    df_records: uploadPayload.data ?? null,
                    df_columns: uploadPayload.meta?.columns ?? null,
                    dfs_records: uploadPayload.dfs_records ?? null,
                })

                set({
                    visuals,
                    isGeneratingVisuals: false,
                    visualsError: null,
                    agent5Done: true,
                    visualsRunInFlight: null,
                })
            } catch (error) {
                console.error('Failed to generate visuals:', error)
                const message = error instanceof Error && error.message.trim().length > 0
                    ? error.message
                    : 'Visual generation failed. Please try again.'
                set({
                    isGeneratingVisuals: false,
                    visualsError: message,
                    visualsRunInFlight: null,
                })
            }
        })()

        set({ visualsRunInFlight: task })
        return task
    },

    loadCharts: async () => {
        const datasetId = get().datasetId
        if (!get().data) return

        set({ isLoadingCharts: true, chartsError: null })

        try {
            const chartData = await getVisualizationData(datasetId || "mock-dataset-123")
            set({ charts: chartData, isLoadingCharts: false })
        } catch (error) {
            console.error("Failed to load chart data:", error)
            set({
                isLoadingCharts: false,
                chartsError: "Failed to load visualizations. Please try again."
            })
        }
    },

    runExport: async (format: ExportFormat) => {
        const payload = get().uploadPayload
        const insights = get().insights
        const domainInfo = get().domainInfo
        const visuals = get().visuals
        const kpis = get().kpiConcepts

        if (!payload || !insights || !domainInfo || visuals.length === 0) {
            set({ exportError: 'Generate visual insights first.' })
            return
        }

        set({ exportingFormat: format, exportError: null })

        try {
            const result = await exportVisualReport({
                format,
                extracted_text: payload.extracted_text ?? null,
                insights_json: { analysis: insights },
                domain_info: domainInfo,
                kpis,
                df_records: payload.data ?? null,
                df_columns: payload.meta?.columns ?? null,
                dfs_records: payload.dfs_records ?? null,
                pre_rendered_visuals: visuals,
            })

            // Trigger browser download immediately.
            const link = document.createElement('a')
            link.href = result.downloadUrl
            link.download = result.fileName
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)

            set({ lastExport: result, exportingFormat: null })
        } catch (error) {
            console.error("Failed to export report:", error)
            set({
                exportingFormat: null,
                exportError: "Failed to generate export. Please try again."
            })
        }
    }
}))


