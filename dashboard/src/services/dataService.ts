import api from './api'

// ── Types ──────────────────────────────────────────────

export interface DetectedTable {
    name: string
    rows: Record<string, unknown>[]
    columns: string[]
}

export type UploadFileType = 'csv' | 'excel' | 'pdf' | 'msg' | 'txt'

export interface AttachmentPreview {
    name: string
    type: string
    is_primary?: boolean
    table_count?: number
    text_preview?: string
    rows?: number
    cols?: number
    columns?: string[]
    records?: Record<string, unknown>[]
}

export interface DatasetResponse {
    data: Record<string, unknown>[]
    meta: DatasetMeta
    tables: DetectedTable[]
    dataset_id: string
    file_type: UploadFileType
    raw_preview?: string | null
    extracted_text?: string | null
    tables_json?: Record<string, unknown>[] | null
    dfs_records?: Record<string, Record<string, unknown>[]> | null
    shape?: { rows: number; cols: number } | null
    attachments_preview?: AttachmentPreview[] | null
}

export interface DatasetMeta {
    rowCount: number
    columnCount: number
    columns: string[]
    missingValues: number
}

interface UploadApiResponse {
    file_type: UploadFileType
    raw_preview?: string | null
    extracted_text?: string | null
    tables_json?: Record<string, unknown>[] | null
    dfs_records?: Record<string, Record<string, unknown>[]> | null
    records?: Record<string, unknown>[] | null
    columns?: string[] | null
    shape?: { rows: number; cols: number } | null
    attachments_preview?: AttachmentPreview[] | null
}


function countMissingValues(rows: Record<string, unknown>[]): number {
    return rows.reduce((acc, row) => {
        let rowMissing = 0
        Object.values(row).forEach((value) => {
            if (value === null || value === undefined || value === '') {
                rowMissing += 1
            }
        })
        return acc + rowMissing
    }, 0)
}

function mapUploadResponseToDataset(resp: UploadApiResponse): DatasetResponse {
    const tables: DetectedTable[] = []

    if ((resp.file_type === 'csv' || resp.file_type === 'excel') && resp.records?.length) {
        const columns = resp.columns ?? Object.keys(resp.records[0])
        tables.push({
            name: 'Uploaded Data',
            rows: resp.records,
            columns,
        })
    }

    if (resp.dfs_records) {
        Object.entries(resp.dfs_records).forEach(([name, rows]) => {
            if (!rows?.length) return
            tables.push({
                name,
                rows,
                columns: Object.keys(rows[0]),
            })
        })
    }

    const firstTable = tables[0]
    const firstRows = firstTable?.rows ?? []
    const firstColumns = firstTable?.columns ?? []

    return {
        data: firstRows,
        meta: {
            rowCount: resp.shape?.rows ?? firstRows.length,
            columnCount: resp.shape?.cols ?? firstColumns.length,
            columns: resp.columns ?? firstColumns,
            missingValues: countMissingValues(firstRows),
        },
        tables,
        dataset_id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `dataset-${Date.now()}`,
        file_type: resp.file_type,
        raw_preview: resp.raw_preview ?? null,
        extracted_text: resp.extracted_text ?? null,
        tables_json: resp.tables_json ?? null,
        dfs_records: resp.dfs_records ?? null,
        shape: resp.shape ?? null,
        attachments_preview: resp.attachments_preview ?? null,
    }
}

// ── Service ────────────────────────────────────────────

/**
 * Upload a file to the backend for processing.
 *
 * MOCK: Simulates 1.5s delay and returns dummy data.
 * REAL: Replace body with `return api.post('/upload', formData).then(res => res.data)`
 */
export async function uploadFile(formData: FormData): Promise<DatasetResponse> {
    const { data } = await api.post<UploadApiResponse>('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    })
    return mapUploadResponseToDataset(data)
}


export interface ProfileResult {
    profiles: Record<string, unknown>[]
}

export async function runDataProfiler(payload: { raw_preview?: string | null; extracted_text?: string | null }): Promise<ProfileResult> {
    const { data } = await api.post<ProfileResult>('/profile', {
        raw_preview: payload.raw_preview ?? null,
        extracted_text: payload.extracted_text ?? null,
    })
    return data
}

// ── Sector Knowledge (Phase 4) ─────────────────────────

export interface SectorBenchmarks {
    sector: string
    confidence: number
    market_sentiment: 'Bullish' | 'Bearish' | 'Neutral'
    topics: { text: string; value: number }[]
    definition?: string
    subdomain?: string
    wiki_url?: string
}

interface SectorApiResponse {
    candidates: Record<string, unknown>[]
    final: {
        domain?: string
        definition?: string
        subdomain?: string
        wiki_url?: string
        confidence?: number
        market_sentiment?: 'Bullish' | 'Bearish' | 'Neutral'
        topics?: Array<{ text: string; value: number }>
    }
    follow_up_questions?: string[]
}

export async function getSectorBenchmarks(payload: {
    data_profile: Record<string, unknown>[]
    memory?: Record<string, unknown>
    extracted_text?: string | null
    user_hints?: string
}): Promise<SectorBenchmarks> {
    const { data } = await api.post<SectorApiResponse>('/sector', {
        data_profile: payload.data_profile,
        memory: payload.memory ?? {},
        extracted_text: payload.extracted_text ?? null,
        user_hints: payload.user_hints ?? '',
    })

    const confidenceRaw = typeof data.final?.confidence === 'number' ? data.final.confidence : 0
    const confidence = confidenceRaw > 1 ? confidenceRaw / 100 : confidenceRaw

    const topics = Array.isArray(data.final?.topics)
        ? data.final.topics
            .filter((item) => item && typeof item.text === 'string' && typeof item.value === 'number')
            .map((item) => ({ text: item.text, value: item.value }))
        : []

    return {
        sector: data.final?.domain || 'Unknown',
        confidence,
        market_sentiment: data.final.market_sentiment ?? 'Neutral',
        topics,
        definition: data.final?.definition,
        subdomain: data.final?.subdomain,
        wiki_url: data.final?.wiki_url,
    }
}

// ── KPI Dashboard (Phase 4.3) ──────────────────────────

export interface KPIConcept {
    concept_phrase?: string
    why_it_matters?: string
    business_relevance?: string
    calculation_steps?: string[] | string
    calculation_explainer?: string
    calculated_value?: unknown
    assumptions?: string[] | string
    data_mapping?: {
        measure?: string[]
        group_by?: string[]
        calculation_method?: string
    }
}

interface MetricsApiResponse {
    concepts: KPIConcept[]
}

export async function getKPIs(payload: {
    domain_info: Record<string, unknown>
    data_profile: Record<string, unknown>[]
    memory?: Record<string, unknown>
    user_metrics: string[]
    key_metrics?: string
    df_records?: Record<string, unknown>[] | null
    df_columns?: string[] | null
    pdf_dfs_records?: Record<string, Record<string, unknown>[]> | null
    extracted_text?: string | null
}): Promise<KPIConcept[]> {
    const requestBody: Record<string, unknown> = {
        domain_info: payload.domain_info,
        data_profile: payload.data_profile,
        memory: payload.memory ?? {},
        user_metrics: payload.user_metrics,
        key_metrics: payload.key_metrics ?? '',
    }

    if (payload.pdf_dfs_records && Object.keys(payload.pdf_dfs_records).length > 0) {
        requestBody.pdf_dfs_records = payload.pdf_dfs_records
    } else if (payload.df_records && payload.df_columns) {
        requestBody.df_records = payload.df_records
        requestBody.df_columns = payload.df_columns
    }

    if (payload.extracted_text) {
        requestBody.extracted_text = payload.extracted_text
    }

    const { data } = await api.post<MetricsApiResponse>('/metrics', requestBody)
    return data.concepts ?? []
}

// ── AI Insights (Phase 4.4) ────────────────────────────

export type InsightCategoryKey =
    | 'descriptive'
    | 'predictive'
    | 'domain_related'
    | 'novel_patterns'
    | 'quality_implications'
    | 'recommended_actions'
    | 'open_questions'

export type InsightsByCategory = Partial<Record<InsightCategoryKey, unknown[]>>

interface InsightsApiResponse {
    analysis: Record<string, unknown>
}

export async function generateInsights(payload: {
    data_profile: Record<string, unknown>[]
    domain_info: Record<string, unknown>
    extracted_concepts: KPIConcept[]
    memory?: Record<string, unknown>
    extracted_text?: string | null
    user_hints?: string
    key_metrics?: string
}): Promise<InsightsByCategory> {
    const { data } = await api.post<InsightsApiResponse>('/insights', {
        data_profile: payload.data_profile,
        domain_info: payload.domain_info,
        extracted_concepts: payload.extracted_concepts,
        memory: payload.memory ?? {},
        extracted_text: payload.extracted_text ?? null,
        user_hints: payload.user_hints ?? '',
        key_metrics: payload.key_metrics ?? '',
    })

    const top = data.analysis ?? {}
    const categoriesCandidate =
        typeof top.analysis === 'object' && top.analysis !== null
            ? (top.analysis as Record<string, unknown>)
            : (top as Record<string, unknown>)

    const out: InsightsByCategory = {}
    const keys: InsightCategoryKey[] = [
        'descriptive',
        'predictive',
        'domain_related',
        'novel_patterns',
        'quality_implications',
        'recommended_actions',
        'open_questions',
    ]

    keys.forEach((key) => {
        const value = categoriesCandidate[key]
        if (Array.isArray(value) && value.length > 0) {
            out[key] = value
        }
    })

    return out
}

// ── Visualizations (Phase 4.5) ─────────────────────────
// This is the TYPED CONTRACT between frontend and backend.
// When FastAPI is ready, only the function body changes — not the interface.

export interface ChartData {
    trends: TrendPoint[]
    regional: RegionalPoint[]
    breakdown: BreakdownPoint[]
    quarterly: QuarterlyPoint[]
    summary: ChartSummary
}

export interface TrendPoint {
    month: string
    Revenue: number
    Expenses: number
    Profit: number
}

export interface RegionalPoint {
    region: string
    Sales: number
    Target: number
}

export interface BreakdownPoint {
    name: string
    value: number
}

export interface QuarterlyPoint {
    quarter: string
    Revenue: number
    Profit: number
    Margin: number
}

export interface ChartSummary {
    totalRevenue: number
    avgMargin: number
    topRegion: string
    yoyGrowth: number
}


/**
 * Fetch visualization data for a given dataset.
 *
 * MOCK: Returns rich 12-month demo data after a short delay.
 * REAL: Replace body with `return api.get('/datasets/${datasetId}/charts').then(res => res.data)`
 */
export async function getVisualizationData(datasetId: string): Promise<ChartData | null> {
    try {
        const { data } = await api.get<ChartData>(`/datasets/${datasetId}/charts`)
        return data
    } catch {
        // Endpoint not yet implemented on backend — return null so callers can handle gracefully
        return null
    }
}


// ── Agent 5 Visual Insights (Dynamic) ─────────────────

export type RechartsType =
    | 'line'
    | 'bar'
    | 'area'
    | 'pie'
    | 'scatter'
    | 'radar'
    | 'radialBar'
    | 'treemap'
    | 'funnel'
    | 'composed'

export interface VisualInsight {
    chart_id?: string
    chart_title?: string
    insight_text: string
    derived_signal?: string
    key_driver?: string
    why_this_chart?: string
    chart_type?: string
    recharts_type?: RechartsType
    chart_data?: Record<string, unknown>[]
    x_key?: string
    series?: Array<{ key: string; label?: string }>
    y_keys?: string[]
    assumptions?: string[]
    image_b64?: string | null
    error?: string | null
}

interface VisualsApiResponse {
    visuals: VisualInsight[]
}

export async function getVisualInsights(payload: {
    extracted_text?: string | null
    insights_json: Record<string, unknown>
    domain_info?: Record<string, unknown>
    kpis?: KPIConcept[]
    df_records?: Record<string, unknown>[] | null
    df_columns?: string[] | null
    dfs_records?: Record<string, Record<string, unknown>[]> | null
}): Promise<VisualInsight[]> {
    try {
        const { data } = await api.post<VisualsApiResponse>('/visuals', {
            extracted_text: payload.extracted_text ?? null,
            insights_json: payload.insights_json,
            domain_info: payload.domain_info ?? {},
            kpis: payload.kpis ?? [],
            df_records: payload.df_records ?? null,
            df_columns: payload.df_columns ?? null,
            dfs_records: payload.dfs_records ?? null,
        })

        return data.visuals ?? []
    } catch (error: any) {
        const backendDetail = error?.response?.data?.detail
        if (typeof backendDetail === 'string' && backendDetail.trim().length > 0) {
            throw new Error(backendDetail)
        }
        throw error
    }
}

// ── Export (Phase 5) ───────────────────────────────────
// Typed contract for report exports. Backend swap = change function body only.

export type ExportFormat = 'pdf' | 'pptx'

export interface ExportResult {
    downloadUrl: string
    fileName: string
    format: ExportFormat
    generatedAt: string
}

/**
 * Request a report export for a dataset.
 *
 * MOCK: Simulates a 2s generation and returns a fake blob URL.
 * REAL: Replace body with `api.get(...)` that returns { downloadUrl, fileName, ... }
 */
export async function exportReport(datasetId: string, format: ExportFormat): Promise<ExportResult> {
    // Legacy signature kept for compatibility in call sites that haven't been migrated.
    console.log(`Exporting ${format} report for dataset: ${datasetId}`)
    const timestamp = new Date().toISOString()
    return {
        downloadUrl: `#legacy-${format}-${Date.now()}`,
        fileName: `visual_insights.${format}`,
        format,
        generatedAt: timestamp,
    }
}

export async function exportVisualReport(payload: {
    format: ExportFormat
    extracted_text?: string | null
    insights_json: Record<string, unknown>
    domain_info?: Record<string, unknown>
    kpis?: KPIConcept[]
    df_records?: Record<string, unknown>[] | null
    df_columns?: string[] | null
    dfs_records?: Record<string, Record<string, unknown>[]> | null
    pre_rendered_visuals: VisualInsight[]
}): Promise<ExportResult> {
    const endpoint = payload.format === 'pdf' ? '/export/pdf' : '/export/pptx'

    const response = await api.post(endpoint, {
        extracted_text: payload.extracted_text ?? null,
        insights_json: payload.insights_json,
        domain_info: payload.domain_info ?? {},
        kpis: payload.kpis ?? [],
        df_records: payload.df_records ?? null,
        df_columns: payload.df_columns ?? null,
        dfs_records: payload.dfs_records ?? null,
        pre_rendered_visuals: payload.pre_rendered_visuals,
    }, {
        responseType: 'blob',
    })

    const blob = new Blob([response.data], {
        type: payload.format === 'pdf'
            ? 'application/pdf'
            : 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    })
    const url = URL.createObjectURL(blob)

    return {
        downloadUrl: url,
        fileName: `visual_insights.${payload.format}`,
        format: payload.format,
        generatedAt: new Date().toISOString(),
    }
}
