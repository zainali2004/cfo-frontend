import api from './api'

// ── Types ──────────────────────────────────────────────

export interface ChatMessage {
    role: 'user' | 'assistant'
    content: string
}

export interface ChatAttachment {
    filename: string
    file_type: string
    description?: string
    data_base64?: string | null
    download_url?: string | null
    size_bytes?: number | null
}

export interface ChatContext {
    profiles: Record<string, unknown>[]
    domain_info: Record<string, unknown>
    concepts: Record<string, unknown>[]
    insights: Record<string, unknown>
    raw_preview?: string | null
    extracted_text?: string | null
    df_records?: Record<string, unknown>[] | null
    df_columns?: string[] | null
    pdf_table_records?: Record<string, Record<string, unknown>[]> | null
}

export interface ChatRequest {
    message: string
    history: ChatMessage[]
    context: ChatContext
}

export interface ChatResponse {
    reply: string
    attachments?: ChatAttachment[]
    web_search_used?: boolean
}

// ── API Functions ──────────────────────────────────────

/**
 * Send a message to the AI chatbot and get a response.
 * The backend is stateless — the frontend maintains conversation history.
 */
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
        console.log('Sending chat message:', { message: request.message, contextKeys: Object.keys(request.context) })
        const response = await api.post<ChatResponse>('/chat', request)
        console.log('Chat response received:', response.data)
        return response.data
    } catch (error: unknown) {
        const err = error as { response?: { status?: number; data?: { detail?: string } }; message?: string }
        console.error('Chat API error:', {
            status: err.response?.status,
            detail: err.response?.data?.detail,
            message: err.message,
        })
        throw error
    }
}
