import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, AlertCircle, Download, File, FileText, Sheet, Image as ImageIcon, Globe } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { useDatasetStore } from '@/stores/useDatasetStore'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendChatMessage, type ChatMessage, type ChatContext, type ChatAttachment } from '@/services/chatService'

// ── Attachment rendering ────────────────────────────────────────
function getFileIcon(mimeType: string) {
    if (mimeType.includes('csv') || mimeType.includes('spreadsheet')) return <Sheet className="h-4 w-4" />
    if (mimeType.includes('pdf')) return <FileText className="h-4 w-4" />
    if (mimeType.includes('image')) return <ImageIcon className="h-4 w-4" />
    if (mimeType.includes('json') || mimeType.includes('text')) return <FileText className="h-4 w-4" />
    return <File className="h-4 w-4" />
}

function downloadFile(attachment: ChatAttachment) {
    if (attachment.download_url) {
        // Open URL in new tab
        window.open(attachment.download_url, '_blank')
        return
    }

    if (attachment.data_base64) {
        // Decode base64 and create blob
        const binaryString = window.atob(attachment.data_base64)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i)
        }
        const blob = new Blob([bytes], { type: attachment.file_type })

        // Create download link
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = attachment.filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
        return
    }

    toast.error('File data not available for download')
}

function AttachmentCard({ attachment }: { attachment: ChatAttachment }) {
    const sizeDisplay = attachment.size_bytes ? `${(attachment.size_bytes / 1024).toFixed(1)} KB` : ''

    return (
        <div className="mt-2 flex items-center justify-between rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 px-3 py-2">
            <div className="flex items-center gap-2 flex-1">
                {getFileIcon(attachment.file_type)}
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">
                        {attachment.filename}
                    </p>
                    {attachment.description && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                            {attachment.description}
                        </p>
                    )}
                    {sizeDisplay && (
                        <p className="text-xs text-gray-500 dark:text-gray-500">{sizeDisplay}</p>
                    )}
                </div>
            </div>
            <Button
                onClick={() => downloadFile(attachment)}
                size="sm"
                variant="ghost"
                className="ml-2 h-8 w-8 p-0"
                title="Download file"
            >
                <Download className="h-4 w-4" />
            </Button>
        </div>
    )
}

export function ChatPage() {
    const uploadPayload = useDatasetStore((s) => s.uploadPayload)
    const profiles = useDatasetStore((s) => s.profiles)
    const domainInfo = useDatasetStore((s) => s.domainInfo)
    const kpiConcepts = useDatasetStore((s) => s.kpiConcepts)
    const insights = useDatasetStore((s) => s.insights)
    const agent5Done = useDatasetStore((s) => s.agent5Done)
    const chatMessages = useDatasetStore((s) => s.chatMessages)
    const chatInputValue = useDatasetStore((s) => s.chatInputValue)
    const chatAttachments = useDatasetStore((s) => s.chatAttachments)
    const setChatMessages = useDatasetStore((s) => s.setChatMessages)
    const setChatInputValue = useDatasetStore((s) => s.setChatInputValue)
    const setChatAttachments = useDatasetStore((s) => s.setChatAttachments)
    const chatContextReady = Boolean(uploadPayload && profiles.length > 0 && domainInfo && kpiConcepts.length > 0 && insights && agent5Done)

    const [isLoading, setIsLoading] = useState(false)
    const [webSearchSourceMap, setWebSearchSourceMap] = useState<Record<number, 'web' | 'model_training'>>({})
    const messagesEndRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom when messages update
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [chatMessages])

    if (!chatContextReady) {
        return (
            <div className="py-12 max-w-2xl mx-auto">
                <div className="text-center mb-10">
                    <h2 className="text-2xl font-semibold text-gray-800 dark:text-white mb-2">AI Chatbot</h2>
                    <p className="text-gray-500 dark:text-gray-400">
                        Ask questions about your data and insights.
                    </p>
                </div>

                <div className="rounded-xl border border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-900/20 p-6">
                    <div className="flex gap-4">
                        <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                        <div>
                            <h3 className="font-semibold text-amber-900 dark:text-amber-200 mb-1">
                                Complete the full pipeline first
                            </h3>
                            <p className="text-sm text-amber-800 dark:text-amber-300">
                                The chatbot needs the Visual Insights pipeline to finish so it can use the stored sector, KPI, insight, and visual context.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    // Build chat context from store
    const buildChatContext = (): ChatContext => {
        return {
            profiles: (profiles ?? []) as Record<string, unknown>[],
            domain_info: (domainInfo ?? {}) as Record<string, unknown>,
            concepts: (kpiConcepts ?? []) as Record<string, unknown>[],
            insights: (insights ?? {}) as Record<string, unknown>,
            raw_preview: uploadPayload?.raw_preview ?? undefined,
            extracted_text: uploadPayload?.extracted_text ?? undefined,
            df_records: (uploadPayload?.data ?? undefined) as Record<string, unknown>[] | undefined,
            df_columns: (uploadPayload?.meta?.columns ?? undefined) as string[] | undefined,
            pdf_table_records: uploadPayload?.dfs_records ?? undefined,
        }
    }

    const handleSendMessage = async () => {
        if (!chatInputValue.trim()) {
            return
        }

        const userMessage: ChatMessage = {
            role: 'user',
            content: chatInputValue,
        }

        const newMessages = [...chatMessages, userMessage]
        setChatMessages(newMessages)
        setChatInputValue('')

        // Send to backend
        try {
            setIsLoading(true)
            const response = await sendChatMessage({
                message: chatInputValue,
                history: chatMessages,
                context: buildChatContext(),
            })

            if (!response?.reply) {
                console.warn('Chat response missing reply field:', response)
                toast.error('Received empty response from chatbot.')
                setChatMessages(chatMessages)
                return
            }

            // Add assistant response
            const assistantMessage: ChatMessage = {
                role: 'assistant',
                content: response.reply,
            }
            const updatedMessages = [...newMessages, assistantMessage]
            setChatMessages(updatedMessages)

            const newMsgIndex = updatedMessages.length - 1

            // Track the source of this message (web search or model training)
            if (response.web_search_source === 'web' || response.web_search_source === 'model_training') {
                setWebSearchSourceMap((prev) => ({ ...prev, [newMsgIndex]: response.web_search_source as 'web' | 'model_training' }))
            }

            // Store attachments for this message index
            if (response.attachments && response.attachments.length > 0) {
                setChatAttachments({
                    ...useDatasetStore.getState().chatAttachments,
                    [newMsgIndex]: response.attachments,
                })
            }
        } catch (error: unknown) {
            const err = error as { response?: { data?: { detail?: string } }; message?: string }
            const errorDetail = err.response?.data?.detail || err.message || 'Failed to get a response. Please try again.'
            console.error('Chat error:', error)
            console.error('Error detail:', errorDetail)
            toast.error(`Chat error: ${errorDetail}`)
            // Remove the user message if request failed
            setChatMessages(chatMessages)
        } finally {
            setIsLoading(false)
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSendMessage()
        }
    }

    return (
        <div className="py-8 h-full flex flex-col max-w-4xl mx-auto">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold text-gray-800 dark:text-white mb-2">AI Chatbot</h2>
                <p className="text-gray-500 dark:text-gray-400">
                    Ask questions about your data, insights, and metrics
                </p>
            </div>

            {/* Chat Messages Container */}
            <div className="flex-1 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto mb-6 p-6 space-y-4">
                {chatMessages.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-center">
                        <div className="space-y-3 text-gray-500 dark:text-gray-400">
                            <div className="text-4xl">💬</div>
                            <p className="font-medium">Start a conversation</p>
                            <p className="text-sm max-w-xs">
                                Ask questions about your data, KPIs, insights, or any analysis results
                            </p>
                        </div>
                    </div>
                ) : (
                    <>
                        {chatMessages.map((msg, idx) => (
                            <div key={idx}>
                                <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div
                                        className={`max-w-xs lg:max-w-2xl px-4 py-2 rounded-lg ${
                                            msg.role === 'user'
                                                ? 'bg-deloitte text-white rounded-br-none'
                                                : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-none'
                                        }`}
                                    >
                                        {msg.role === 'user' ? (
                                            <p className="text-sm whitespace-pre-wrap break-words">{msg.content}</p>
                                        ) : (
                                            <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                                                <ReactMarkdown
                                                    remarkPlugins={[remarkGfm]}
                                                    components={{
                                                        p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                                                        ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
                                                        ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
                                                        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                                                        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                                                        em: ({ children }) => <em className="italic">{children}</em>,
                                                        h1: ({ children }) => <h1 className="text-base font-bold mb-1 mt-2">{children}</h1>,
                                                        h2: ({ children }) => <h2 className="text-sm font-bold mb-1 mt-2">{children}</h2>,
                                                        h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-1">{children}</h3>,
                                                        a: ({ href, children }) => (
                                                            <a href={href} target="_blank" rel="noopener noreferrer" className="text-deloitte underline hover:opacity-80 break-all">
                                                                {children}
                                                            </a>
                                                        ),
                                                        code: ({ children }) => <code className="bg-gray-200 dark:bg-gray-600 rounded px-1 py-0.5 text-xs font-mono">{children}</code>,
                                                        hr: () => <hr className="my-2 border-gray-300 dark:border-gray-500" />,
                                                    }}
                                                >
                                                    {msg.content}
                                                </ReactMarkdown>
                                            </div>
                                        )}
                                    </div>
                                </div>
                                {/* Source attribution badge */}
                                {msg.role === 'assistant' && webSearchSourceMap[idx] === 'web' && (
                                    <div className="flex justify-start mt-1.5 pl-1">
                                        <div className="flex items-center gap-1.5 rounded-full bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 px-2.5 py-1 text-xs text-blue-600 dark:text-blue-400">
                                            <Globe className="h-3 w-3 flex-shrink-0" />
                                            <span>Via live web search · accuracy &amp; date may vary</span>
                                        </div>
                                    </div>
                                )}


                                {/* Show attachments for assistant messages */}
                                {msg.role === 'assistant' && chatAttachments[idx] && chatAttachments[idx].length > 0 && (
                                    <div className="flex justify-start mt-2 pl-2">
                                        <div className="space-y-2 w-full max-w-xs lg:max-w-md">
                                            {chatAttachments[idx].map((att, attIdx) => (
                                                <AttachmentCard key={attIdx} attachment={att} />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 px-4 py-2 rounded-lg rounded-bl-none">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </>
                )}
            </div>

            {/* Input Area */}
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
                <div className="flex gap-3">
                    <textarea
                        value={chatInputValue}
                        onChange={(e) => setChatInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question... (Shift+Enter for new line)"
                        disabled={isLoading}
                        className="flex-1 resize-none rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 px-4 py-3 text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-deloitte/40 focus:border-deloitte transition-colors disabled:opacity-50 [color-scheme:light] dark:[color-scheme:dark]"
                        rows={3}
                    />
                    <Button
                        onClick={handleSendMessage}
                        disabled={isLoading || !chatInputValue.trim()}
                        className="bg-deloitte hover:bg-deloitte-dark text-white self-end"
                    >
                        {isLoading ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                            <Send className="h-5 w-5" />
                        )}
                    </Button>
                </div>
            </div>
        </div>
    )
}
