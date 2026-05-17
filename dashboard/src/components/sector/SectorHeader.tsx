import { Building2, TrendingUp, TrendingDown, Minus, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SectorHeaderProps {
    sector: string
    confidence: number
    sentiment: 'Bullish' | 'Bearish' | 'Neutral'
    wikiUrl?: string
}

export function SectorHeader({ sector, confidence, sentiment, wikiUrl }: SectorHeaderProps) {
    const sentimentColor = {
        'Bullish': 'text-green-600 bg-green-50 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800',
        'Bearish': 'text-red-600 bg-red-50 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800',
        'Neutral': 'text-gray-600 bg-gray-50 border-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-700',
    }[sentiment]

    const SentimentIcon = {
        'Bullish': TrendingUp,
        'Bearish': TrendingDown,
        'Neutral': Minus,
    }[sentiment]

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-all">
            <div className="flex items-start gap-4">
                <div className="p-3 bg-deloitte/10 dark:bg-deloitte/20 rounded-lg">
                    <Building2 className="h-8 w-8 text-deloitte" />
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{sector}</h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400">AI-Detected Sector Classification</p>
                </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-6">
                {/* Confidence Score */}
                <div className="space-y-2 min-w-[140px]">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-500 dark:text-gray-400">Confidence</span>
                        <span className="font-medium text-gray-900 dark:text-white">{(confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 w-full bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-deloitte transition-all duration-1000 ease-out"
                            style={{ width: `${confidence * 100}%` }}
                        />
                    </div>
                </div>

                {/* Market Sentiment Badge */}
                <div className="flex items-center">
                    <div className={cn("flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-medium", sentimentColor)}>
                        <SentimentIcon className="h-4 w-4" />
                        <span>{sentiment} Market</span>
                    </div>
                </div>

                {wikiUrl && (
                    <a
                        href={wikiUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2 rounded-full border border-deloitte/30 px-4 py-2 text-sm font-medium text-deloitte hover:bg-deloitte/10 transition-colors"
                    >
                        <ExternalLink className="h-4 w-4" />
                        Wikipedia
                    </a>
                )}
            </div>
        </div>
    )
}
