import { cn } from '@/lib/utils'

interface Topic {
    text: string
    value: number // 0-100 relevance
}

interface TopicCloudProps {
    topics: Topic[]
}

export function TopicCloud({ topics }: TopicCloudProps) {
    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Trending Topics</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                Key themes detected in your sector this quarter based on trusted market reports.
            </p>

            {topics.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400">No trending topics were returned.</p>
            ) : (
                <div className="flex flex-wrap gap-3 content-start">
                    {topics.map((topic, i) => {
                    // Uniform size for all badges
                    const sizeClass = 'px-3 py-1 text-xs'
                    const opacity = Math.max(0.1, topic.value / 100)

                    return (
                        <div
                            key={i}
                            className={cn(
                                "inline-flex items-center rounded-full font-medium transition-all hover:scale-105 cursor-default border",
                                "bg-deloitte/5 border-deloitte/20 text-deloitte-dark dark:text-deloitte-light",
                                sizeClass
                            )}
                            style={{
                                backgroundColor: `rgba(134, 188, 37, ${opacity * 0.2})`, // Deloitte Green with opacity
                                borderColor: `rgba(134, 188, 37, ${opacity * 0.5})`
                            }}
                        >
                            {topic.text}
                            <span className="ml-2 text-xs opacity-60">{topic.value}%</span>
                        </div>
                    )
                    })}
                </div>
            )}
        </div>
    )
}
