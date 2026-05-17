import { ArrowUpRight, ArrowDownRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface KPIData {
    name: string
    value: string
    change: number
    isCustom?: boolean
}

interface KPIGridProps {
    kpis: KPIData[]
    selectedKpiName: string
    onSelectKpi: (name: string) => void
}

export function KPIGrid({ kpis, selectedKpiName, onSelectKpi }: KPIGridProps) {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {kpis.map((kpi) => {
                const isPositive = kpi.change >= 0
                const isSelected = selectedKpiName === kpi.name

                return (
                    <button
                        key={kpi.name}
                        onClick={() => onSelectKpi(kpi.name)}
                        className={cn(
                            "relative flex flex-col items-start p-5 rounded-xl border transition-all text-left group",
                            "bg-white dark:bg-gray-800 hover:shadow-md",
                            isSelected
                                ? "border-deloitte ring-1 ring-deloitte dark:border-deloitte"
                                : "border-gray-200 dark:border-gray-700 hover:border-deloitte/50"
                        )}
                    >
                        <span className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                            {kpi.name}
                            {kpi.isCustom && (
                                <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                                    Custom
                                </span>
                            )}
                        </span>

                        <div className="flex items-baseline gap-2 mt-1">
                            <span className="text-2xl font-bold text-gray-900 dark:text-white">
                                {kpi.value}
                            </span>
                        </div>

                        <div className={cn(
                            "flex items-center gap-1 mt-3 text-xs font-medium px-2 py-1 rounded-full",
                            isPositive
                                ? "text-green-700 bg-green-50 dark:bg-green-900/20 dark:text-green-400"
                                : "text-red-700 bg-red-50 dark:bg-red-900/20 dark:text-red-400"
                        )}>
                            {isPositive ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                            {Math.abs(kpi.change)}% vs last month
                        </div>
                    </button>
                )
            })}
        </div>
    )
}
