import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts"
import { CHART_COLORS, CHART_THEME } from "@/config/chart-config"

interface BenchmarkChartProps {
    industryGrowth: number
    yourGrowth: number
    topPerformerGrowth: number
}

export function BenchmarkChart({ industryGrowth, yourGrowth, topPerformerGrowth }: BenchmarkChartProps) {
    const data = [
        {
            name: "Industry Avg",
            value: industryGrowth,
            fill: CHART_COLORS.tertiary, // Gray
        },
        {
            name: "Your Growth",
            value: yourGrowth,
            fill: CHART_COLORS.primary, // Deloitte Green
        },
        {
            name: "Top Performer",
            value: topPerformerGrowth,
            fill: CHART_COLORS.quaternary, // Blue
        },
    ]

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm transition-all h-[350px] w-full">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Growth Rate Comparison</h3>
            <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={CHART_THEME.grid.stroke} />
                        <XAxis
                            dataKey="name"
                            stroke="#888888"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#888888"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(value) => `${value}%`}
                        />
                        <Tooltip
                            cursor={{ fill: 'transparent' }}
                            contentStyle={CHART_THEME.tooltip.contentStyle}
                            labelStyle={{ color: '#111827', fontWeight: 600 }}
                        />
                        <Bar
                            dataKey="value"
                            radius={[4, 4, 0, 0]}
                            maxBarSize={60}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}
