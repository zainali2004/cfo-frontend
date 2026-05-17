import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import { useDatasetStore } from "@/stores/useDatasetStore"
import { Database, FileSpreadsheet, AlertTriangle, Layers, Table } from "lucide-react"

export function DataMetadataCard() {
    const rowCount = useDatasetStore((s) => s.rowCount)
    const columnCount = useDatasetStore((s) => s.columnCount)
    const missingValues = useDatasetStore((s) => s.missingValues)
    const columns = useDatasetStore((s) => s.columns)
    const tables = useDatasetStore((s) => s.tables)

    const stats = [
        {
            title: "Tables Detected",
            value: (tables?.length || 0).toLocaleString(),
            icon: Table,
            color: "text-indigo-500",
            bg: "bg-indigo-50 dark:bg-indigo-900/20",
        },
        {
            title: "Total Rows",
            value: rowCount.toLocaleString(),
            icon: Database,
            color: "text-blue-500",
            bg: "bg-blue-50 dark:bg-blue-900/20",
        },
        {
            title: "Total Columns",
            value: columnCount.toLocaleString(),
            icon: FileSpreadsheet,
            color: "text-purple-500",
            bg: "bg-purple-50 dark:bg-purple-900/20",
        },
        {
            title: "Missing Values",
            value: missingValues.toLocaleString(),
            icon: AlertTriangle,
            color: missingValues > 0 ? "text-amber-500" : "text-green-500",
            bg: missingValues > 0 ? "bg-amber-50 dark:bg-amber-900/20" : "bg-green-50 dark:bg-green-900/20",
        },
    ]

    return (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {/* KPI Cards */}
            {stats.map((stat, index) => (
                <Card key={index} className="border-gray-200 dark:border-gray-700 shadow-sm">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {stat.title}
                        </CardTitle>
                        <div className={`p-2 rounded-full ${stat.bg}`}>
                            <stat.icon className={`h-4 w-4 ${stat.color}`} />
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-gray-800 dark:text-white">{stat.value}</div>
                    </CardContent>
                </Card>
            ))}

            {/* Extracted Fields (Spans full width or separate section) */}
            <Card className="col-span-full border-gray-200 dark:border-gray-700 shadow-sm">
                <CardHeader className="flex flex-row items-center space-y-0 pb-2 gap-2">
                    <div className="p-2 rounded-full bg-gray-100 dark:bg-gray-800">
                        <Layers className="h-4 w-4 text-gray-600 dark:text-gray-300" />
                    </div>
                    <div>
                        <CardTitle className="text-base font-semibold text-gray-800 dark:text-white">
                            Key Extracted Fields
                        </CardTitle>
                        <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                            {columns.length} fields detected in the source file
                        </p>
                    </div>
                </CardHeader>
                <CardContent className="pt-4">
                    <div className="flex flex-wrap gap-2">
                        {columns.length > 0 ? (
                            columns.map((col, i) => (
                                <span
                                    key={i}
                                    className="inline-flex items-center rounded-full bg-deloitte/10 dark:bg-deloitte/20 px-2.5 py-1 text-xs font-medium text-deloitte dark:text-deloitte-light border border-deloitte/20"
                                >
                                    {col}
                                </span>
                            ))
                        ) : (
                            <span className="text-sm text-gray-400 italic">No fields detected</span>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
