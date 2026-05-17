import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/ui/accordion"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { useDatasetStore } from "@/stores/useDatasetStore"

export function DetectedTablesPreview() {
    const tables = useDatasetStore((s) => s.tables)

    if (!tables || tables.length === 0) {
        return (
            <div className="text-center py-8 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
                <p className="text-sm text-gray-400 dark:text-gray-500">No tables detected yet.</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Status Bar (Green) */}
            <div className="rounded-md bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800/30 p-4">
                <div className="flex items-center gap-3">
                    <div className="h-5 w-5 rounded-full bg-green-500 text-white flex items-center justify-center text-xs">
                        ✓
                    </div>
                    <div>
                        <p className="text-sm font-medium text-green-800 dark:text-green-300">
                            Preview of {tables.length} detected table{tables.length !== 1 ? 's' : ''} from source.
                        </p>
                    </div>
                </div>
            </div>

            {/* Accordion List */}
            <Accordion type="multiple" defaultValue={['item-0']} className="w-full">
                {tables.map((table, index) => (
                    <AccordionItem key={index} value={`item-${index}`} className="border rounded-lg bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 mb-4 px-4">
                        <AccordionTrigger className="hover:no-underline hover:text-deloitte dark:hover:text-deloitte transition-colors">
                            <span className="font-medium text-gray-700 dark:text-gray-200">
                                {table.name}
                            </span>
                        </AccordionTrigger>
                        <AccordionContent>
                            <div className="rounded-md border border-gray-200 dark:border-gray-700 overflow-hidden">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            {table.columns.map((col, i) => (
                                                <TableHead key={i} className="bg-gray-50 dark:bg-gray-900/50 text-xs font-semibold uppercase tracking-wider h-10 text-gray-700 dark:text-gray-200 whitespace-nowrap">
                                                    {col}
                                                </TableHead>
                                            ))}
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {/* Dynamic: Render rows based on store data */}
                                        {table.rows.slice(0, 5).map((row, rS) => (
                                            <TableRow key={rS}>
                                                {table.columns.map((col, cS) => (
                                                    <TableCell key={cS} className="py-2.5 text-gray-600 dark:text-gray-300 whitespace-nowrap">
                                                        {String(row[col] ?? '-')}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        ))}
                                        {table.rows.length > 5 && (
                                            <TableRow>
                                                <TableCell colSpan={table.columns.length} className="bg-gray-50 dark:bg-gray-900/30 text-center text-xs text-gray-400 py-2">
                                                    ... {table.rows.length - 5} more rows ...
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </TableBody>
                                </Table>
                            </div>
                        </AccordionContent>
                    </AccordionItem>
                ))}
            </Accordion>
        </div>
    )
}
