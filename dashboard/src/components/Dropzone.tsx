import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, File as FileIcon, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DropzoneProps {
    onFileSelect: (file: File | null) => void
    selectedFile: File | null
    disabled?: boolean
}

export function Dropzone({ onFileSelect, selectedFile, disabled }: DropzoneProps) {
    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles?.length > 0) {
                onFileSelect(acceptedFiles[0])
            }
        },
        [onFileSelect]
    )

    const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
        onDrop,
        maxFiles: 1,
        // Max 200MB
        maxSize: 200 * 1024 * 1024,
        disabled,
        accept: {
            'text/csv': ['.csv'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
            'application/vnd.ms-excel': ['.xls'],
            'application/pdf': ['.pdf'],
            'application/vnd.ms-outlook': ['.msg'],
            'text/plain': ['.txt'],
        },
    })

    return (
        <div className="w-full">
            {!selectedFile ? (
                <div
                    {...getRootProps()}
                    className={cn(
                        'group relative flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-10 text-center transition-all duration-200 cursor-pointer',
                        // Default State
                        'border-gray-300 bg-white hover:border-deloitte hover:bg-gray-50',
                        // Dark Mode
                        'dark:border-gray-700 dark:bg-gray-800 dark:hover:bg-gray-700/50',
                        // Drag Active
                        isDragActive && 'border-deloitte bg-deloitte/5 scale-[0.99]',
                        // Drag Reject
                        isDragReject && 'border-red-500 bg-red-50',
                        // Disabled
                        disabled && 'opacity-60 cursor-not-allowed hover:border-gray-300 hover:bg-white'
                    )}
                >
                    <input {...getInputProps()} />

                    <div className={cn(
                        "flex h-16 w-16 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-700 transition-all duration-300 group-hover:scale-110 group-hover:bg-deloitte/10 group-hover:text-deloitte",
                        isDragActive && "bg-deloitte/10 text-deloitte"
                    )}>
                        <UploadCloud className={cn("h-8 w-8 text-gray-400 dark:text-gray-500 transition-colors group-hover:text-deloitte", isDragActive && "text-deloitte")} />
                    </div>

                    <div className="space-y-1">
                        <p className={cn("text-sm font-medium text-gray-700 dark:text-gray-200 transition-colors group-hover:text-deloitte", isDragActive && "text-deloitte")}>
                            {isDragActive ? "Drop the file here" : "Click to upload or drag and drop"}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                            CSV, XLSX, XLS, PDF, MSG, TXT (Max 200MB)
                        </p>
                    </div>
                </div>
            ) : (
                <div className="relative flex items-center gap-4 rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800 animate-in fade-in zoom-in-95 duration-200">
                    <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-deloitte/10">
                        <FileIcon className="h-6 w-6 text-deloitte" />
                    </div>

                    <div className="flex-1 min-w-0">
                        <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
                            {selectedFile.name}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                    </div>

                    <button
                        onClick={() => onFileSelect(null)}
                        disabled={disabled}
                        className="rounded-full p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-500 dark:hover:bg-gray-700 dark:hover:text-gray-300 transition-colors"
                        aria-label="Remove file"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>
            )}
        </div>
    )
}
