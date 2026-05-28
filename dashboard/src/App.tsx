import { useEffect, useState, Component, type ReactNode } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'

import { DashboardLayout } from '@/layouts/DashboardLayout'
import { useAuthStore } from '@/stores/useAuthStore'
import {
    UploadPage,
    DataProfilePage,
    SectorPage,
    VisualPage,
    ExportPage,
    ChatPage,
} from '@/pages/DashboardPages'

// ── Error Boundary ─────────────────────────────────────
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; error: Error | null }> {
    constructor(props: { children: ReactNode }) {
        super(props)
        this.state = { hasError: false, error: null }
    }

    static getDerivedStateFromError(error: Error) {
        return { hasError: true, error }
    }

    componentDidCatch(error: Error, info: { componentStack: string }) {
        console.error('React Error Boundary caught an error:', error, info)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-slate-950 text-white p-8">
                    <div className="max-w-md text-center space-y-4">
                        <div className="text-5xl">⚠️</div>
                        <h2 className="text-2xl font-semibold text-red-400">Something went wrong</h2>
                        <p className="text-gray-400 text-sm">
                            {this.state.error?.message || 'An unexpected error occurred.'}
                        </p>
                        <button
                            onClick={() => window.location.reload()}
                            className="mt-4 px-6 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
                        >
                            Reload Page
                        </button>
                    </div>
                </div>
            )
        }
        return this.props.children
    }
}

// ── App ────────────────────────────────────────────────
function App() {
    const [isHydrating, setIsHydrating] = useState(true)
    const hydrate = useAuthStore((s) => s.hydrate)
// Restore session from storage on app startup
    useEffect(() => {
        hydrate().finally(() => setIsHydrating(false))
    }, [hydrate])

    // Show nothing until hydration completes — prevents PrivateRoute
    // from redirecting to /login before the session is restored
    if (isHydrating) {
        return null
    }

    return (
        <ErrorBoundary>
            <BrowserRouter>
                {/* Sonner toast container — persists across route changes */}
                <Toaster richColors position="top-right" />

                <Routes>
                    {/* Dashboard routes */}
                    <Route element={<DashboardLayout />}>
                        <Route path="/dashboard" element={<UploadPage />} />
                        <Route path="/dashboard/profile" element={<DataProfilePage />} />
                        <Route path="/dashboard/sector" element={<SectorPage />} />
                        <Route path="/dashboard/visual" element={<VisualPage />} />
                        <Route path="/dashboard/export" element={<ExportPage />} />
                        <Route path="/dashboard/chat" element={<ChatPage />} />
                    </Route>

                    {/* Catch-all */}
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
            </BrowserRouter>
        </ErrorBoundary>
    )
}

export default App
