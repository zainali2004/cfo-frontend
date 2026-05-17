import { useEffect, useState } from 'react'
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
    )
}

export default App
