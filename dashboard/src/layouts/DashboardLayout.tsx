import { useState, useEffect, useRef } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
    Upload,
    Building2,
    LineChart,
    Download,
    MessageCircle,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import { Navbar } from '@/components/Navbar'
import { useDatasetStore } from '@/stores/useDatasetStore'

// ── Tab Configuration ─────────────────────────────────
interface TabItem {
    label: string
    path: string
    icon: LucideIcon
    alwaysActive: boolean
}

const TABS: TabItem[] = [
    { label: 'Upload', path: '/dashboard', icon: Upload, alwaysActive: true },
    // Data Profiling runs silently after upload, no tab shown
    { label: 'Sector & KPI', path: '/dashboard/sector', icon: Building2, alwaysActive: false },
    { label: 'Visual', path: '/dashboard/visual', icon: LineChart, alwaysActive: false },
    { label: 'Export', path: '/dashboard/export', icon: Download, alwaysActive: false },
    { label: 'Chat', path: '/dashboard/chat', icon: MessageCircle, alwaysActive: false },
]

// ── Layout ────────────────────────────────────────────

/**
 * DashboardLayout — The dashboard frame.
 *
 * Structure:
 *   1. Dark navbar (Navbar component)
 *   2. Underline tab bar (with icons + text)
 *   3. Padded content area (<Outlet />)
 *
 * Tab logic:
 *   - "Upload" is always active
 *   - All other tabs disabled until data is uploaded (useDatasetStore.data !== null)
 */
export function DashboardLayout() {
    const data = useDatasetStore((s) => s.data)
    const agent1Done = useDatasetStore((s) => s.agent1Done)
    const agent2Done = useDatasetStore((s) => s.agent2Done)
    const agent3Done = useDatasetStore((s) => s.agent3Done)
    const agent5Done = useDatasetStore((s) => s.agent5Done)
    const hasData = data !== null
    const location = useLocation()

    // Animation State
    const [justUnlocked, setJustUnlocked] = useState(false)
    const prevHasData = useRef(hasData)
    const navigate = useNavigate()

    useEffect(() => {
        // Trigger animation when hasData flips from false -> true
        if (!prevHasData.current && hasData) {
            setJustUnlocked(true)
            const timer = setTimeout(() => setJustUnlocked(false), 2000)
            return () => clearTimeout(timer)
        }
        prevHasData.current = hasData
    }, [hasData])

    // Auto-Redirect: If no data, force user back to upload page
    // (Prevents accessing /dashboard/profile directly)
    useEffect(() => {
        if (!hasData && location.pathname !== '/dashboard') {
            navigate('/dashboard', { replace: true })
            toast.error("Please upload a dataset first", { id: 'redirect-error' })
        }
    }, [hasData, location.pathname, navigate])

    useEffect(() => {
        if (!hasData) {
            return
        }

        if (location.pathname.startsWith('/dashboard/sector') && !agent1Done) {
            navigate('/dashboard/profile', { replace: true })
            toast.error('Run Data Profiling first', { id: 'redirect-sector' })
            return
        }
        if (location.pathname.startsWith('/dashboard/visual') && !agent3Done) {
            navigate('/dashboard/sector', { replace: true })
            toast.error('Run Sector & KPI first', { id: 'redirect-visual' })
            return
        }
        if (location.pathname.startsWith('/dashboard/export') && !agent5Done) {
            navigate('/dashboard/visual', { replace: true })
            toast.error('Generate Visual Insights first', { id: 'redirect-export' })
            return
        }
        if (location.pathname.startsWith('/dashboard/chat') && !agent5Done) {
            navigate('/dashboard/visual', { replace: true })
            toast.error('Generate Visual Insights first', { id: 'redirect-chat' })
        }
    }, [hasData, location.pathname, agent1Done, agent2Done, agent3Done, agent5Done, navigate])

    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-slate-950 transition-colors">
            {/* 1. Navbar */}
            <Navbar />

            {/* 2. Tab Bar */}
            <div className="bg-white dark:bg-gray-950 border-b border-gray-200 dark:border-gray-700 transition-colors">
                <div className="max-w-7xl mx-auto px-6">
                    <nav className="flex gap-1 overflow-x-auto" role="tablist">
                        {TABS.map((tab) => {
                            const Icon = tab.icon
                            const isDisabled = (() => {
                                if (tab.alwaysActive) return false
                                if (tab.path === '/dashboard/profile') return !hasData
                                if (tab.path === '/dashboard/sector') return !agent1Done
                                if (tab.path === '/dashboard/visual') return !agent3Done
                                if (tab.path === '/dashboard/export') return !agent5Done
                                if (tab.path === '/dashboard/chat') return !agent5Done
                                return !hasData
                            })()

                            // For Upload tab, match exact path only
                            const isActive = tab.path === '/dashboard'
                                ? location.pathname === '/dashboard'
                                : location.pathname.startsWith(tab.path)

                            // Animation: Apply to non-active unlocked tabs
                            const isAnimating = justUnlocked && !tab.alwaysActive && !isActive

                            if (isDisabled) {
                                return (
                                    <div
                                        key={tab.path}
                                        className="flex items-center gap-2 px-4 py-3 text-sm text-gray-500 dark:text-gray-600 cursor-not-allowed border-b-2 border-transparent select-none transition-colors"
                                        title="Upload data first"
                                        role="tab"
                                        aria-disabled="true"
                                    >
                                        <Icon className="h-4 w-4" />
                                        <span>{tab.label}</span>
                                    </div>
                                )
                            }

                            return (
                                <NavLink
                                    key={tab.path}
                                    to={tab.path}
                                    end={tab.path === '/dashboard'}
                                    role="tab"
                                    className={() =>
                                        `flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-all duration-700 whitespace-nowrap ${isActive
                                            ? 'border-deloitte text-deloitte'
                                            : `border-transparent hover:text-gray-700 dark:hover:text-gray-200 hover:border-gray-300 dark:hover:border-gray-500 ${isAnimating
                                                ? 'text-deloitte scale-105 font-bold'
                                                : 'text-gray-500 dark:text-gray-400'
                                            }`
                                        }`
                                    }
                                >
                                    <Icon className={`h-4 w-4 transition-transform duration-700 ${isAnimating ? 'scale-125' : ''}`} />
                                    <span>{tab.label}</span>
                                </NavLink>
                            )
                        })}
                    </nav>
                </div>
            </div>

            {/* 3. Content Area */}
            <main className="flex-1">
                <div className="max-w-7xl mx-auto px-6 py-8">
                    <Outlet />
                </div>
            </main>
        </div>
    )
}
