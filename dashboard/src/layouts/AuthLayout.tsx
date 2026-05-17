import { Outlet } from 'react-router-dom'
import { ThemeToggle } from '@/components/ThemeToggle'

/**
 * AuthLayout — Centered card wrapper for auth pages.
 *
 * Used for: Login, Forgot Password (future)
 * Light/dark background, vertically + horizontally centered content.
 * Theme toggle in top-right corner.
 */
export function AuthLayout() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4 transition-colors">
            {/* Theme toggle — top right */}
            <div className="absolute top-4 right-4">
                <ThemeToggle />
            </div>

            <div className="w-full max-w-md">
                <Outlet />
            </div>
        </div>
    )
}
