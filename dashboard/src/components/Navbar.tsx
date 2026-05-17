import { useState, useRef, useEffect } from 'react'

import { ChevronDown, Mail } from 'lucide-react'

import { useAuthStore } from '@/stores/useAuthStore'
import { ThemeToggle } from '@/components/ThemeToggle'

/**
 * Get user initials from display name.
 * "Admin User" → "AU", "John" → "JO"
 */
function getInitials(name: string): string {
    const parts = name.trim().split(/\s+/)
    if (parts.length >= 2) {
        return (parts[0][0] + parts[1][0]).toUpperCase()
    }
    return name.slice(0, 2).toUpperCase()
}

/**
 * Navbar — Dark charcoal top bar.
 *
 * Left:  Deloitte logo + "Dashboard" text
 * Right: Theme toggle + clickable avatar → dropdown with email, logout
 */
export function Navbar() {
    const user = useAuthStore((s) => s.user)

    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef<HTMLDivElement>(null)

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(e: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setIsOpen(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    return (
        <nav className="bg-gray-800 dark:bg-gray-950 text-white px-6 py-3 flex items-center justify-between transition-colors">
            {/* Left — Logo + Title */}
            <div className="flex items-center gap-2">
                <img
                    src="/deloitte_white.png"
                    alt="Deloitte"
                    className="h-5"
                />
                <span className="text-base font-medium text-gray-300 translate-y-0.5 -ml-0.5">
                    Dashboard
                </span>
            </div>

            {/* Right — Theme toggle + Avatar dropdown */}
            <div className="flex items-center gap-2">
                <ThemeToggle variant="dark" />

                <div className="relative" ref={dropdownRef}>
                    <button
                        onClick={() => setIsOpen(!isOpen)}
                        className="flex items-center gap-2 hover:bg-gray-700 rounded-lg px-2 py-1.5 transition-colors cursor-pointer"
                    >
                        {/* Avatar circle */}
                        <div className="w-8 h-8 rounded-full bg-deloitte flex items-center justify-center text-sm font-semibold text-white">
                            {user ? getInitials(user.name) : '??'}
                        </div>
                        <span className="text-sm text-gray-300 hidden sm:inline">
                            {user?.name}
                        </span>
                        <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Dropdown menu */}
                    {isOpen && (
                        <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-1 z-50">
                            {/* User info header */}
                            <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700">
                                <p className="text-sm font-semibold text-gray-800 dark:text-gray-100">
                                    {user?.name}
                                </p>
                                <div className="flex items-center gap-1.5 mt-1">
                                    <Mail className="h-3.5 w-3.5 text-gray-400" />
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        {user?.email}
                                    </p>
                                </div>
                            </div>

                        </div>
                    )}
                </div>
            </div>
        </nav>
    )
}
