import { Sun, Moon } from 'lucide-react'
import { useThemeStore } from '@/stores/useThemeStore'

/**
 * ThemeToggle — Sun/Moon icon button to switch between light and dark mode.
 * Works on both login page (light bg) and navbar (dark bg) via `variant` prop.
 */
export function ThemeToggle({ variant = 'light' }: { variant?: 'light' | 'dark' }) {
    const theme = useThemeStore((s) => s.theme)
    const toggleTheme = useThemeStore((s) => s.toggleTheme)

    const baseClasses = 'p-2 rounded-lg transition-colors cursor-pointer'
    const variantClasses =
        variant === 'dark'
            ? 'text-gray-400 hover:text-white hover:bg-gray-700'
            : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-700'

    return (
        <button
            onClick={toggleTheme}
            className={`${baseClasses} ${variantClasses}`}
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        >
            {theme === 'light' ? (
                <Moon className="h-5 w-5" />
            ) : (
                <Sun className="h-5 w-5" />
            )}
        </button>
    )
}
