import { create } from 'zustand'

const THEME_KEY = 'theme_preference'

type Theme = 'light' | 'dark'

interface ThemeState {
    theme: Theme
    toggleTheme: () => void
}

/**
 * Apply or remove the `dark` class on <html>.
 * Tailwind v4 uses CSS `@custom-variant dark (&:where(.dark, .dark *))` by default.
 */
function applyTheme(theme: Theme) {
    const root = document.documentElement
    if (theme === 'dark') {
        root.classList.add('dark')
    } else {
        root.classList.remove('dark')
    }
}

// Read initial theme from localStorage (default: light)
function getInitialTheme(): Theme {
    const saved = localStorage.getItem(THEME_KEY) as Theme | null
    if (saved === 'dark' || saved === 'light') return saved
    // Respect OS preference as fallback
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark'
    return 'light'
}

const initialTheme = getInitialTheme()
applyTheme(initialTheme)

export const useThemeStore = create<ThemeState>((set) => ({
    theme: initialTheme,

    toggleTheme: () =>
        set((state) => {
            const next: Theme = state.theme === 'light' ? 'dark' : 'light'
            localStorage.setItem(THEME_KEY, next)
            applyTheme(next)
            return { theme: next }
        }),
}))
