import { create } from 'zustand'
import { loginUser, validateToken, type AuthUser } from '@/services/authService'

// ── Storage Keys ──────────────────────────────────────
const TOKEN_KEY = 'auth_token'
const USER_KEY = 'auth_user'
const REMEMBER_KEY = 'auth_remember'

// ── Helpers ───────────────────────────────────────────

/** Save auth data to the appropriate storage */
function persistAuth(token: string, user: AuthUser, rememberMe: boolean) {
    // Always save the remember preference to localStorage
    // (so hydrate() knows where to look)
    localStorage.setItem(REMEMBER_KEY, String(rememberMe))

    const storage = rememberMe ? localStorage : sessionStorage
    storage.setItem(TOKEN_KEY, token)
    storage.setItem(USER_KEY, JSON.stringify(user))
}

/** Clear auth data from BOTH storages (safe cleanup) */
function clearAuth() {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    localStorage.removeItem(REMEMBER_KEY)
    sessionStorage.removeItem(TOKEN_KEY)
    sessionStorage.removeItem(USER_KEY)
}

// ── Store Interface ───────────────────────────────────
interface AuthState {
    user: AuthUser | null
    token: string | null
    isAuthenticated: boolean
    rememberMe: boolean
    isLoading: boolean

    login: (email: string, password: string, rememberMe: boolean) => Promise<void>
    logout: () => void
    hydrate: () => Promise<void>
}

// ── Store ─────────────────────────────────────────────
export const useAuthStore = create<AuthState>((set) => ({
    user: { email: 'admin@company.com', name: 'Admin User' },
    token: 'bypass-token',
    isAuthenticated: true,
    rememberMe: false,
    isLoading: false,

    /**
     * Authenticate user via authService (mock or real).
     * Saves token to localStorage or sessionStorage based on rememberMe.
     */
    login: async (email, password, rememberMe) => {
        set({ isLoading: true })
        try {
            const { token, user } = await loginUser(email, password)
            persistAuth(token, user, rememberMe)
            set({
                user,
                token,
                isAuthenticated: true,
                rememberMe,
                isLoading: false,
            })
        } catch (error) {
            set({ isLoading: false })
            throw error // Re-throw so LoginPage can show error toast
        }
    },

    /**
     * Clear all auth state and storage. Redirect handled by caller.
     */
    logout: () => {
        clearAuth()
        set({
            user: null,
            token: null,
            isAuthenticated: false,
            rememberMe: false,
            isLoading: false,
        })
    },

    /**
     * Restore session on app startup.
     * Checks localStorage first (persisted), then sessionStorage (temporary).
     * Validates token is still good via authService.
     */
    hydrate: async () => {
        // Determine which storage was used
        const remember = localStorage.getItem(REMEMBER_KEY) === 'true'
        const storage = remember ? localStorage : sessionStorage

        const token = storage.getItem(TOKEN_KEY)
        const userJson = storage.getItem(USER_KEY)

        if (!token || !userJson) {
            // No saved session — stay logged out
            return
        }

        try {
            // Validate the token is still valid
            const user = await validateToken(token)
            if (user) {
                set({
                    user,
                    token,
                    isAuthenticated: true,
                    rememberMe: remember,
                })
            } else {
                // Token invalid — clean up
                clearAuth()
            }
        } catch {
            clearAuth()
        }
    },
}))
