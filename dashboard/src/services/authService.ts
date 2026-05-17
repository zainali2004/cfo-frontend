/**
 * Mock Authentication Service
 *
 * HOT-SWAPPABLE: This is the ONLY file that needs to change
 * when connecting to the real FastAPI backend.
 *
 * Today:  Hard-coded credentials, fake token
 * Later:  POST /login via Axios → real JWT token
 */

// ── Types ──────────────────────────────────────────────
export interface AuthUser {
    email: string
    name: string
}

export interface LoginResponse {
    token: string
    user: AuthUser
}

// ── Mock Data ──────────────────────────────────────────
const MOCK_USERS = [
    { email: 'admin@company.com', password: 'admin123', name: 'Admin User' },
    { email: 'analyst@company.com', password: 'analyst123', name: 'Analyst User' },
]

// ── Service ────────────────────────────────────────────

/**
 * Authenticate a user with email and password.
 *
 * MOCK: Validates against hard-coded credentials above.
 * REAL: Replace body with `api.post('/login', { email, password })`
 */
export async function loginUser(
    email: string,
    password: string
): Promise<LoginResponse> {
    // Simulate network delay (300–600ms)
    await new Promise((resolve) => setTimeout(resolve, 300 + Math.random() * 300))

    const match = MOCK_USERS.find(
        (u) => u.email.toLowerCase() === email.toLowerCase() && u.password === password
    )

    if (!match) {
        throw new Error('Invalid credentials')
    }

    return {
        token: `mock-jwt-${match.email}-${Date.now()}`,
        user: { email: match.email, name: match.name },
    }
}

/**
 * Validate if a token is still valid.
 *
 * MOCK: Always returns true if token starts with 'mock-jwt-'
 * REAL: Replace with GET /me or token introspection endpoint
 */
export async function validateToken(token: string): Promise<AuthUser | null> {
    if (!token.startsWith('mock-jwt-')) return null

    // Extract email from mock token format: mock-jwt-{email}-{timestamp}
    const parts = token.split('-')
    const email = parts.slice(2, -1).join('-') // handles email with hyphens... but our emails use @

    const match = MOCK_USERS.find((u) => u.email === email)
    return match ? { email: match.email, name: match.name } : null
}
