import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/stores/useAuthStore'

/**
 * GuestRoute — Opposite of PrivateRoute.
 *
 * If user IS authenticated → redirect to /dashboard
 * If user is NOT authenticated → render child routes (login, etc.)
 *
 * Prevents authenticated users from seeing the login page.
 */
export function GuestRoute() {
    const isAuthenticated = useAuthStore((s) => s.isAuthenticated)

    if (isAuthenticated) {
        return <Navigate to="/dashboard" replace />
    }

    return <Outlet />
}
