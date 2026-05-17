import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/stores/useAuthStore'

/**
 * Route guard component.
 *
 * Wrap protected routes with this:
 *   <Route element={<PrivateRoute />}>
 *     <Route path="/dashboard" element={<Dashboard />} />
 *   </Route>
 *
 * If the user is not authenticated, they are redirected to /login.
 * The `replace` prop prevents the login redirect from being added
 * to browser history (so "Back" doesn't loop back to the redirect).
 */
export function PrivateRoute() {
    const isAuthenticated = useAuthStore((state) => state.isAuthenticated)

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />
    }

    return <Outlet />
}
