import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

import { useAuthStore } from '@/stores/useAuthStore'
import { useThemeStore } from '@/stores/useThemeStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'

// ── Zod Schema ────────────────────────────────────────
const loginSchema = z.object({
    email: z
        .string()
        .min(1, 'Email is required')
        .email('Invalid email format'),
    password: z
        .string()
        .min(1, 'Password is required'),
})

type LoginFormData = z.infer<typeof loginSchema>

// ── Component ─────────────────────────────────────────
export function LoginPage() {
    const navigate = useNavigate()
    const login = useAuthStore((s) => s.login)
    const isLoading = useAuthStore((s) => s.isLoading)

    const [showPassword, setShowPassword] = useState(false)
    const [rememberMe, setRememberMe] = useState(false)
    const theme = useThemeStore((s) => s.theme)

    const {
        register,
        handleSubmit,
        formState: { errors },
    } = useForm<LoginFormData>({
        resolver: zodResolver(loginSchema),
        defaultValues: { email: '', password: '' },
    })

    const onSubmit = async (data: LoginFormData) => {
        try {
            await login(data.email, data.password, rememberMe)
            toast.success('Welcome back!', { duration: 2000 })
            navigate('/dashboard', { replace: true })
        } catch {
            toast.error('Invalid credentials')
        }
    }

    const handleForgotPassword = () => {
        toast.info('Coming soon — password reset is not yet available.')
    }

    return (
        <Card className="shadow-lg border-0 dark:bg-gray-800 dark:text-gray-100 transition-colors">
            <CardHeader className="text-center pb-2">
                <img
                    src={theme === 'dark' ? '/deloitte_white.png' : '/deloitte.png'}
                    alt="Deloitte"
                    className="h-12 mx-auto mb-4"
                />
                <h1 className="text-2xl font-semibold text-gray-800 dark:text-white">
                    Sign in to your account
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Enter your credentials to access the dashboard
                </p>
            </CardHeader>

            <CardContent className="pt-4">
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
                    {/* Email Field */}
                    <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                            id="email"
                            type="email"
                            placeholder="you@company.com"
                            autoComplete="email"
                            disabled={isLoading}
                            {...register('email')}
                            className={errors.email ? 'border-red-500 focus-visible:ring-red-500' : ''}
                        />
                        {errors.email && (
                            <p className="text-sm text-red-500">{errors.email.message}</p>
                        )}
                    </div>

                    {/* Password Field */}
                    <div className="space-y-2">
                        <Label htmlFor="password">Password</Label>
                        <div className="relative">
                            <Input
                                id="password"
                                type={showPassword ? 'text' : 'password'}
                                placeholder="Enter your password"
                                autoComplete="current-password"
                                disabled={isLoading}
                                {...register('password')}
                                className={`pr-10 ${errors.password ? 'border-red-500 focus-visible:ring-red-500' : ''}`}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                tabIndex={-1}
                                aria-label={showPassword ? 'Hide password' : 'Show password'}
                            >
                                {showPassword ? (
                                    <EyeOff className="h-4 w-4" />
                                ) : (
                                    <Eye className="h-4 w-4" />
                                )}
                            </button>
                        </div>
                        {errors.password && (
                            <p className="text-sm text-red-500">{errors.password.message}</p>
                        )}
                    </div>

                    {/* Remember Me */}
                    <div className="flex items-center space-x-2">
                        <Checkbox
                            id="remember"
                            checked={rememberMe}
                            onCheckedChange={(checked) => setRememberMe(checked === true)}
                            disabled={isLoading}
                        />
                        <Label
                            htmlFor="remember"
                            className="text-sm font-normal text-gray-600 dark:text-gray-400 cursor-pointer"
                        >
                            Remember me
                        </Label>
                    </div>

                    {/* Submit Button */}
                    <Button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-deloitte hover:bg-deloitte-dark text-white font-medium cursor-pointer"
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Signing in...
                            </>
                        ) : (
                            'Sign In'
                        )}
                    </Button>

                    {/* Forgot Password */}
                    <div className="text-center">
                        <button
                            type="button"
                            onClick={handleForgotPassword}
                            className="text-sm text-deloitte hover:text-deloitte-dark hover:underline transition-colors cursor-pointer"
                        >
                            Forgot password?
                        </button>
                    </div>
                </form>
            </CardContent>
        </Card>
    )
}
