// ── API Base URL Configuration ─────────────────────────────────────────
function getApiBase() {
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0') {
        return '/api';
    }
    const tunnelUrl = sessionStorage.getItem('backend_url') || window.BACKEND_URL || '';
    if (tunnelUrl) {
        return tunnelUrl.replace(/\/$/, '') + '/api';
    }
    return '/api';
}

const API_BASE = getApiBase();

// ── High Security Session Management ───────────────────────────────────

// 1. Use sessionStorage instead of localStorage (per-tab lifecycle)
export async function getToken() {
    return sessionStorage.getItem('auth_token');
}

export async function getUser() {
    const token = await getToken();
    if (!token) return null;

    try {
        const parts = token.split('.');
        const payload = JSON.parse(atob(parts[1]));

        if (payload.exp && (payload.exp * 1000) < Date.now()) {
            api.logout();
            return null;
        }
        return payload;
    } catch (e) {
        api.logout();
        return null;
    }
}

export async function requireAuth() {
    const user = await getUser();
    if (!user) {
        window.location.href = 'index.html';
        return false;
    }
    return true;
}

// 2. Immediate Termination when user leaves (tab switch / minimize)
// Uses visibilitychange instead of pagehide to avoid firing on same-site navigation.
// sessionStorage already auto-clears on tab close, so that case is covered.
let leaveTimer;
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        // Start a 1-second countdown — if tab stays hidden, wipe session
        leaveTimer = setTimeout(() => {
            sessionStorage.removeItem('auth_token');
            if (window.location.pathname !== '/index.html' && window.location.pathname !== '/') {
                window.location.href = 'index.html';
            }
        }, 1000);
    } else {
        // User came back before 1s — cancel wipe
        clearTimeout(leaveTimer);
    }
});

// 3. Auto-Logout on Inactivity (10 Minutes)
let inactivityTimer;
const resetInactivityTimer = () => {
    clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(() => {
        console.warn('High security: Logging out due to inactivity');
        api.logout();
    }, 10 * 60 * 1000); // 10 minutes
};

// Listen for interactions to reset idle timer
if (typeof window !== 'undefined') {
    ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(evt =>
        window.addEventListener(evt, resetInactivityTimer, { passive: true })
    );
    resetInactivityTimer();
}

async function request(method, path, body) {
    const token = await getToken();
    const isFormData = body instanceof FormData;

    const opts = {
        method,
        headers: {
            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        }
    };

    if (!isFormData) {
        opts.headers['Content-Type'] = 'application/json';
        if (body) opts.body = JSON.stringify(body);
    } else {
        opts.body = body;
    }

    try {
        const res = await fetch(`${API_BASE}${path}`, opts);
        if (res.status === 401) {
            api.logout(); // Immediate logout on unauthorized
        }
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || 'Request failed');
        }
        return await res.json();
    } catch (error) {
        console.error(`Request failed for ${path}`, error);
        throw error;
    }
}

export const api = {
    login: async (email, password) => {
        const data = await request('POST', '/auth/login', { email, password });
        if (data.access_token) {
            sessionStorage.setItem('auth_token', data.access_token);
        }
        return data; // Might have otp_required: true
    },
    verifyOtp: async (email, otp) => {
        const data = await request('POST', '/auth/verify-otp', { email, otp });
        sessionStorage.setItem('auth_token', data.access_token);
        return data;
    },
    register: async (payload) => {
        const data = await request('POST', '/auth/register', payload);
        sessionStorage.setItem('auth_token', data.access_token);
        return data;
    },
    uploadInvoice: async (formData) => {
        return await request('POST', '/invoices/upload', formData);
    },
    logout: () => {
        sessionStorage.removeItem('auth_token');
        if (window.location.pathname !== '/index.html') {
            window.location.href = 'index.html';
        }
    },
    getDashboardStats: () => request('GET', '/dashboard/stats'),
    getRecentActivity: () => request('GET', '/dashboard/recent-activity'),
    getVendorRisks: () => request('GET', '/dashboard/vendor-risks'),
    getAnalytics: () => request('GET', '/analytics/detailed'),
    getSettings: () => request('GET', '/settings'),
    getProfile: () => request('GET', '/user/profile'),
    updateProfile: (data) => request('PATCH', '/user/profile', data),

    // Knowledge Graph APIs
    getUserGraph: () => request('GET', '/graph/user'),
    getAdminGraph: () => request('GET', '/graph/admin'),
    getGraphNode: (gstin) => request('GET', `/graph/node/${gstin}`),

    // Fraud Intelligence APIs
    getEntityRisk: (gstin) => request('GET', `/fraud/entity/${gstin}`),
    getCycles: () => request('GET', '/fraud/cycles'),
    runDbScan: () => request('GET', '/fraud/db-scan'),
};
