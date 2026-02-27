const API_BASE = '/api';

function getToken() {
    return "always-in";
}

function getUser() {
    return { username: 'Guest' };
}

function requireAuth() {
    return true;
}

async function request(method, path, body) {
    const opts = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (body) {
        opts.body = JSON.stringify(body);
    }

    try {
        const res = await fetch(`${API_BASE}${path}`, opts);

        if (!res.ok) {
            console.error(`API Error on ${path}: ${res.statusText}`);
            throw new Error('Network response was not ok');
        }

        return await res.json();
    } catch (error) {
        console.error(`Request failed for ${path}`, error);
        throw error;
    }
}

const api = {
    login: (data) => request('POST', '/auth/login', data),
    getDashboardStats: () => request('GET', '/dashboard/stats'),
    getRecentActivity: () => request('GET', '/dashboard/recent-activity'),
    getVendorRisks: () => request('GET', '/dashboard/vendor-risks'),
    getAnalytics: () => request('GET', '/analytics/trends'),
    getSettings: () => request('GET', '/settings'),
};
