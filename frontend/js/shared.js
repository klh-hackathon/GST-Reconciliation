function initSidebar(activePage) {
    const sidebar = document.getElementById('sidebar');

    if (!sidebar) {
        return;
    }

    const navItems = [
        { name: 'Dashboard', href: '/dashboard.html', icon: '📊' },
        { name: 'Analytics', href: '/analytics.html', icon: '📈' },
        { name: 'Settings', href: '/settings.html', icon: '⚙️' }
    ];

    let navHTML = navItems.map(item => {
        const isActive = item.href.includes(activePage) ? 'active' : '';

        return `
            <a href="${item.href}" class="nav-link ${isActive}">
                <span>${item.icon}</span> 
                ${item.name}
            </a>
        `;
    }).join('');

    sidebar.innerHTML = `
        <div class="sidebar-logo">GST ENGINE</div>
        
        <nav style="flex:1">
            ${navHTML}
        </nav>
        
        <div style="padding-top:20px; border-top:1px solid rgba(255,255,255,0.05)">
            <button class="btn w-full" 
                    style="background:transparent; box-shadow:none; border:1px solid rgba(255,255,255,0.1)" 
                    onclick="window.location.href='/index.html'">
                LOGOUT
            </button>
        </div>
    `;
}

function formatINR(value) {
    const num = Number(value || 0);
    return '₹' + num.toLocaleString('en-IN');
}

function riskBadge(score) {
    let badgeType = 'success';

    if (score > 40) {
        badgeType = 'warning';
    }
    if (score > 70) {
        badgeType = 'danger';
    }

    return `<span class="badge badge-${badgeType}">${score}%</span>`;
}
