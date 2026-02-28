export function initSidebar(activePage) {
    console.log('[UI] Initializing Sidebar...');
    const sidebar = document.getElementById('sidebar');

    if (!sidebar) {
        console.warn('[UI] Sidebar container #sidebar not found');
        return;
    }

    const navItems = [
        { name: 'Dashboard', href: 'dashboard.html', icon: '📊' },
        { name: 'Graph Intel', href: 'graph.html', icon: '🕸️' },
        { name: 'Analytics', href: 'analytics.html', icon: '📈' },
        { name: 'Settings', href: 'settings.html', icon: '⚙️' }
    ];

    let navHTML = navItems.map(item => {
        const isActive = item.href.includes(activePage) ? 'active' : '';
        return `
            <a href="${item.href}" class="nav-link ${isActive}">
                <span style="font-size:18px">${item.icon}</span> 
                <span style="font-weight:500">${item.name}</span>
            </a>
        `;
    }).join('');

    sidebar.innerHTML = `
        <div class="sidebar-logo" style="margin-bottom:40px; text-align:left; padding-left:15px">
            <div style="font-size:10px; color:var(--text-dim); letter-spacing:2px; font-weight:700">CORE SYSTEM</div>
            <div style="font-size:22px; color:var(--accent-light); font-weight:800; letter-spacing:-1px">GST RECON</div>
        </div>
        
        <nav style="flex:1; display:flex; flex-direction:column; gap:5px">
            ${navHTML}
        </nav>
        
        <div style="padding-top:20px; border-top:1px solid rgba(255,255,255,0.05)">
            <button class="btn w-full" id="logout-btn"
                    style="background:rgba(255,255,255,0.03); box-shadow:none; border:1px solid rgba(255,255,255,0.05); color:var(--text-dim); transition: 0.3s;">
                LOGOUT
            </button>
        </div>
    `;

    document.getElementById('logout-btn').onmouseenter = (e) => e.target.style.background = 'rgba(244, 63, 94, 0.1)';
    document.getElementById('logout-btn').onmouseleave = (e) => e.target.style.background = 'rgba(255,255,255,0.03)';
}

export function formatINR(value) {
    const num = Number(value || 0);
    return '₹' + num.toLocaleString('en-IN');
}

export function riskBadge(score) {
    let badgeType = 'success';
    if (score > 40) badgeType = 'warning';
    if (score > 70) badgeType = 'danger';
    return `<span class="badge badge-${badgeType}">${score}%</span>`;
}

export async function initHeader(api) {
    console.log('[UI] Initializing Profile Header...');
    const mainContent = document.querySelector('.main-content');
    if (!mainContent) return;

    let header = mainContent.querySelector('header');
    if (!header) {
        header = document.createElement('header');
        mainContent.prepend(header);
    }

    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.marginBottom = '40px';

    try {
        if (!api) throw new Error('API object not provided to initHeader');

        const profile = await api.getProfile();
        console.log('[UI] Profile loaded:', profile.email);

        const initials = profile.name ? profile.name.split(' ').filter(n => n).map(n => n[0]).join('').toUpperCase().substring(0, 2) : '??';
        const profilePicUrl = `/api/user/profile-pic/${profile.id}`;

        const profileHTML = `
            <div class="profile-container" id="profile-container" style="margin-left: 20px; flex-shrink: 0;">
                <div class="profile-icon" id="profile-trigger">
                    ${profile.profile_pic ? `<img src="${profilePicUrl}" alt="Profile">` : `<span>${initials}</span>`}
                </div>
                <div class="profile-dropdown" id="profile-dropdown">
                    <div class="profile-dropdown-info">
                        <div class="profile-dropdown-pic">
                            ${profile.profile_pic ? `<img src="${profilePicUrl}" alt="Profile">` : `<span>${initials}</span>`}
                        </div>
                        <h4>${profile.name || 'User'}</h4>
                        <p class="email">${profile.email}</p>
                        <div style="margin-top:10px">
                            <span class="gst-badge">GSTIN: ${profile.gst_number || 'Not Set'}</span>
                        </div>
                    </div>
                    <button class="btn w-full" id="profile-logout-btn" style="background:rgba(244, 63, 94, 0.1); color:var(--danger); border:1px solid rgba(244, 63, 94, 0.2); box-shadow:none; font-size:12px; padding:10px;">LOGOUT SYSTEM</button>
                </div>
            </div>
        `;

        header.insertAdjacentHTML('beforeend', profileHTML);

        const trigger = document.getElementById('profile-trigger');
        const dropdown = document.getElementById('profile-dropdown');

        if (trigger && dropdown) {
            trigger.onclick = (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('active');
            };
            document.addEventListener('click', () => dropdown.classList.remove('active'));
            dropdown.onclick = (e) => e.stopPropagation();
            document.getElementById('profile-logout-btn').onclick = () => api.logout();
        }
    } catch (err) {
        console.error('[UI] Header Init Failed:', err);
    }
}
