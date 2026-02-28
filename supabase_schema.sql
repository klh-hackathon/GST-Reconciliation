-- ═══════════════════════════════════════════════════════════════════════
-- KNOWLEDGE GRAPH GST SYSTEM — FULL DATABASE SCHEMA
-- ═══════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── 1. Users Table ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.recon_users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    login_type TEXT NOT NULL CHECK (login_type IN ('user', 'admin')),
    gst_number TEXT,
    profile_pic BYTEA,
    
    gstr_1 JSONB,
    gstr_2b JSONB,
    gstr_3b JSONB,
    e_invoice JSONB,
    e_way_bill BYTEA,
    json_data JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_gst ON recon_users(gst_number);
CREATE INDEX IF NOT EXISTS idx_users_email ON recon_users(email);


-- ── 2. Invoices Table (normalised, deduplicated) ─────────────────────
CREATE TABLE IF NOT EXISTS public.invoices (
    id BIGSERIAL PRIMARY KEY,
    
    invoice_number TEXT NOT NULL,
    supplier_gstin TEXT NOT NULL,
    return_period TEXT NOT NULL DEFAULT '',
    
    buyer_gstin TEXT,
    invoice_date TEXT,
    invoice_value NUMERIC(15,2) DEFAULT 0,
    taxable_value NUMERIC(15,2) DEFAULT 0,
    igst NUMERIC(12,2) DEFAULT 0,
    cgst NUMERIC(12,2) DEFAULT 0,
    sgst NUMERIC(12,2) DEFAULT 0,
    gst_rate_percent NUMERIC(5,2) DEFAULT 0,
    
    invoice_type TEXT,
    place_of_supply TEXT,
    hsn_code TEXT,
    source TEXT CHECK (source IN ('gstr_1','gstr_2b','gstr_3b','e_invoice','upload')),
    uploaded_by TEXT REFERENCES recon_users(id),
    file_name TEXT,
    
    is_duplicate BOOLEAN DEFAULT FALSE,
    risk_score INTEGER DEFAULT 0,
    risk_level TEXT DEFAULT 'LOW',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_invoice_dedup UNIQUE (invoice_number, supplier_gstin, return_period)
);

CREATE INDEX IF NOT EXISTS idx_inv_supplier ON invoices(supplier_gstin);
CREATE INDEX IF NOT EXISTS idx_inv_buyer ON invoices(buyer_gstin);
CREATE INDEX IF NOT EXISTS idx_inv_period ON invoices(return_period);
CREATE INDEX IF NOT EXISTS idx_inv_source ON invoices(source);
CREATE INDEX IF NOT EXISTS idx_inv_uploaded_by ON invoices(uploaded_by);


-- ── 3. Graph Nodes Table ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.graph_nodes (
    id BIGSERIAL PRIMARY KEY,
    gstin TEXT UNIQUE NOT NULL,
    
    node_type TEXT CHECK (node_type IN (
        'registered_user',
        'external_known',
        'external_unknown'
    )),
    
    user_id TEXT REFERENCES recon_users(id),
    
    total_invoices_as_supplier INTEGER DEFAULT 0,
    total_invoices_as_buyer INTEGER DEFAULT 0,
    total_value_supplied NUMERIC(18,2) DEFAULT 0,
    total_value_received NUMERIC(18,2) DEFAULT 0,
    
    risk_score INTEGER DEFAULT 0,
    risk_level TEXT DEFAULT 'LOW',
    in_circular_trade BOOLEAN DEFAULT FALSE,
    
    label TEXT,
    state_code TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_node_gstin ON graph_nodes(gstin);
CREATE INDEX IF NOT EXISTS idx_node_user ON graph_nodes(user_id);


-- ── 4. Graph Edges Table (directed: Supplier → Buyer) ────────────────
CREATE TABLE IF NOT EXISTS public.graph_edges (
    id BIGSERIAL PRIMARY KEY,
    
    source_gstin TEXT NOT NULL,
    target_gstin TEXT NOT NULL,
    
    invoice_count INTEGER DEFAULT 1,
    total_taxable_value NUMERIC(18,2) DEFAULT 0,
    total_tax_amount NUMERIC(15,2) DEFAULT 0,
    return_period TEXT,
    
    invoice_id BIGINT,
    invoice_date TEXT,
    
    risk_score INTEGER DEFAULT 0,
    is_circular BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edge_source ON graph_edges(source_gstin);
CREATE INDEX IF NOT EXISTS idx_edge_target ON graph_edges(target_gstin);


-- ── 5. Fraud Scan History ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.fraud_scans (
    id BIGSERIAL PRIMARY KEY,
    scan_type TEXT CHECK (scan_type IN ('user_upload', 'db_scan', 'admin_audit')),
    triggered_by TEXT REFERENCES recon_users(id),
    
    total_invoices_scanned INTEGER DEFAULT 0,
    summary_low INTEGER DEFAULT 0,
    summary_medium INTEGER DEFAULT 0,
    summary_high INTEGER DEFAULT 0,
    summary_critical INTEGER DEFAULT 0,
    
    cycles_detected JSONB DEFAULT '[]',
    top_rules JSONB DEFAULT '{}',
    full_results JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);
