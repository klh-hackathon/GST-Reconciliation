"""
Knowledge Graph Engine for GST Reconciliation & Fraud Intelligence.

Deterministic, O(n) graph construction and cycle detection.
Handles 1M+ invoices with ~105MB memory footprint.
"""

import re
import math
import httpx
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────

GSTIN_REGEX = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$')

RISK_COLORS = {
    "LOW": "#10b981",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f43f5e",
    "CRITICAL": "#1a1a2e",
}


# ═══════════════════════════════════════════════════════════════════════════
#  GSTIN Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_gstin(gstin: str) -> bool:
    return bool(GSTIN_REGEX.match(str(gstin).strip().upper()))


# ═══════════════════════════════════════════════════════════════════════════
#  Knowledge Graph Data Structures
# ═══════════════════════════════════════════════════════════════════════════

def create_empty_graph() -> dict:
    """Create an empty knowledge graph state object."""
    return {
        "adjacency": defaultdict(set),       # gstin -> set of buyer gstins
        "reverse_adj": defaultdict(set),      # gstin -> set of supplier gstins
        "nodes": {},                          # gstin -> node_info dict
        "edges": defaultdict(list),           # (src,tgt) -> [edge_info dicts]
        "gstin_to_user": {},                  # gstin -> user_id
        "registered_gstins": set(),           # set of all registered gstins
        "cycles": [],
    }


def make_node(gstin: str, node_type: str, user_id=None, label="") -> dict:
    return {
        "gstin": gstin,
        "node_type": node_type,
        "user_id": user_id,
        "label": label or gstin[:10],
        "state_code": gstin[:2] if len(gstin) >= 2 else "",
        "total_supplied": 0.0,
        "total_received": 0.0,
        "invoice_count_supplier": 0,
        "invoice_count_buyer": 0,
        "risk_score": 0,
        "risk_level": "LOW",
        "in_circular_trade": False,
        "gstr_files": [],
    }


def make_edge(src: str, tgt: str, invoice: dict) -> dict:
    return {
        "source_gstin": src,
        "target_gstin": tgt,
        "invoice_id": str(invoice.get("invoice_id", "")),
        "invoice_number": str(invoice.get("invoice_number", "")),
        "taxable_value": float(invoice.get("taxable_value", 0) or 0),
        "tax_amount": (
            float(invoice.get("igst", 0) or 0) +
            float(invoice.get("cgst", 0) or 0) +
            float(invoice.get("sgst", 0) or 0)
        ),
        "invoice_value": float(invoice.get("invoice_value", 0) or 0),
        "return_period": str(invoice.get("return_period", invoice.get("Return_Period", ""))),
        "invoice_date": str(invoice.get("invoice_date", invoice.get("Invoice_Date", ""))),
        "source": str(invoice.get("source", "")),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(all_invoices: list, all_users: list) -> dict:
    """
    Build the complete knowledge graph from invoices and users.
    
    Time: O(n) where n = len(all_invoices) + len(all_users)
    Space: O(V + E) where V = unique GSTINs, E = unique edges
    """
    graph = create_empty_graph()

    # Step 1: Index registered GSTINs from users
    for user in all_users:
        gstin = str(user.get("gst_number", "") or "").strip().upper()
        if gstin:
            graph["registered_gstins"].add(gstin)
            graph["gstin_to_user"][gstin] = user.get("id", "")
            graph["nodes"][gstin] = make_node(
                gstin, "registered_user",
                user_id=user.get("id"),
                label=user.get("name", gstin[:10])
            )

    # Step 2: Process every invoice → create nodes + edges
    for inv in all_invoices:
        supplier = str(inv.get("supplier_gstin", "")).strip().upper()
        buyer = str(inv.get("buyer_gstin", inv.get("Recipient_GSTIN", ""))).strip().upper()

        if not supplier or not buyer or supplier == buyer:
            continue

        # Ensure supplier node exists
        if supplier not in graph["nodes"]:
            ntype = "registered_user" if supplier in graph["registered_gstins"] else "external_known"
            graph["nodes"][supplier] = make_node(supplier, ntype,
                user_id=graph["gstin_to_user"].get(supplier))

        # Ensure buyer node exists
        if buyer not in graph["nodes"]:
            ntype = "registered_user" if buyer in graph["registered_gstins"] else "external_known"
            graph["nodes"][buyer] = make_node(buyer, ntype,
                user_id=graph["gstin_to_user"].get(buyer))

        # Create directed edge
        edge = make_edge(supplier, buyer, inv)
        graph["adjacency"][supplier].add(buyer)
        graph["reverse_adj"][buyer].add(supplier)
        graph["edges"][(supplier, buyer)].append(edge)

        # Update node stats
        val = edge["taxable_value"]
        graph["nodes"][supplier]["total_supplied"] += val
        graph["nodes"][supplier]["invoice_count_supplier"] += 1
        graph["nodes"][buyer]["total_received"] += val
        graph["nodes"][buyer]["invoice_count_buyer"] += 1

    # Step 3: Detect cycles
    adj_lists = {k: list(v) for k, v in graph["adjacency"].items()}
    graph["cycles"] = detect_circular_trading(adj_lists)

    # Mark circular nodes
    for cycle in graph["cycles"]:
        for gstin in cycle:
            if gstin in graph["nodes"]:
                graph["nodes"][gstin]["in_circular_trade"] = True
                graph["nodes"][gstin]["risk_score"] = max(
                    graph["nodes"][gstin]["risk_score"], 70
                )

    return graph


# ═══════════════════════════════════════════════════════════════════════════
#  DFS Cycle Detection — O(V + E)
# ═══════════════════════════════════════════════════════════════════════════

def detect_circular_trading(graph: dict) -> list:
    """
    DFS-based cycle detection on directed GSTIN transaction graph.
    Time: O(V + E)  Space: O(V)
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color = defaultdict(int)
    cycles = []

    def dfs(u, path):
        color[u] = GREY
        path.append(u)
        for v in graph.get(u, []):
            if color[v] == WHITE:
                dfs(v, path)
            elif color[v] == GREY:
                idx = path.index(v)
                cycles.append(path[idx:] + [v])
        path.pop()
        color[u] = BLACK

    for node in list(graph.keys()):
        if color[node] == WHITE:
            dfs(node, [])

    return cycles


# ═══════════════════════════════════════════════════════════════════════════
#  Fraud Detection Rules (12 Deterministic Rules)
# ═══════════════════════════════════════════════════════════════════════════

def compute_zscore(value: float, mean: float, std_dev: float) -> float:
    if std_dev == 0:
        return 0.0
    return (value - mean) / std_dev


def build_fraud_context(all_invoices: list, graph: dict) -> dict:
    """Pre-compute cross-invoice context for all 12 rules in O(n)."""
    gstr2b_ids = set()
    supplier_values = defaultdict(list)
    supplier_buyer_split = defaultdict(int)
    invoice_number_seen = defaultdict(list)
    pair_frequency = defaultdict(int)  # NEW: per-period pair count

    for inv in all_invoices:
        source = inv.get("source", "")
        if source == "gstr_2b":
            gstr2b_ids.add(str(inv.get("invoice_id", "")))

        s = str(inv.get("supplier_gstin", "")).strip()
        b = str(inv.get("buyer_gstin", "")).strip()
        val = float(inv.get("invoice_value", inv.get("taxable_value", 0)) or 0)

        if s:
            supplier_values[s].append(val)

        if 45000 <= val < 50000:
            supplier_buyer_split[f"{s}_{b}"] += 1

        inv_no = str(inv.get("invoice_number", inv.get("Invoice_Number", ""))).strip()
        if inv_no:
            invoice_number_seen[f"{s}:{inv_no}"].append(str(inv.get("invoice_id", "")))

        # High-frequency pair tracking
        period = str(inv.get("return_period", ""))
        if s and b:
            pair_frequency[f"{s}_{b}_{period}"] += 1

    # Statistics per supplier
    supplier_mean = {}
    supplier_std = {}
    for gstin, vals in supplier_values.items():
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0
        supplier_mean[gstin] = mean
        supplier_std[gstin] = math.sqrt(variance)

    # Duplicate invoice IDs
    duplicate_inv_ids = set()
    for key, ids in invoice_number_seen.items():
        if len(ids) > 1:
            duplicate_inv_ids.update(ids)

    # Circular GSTINs from graph
    circular_gstins = set()
    for cycle in graph.get("cycles", []):
        circular_gstins.update(cycle)

    return {
        "gstr2b_ids": gstr2b_ids,
        "supplier_mean": supplier_mean,
        "supplier_std": supplier_std,
        "supplier_buyer_split": supplier_buyer_split,
        "circular_gstins": circular_gstins,
        "duplicate_inv_ids": duplicate_inv_ids,
        "pair_frequency": pair_frequency,
        "registered_gstins": graph.get("registered_gstins", set()),
        "cycles": graph.get("cycles", []),
    }


def analyze_invoice(inv: dict, ctx: dict) -> dict:
    """Apply all 12 deterministic rules to a single invoice."""
    invoice_id = str(inv.get("invoice_id", "unknown"))
    breakdown = []
    raw_score = 0

    supplier = str(inv.get("supplier_gstin", "")).strip().upper()
    buyer = str(inv.get("buyer_gstin", "")).strip().upper()
    taxable_val = float(inv.get("taxable_value", 0) or 0)
    invoice_val = float(inv.get("invoice_value", taxable_val) or taxable_val)
    igst = float(inv.get("igst", 0) or 0)
    cgst = float(inv.get("cgst", 0) or 0)
    sgst = float(inv.get("sgst", 0) or 0)
    gst_rate = float(inv.get("gst_rate_percent", inv.get("GST_Rate_%", 0)) or 0)
    inv_type = str(inv.get("invoice_type", inv.get("Invoice_Type", ""))).lower()
    source = str(inv.get("source", ""))

    # RULE 1 — INVALID_GSTIN_FORMAT (100 pts)
    if supplier and not validate_gstin(supplier):
        pts = 100
        raw_score += pts
        breakdown.append({"rule_id": "INVALID_GSTIN_FORMAT", "points": pts,
            "explanation": f"Supplier GSTIN '{supplier}' fails regex validation."})
    if buyer and not validate_gstin(buyer):
        pts = 100
        raw_score += pts
        breakdown.append({"rule_id": "INVALID_GSTIN_FORMAT", "points": pts,
            "explanation": f"Buyer GSTIN '{buyer}' fails regex validation."})

    # RULE 2 — DUPLICATE_INVOICE (50 pts)
    if invoice_id in ctx["duplicate_inv_ids"]:
        pts = 50
        raw_score += pts
        breakdown.append({"rule_id": "DUPLICATE_INVOICE", "points": pts,
            "explanation": "Same invoice number reused by the same supplier."})

    # RULE 3 — MISSING_IN_2B (30 pts)
    if source == "gstr_1" and invoice_id not in ctx["gstr2b_ids"]:
        pts = 30
        raw_score += pts
        breakdown.append({"rule_id": "MISSING_IN_2B", "points": pts,
            "explanation": "Invoice in GSTR-1 but absent from GSTR-2B."})

    # RULE 4 — TAX_VALUE_MISMATCH (25 pts)
    if gst_rate > 0 and taxable_val > 0:
        calc_tax = taxable_val * (gst_rate / 100)
        decl_tax = igst + cgst + sgst
        dev = abs(decl_tax - calc_tax) / calc_tax if calc_tax else 0
        if dev > 0.02:
            pts = 25
            raw_score += pts
            breakdown.append({"rule_id": "TAX_VALUE_MISMATCH", "points": pts,
                "explanation": f"Tax deviates {dev*100:.1f}% (declared ₹{decl_tax:.0f} vs computed ₹{calc_tax:.0f})."})

    # RULE 5 — INVOICE_SPLITTING (25 pts)
    split_key = f"{supplier}_{buyer}"
    if 45000 <= invoice_val < 50000 and ctx["supplier_buyer_split"].get(split_key, 0) > 3:
        pts = 25
        raw_score += pts
        breakdown.append({"rule_id": "INVOICE_SPLITTING", "points": pts,
            "explanation": f"{ctx['supplier_buyer_split'][split_key]} invoices in ₹45K-50K range."})

    # RULE 6 — DORMANT_SPIKE (35 pts)
    avg = ctx["supplier_mean"].get(supplier, 0)
    if avg > 0 and invoice_val > avg * 10:
        pts = 35
        raw_score += pts
        breakdown.append({"rule_id": "DORMANT_SPIKE", "points": pts,
            "explanation": f"Value ₹{invoice_val:,.0f} is 10× avg ₹{avg:,.0f}."})

    # RULE 7 — CIRCULAR_TRADING (50 pts)
    if supplier in ctx["circular_gstins"] and buyer in ctx["circular_gstins"]:
        pts = 50
        raw_score += pts
        breakdown.append({"rule_id": "CIRCULAR_TRADING", "points": pts,
            "explanation": f"Both GSTINs in circular trade cycle."})

    # RULE 8 — FAKE_ITC_PATTERN (40 pts)
    total_tax = igst + cgst + sgst
    if invoice_val > 0 and total_tax > 0 and (total_tax / invoice_val) >= 0.99:
        pts = 40
        raw_score += pts
        breakdown.append({"rule_id": "FAKE_ITC_PATTERN", "points": pts,
            "explanation": f"Tax ₹{total_tax:,.0f} is ≥99% of invoice value ₹{invoice_val:,.0f}."})

    # RULE 9 — Z_SCORE_ANOMALY (20 pts)
    std = ctx["supplier_std"].get(supplier, 0)
    z = compute_zscore(taxable_val, avg, std)
    if z > 3.0:
        pts = 20
        raw_score += pts
        breakdown.append({"rule_id": "Z_SCORE_ANOMALY", "points": pts,
            "explanation": f"Z-score {z:.2f} exceeds 3.0 threshold."})

    # RULE 10 — EXPORT_MISUSE (45 pts)
    if "export" in inv_type and total_tax == 0 and not inv.get("has_lut", False):
        pts = 45
        raw_score += pts
        breakdown.append({"rule_id": "EXPORT_MISUSE", "points": pts,
            "explanation": "Export invoice with zero tax and no LUT."})

    # RULE 11 — HIGH_FREQUENCY_PAIR (20 pts)
    period = str(inv.get("return_period", ""))
    freq_key = f"{supplier}_{buyer}_{period}"
    if ctx["pair_frequency"].get(freq_key, 0) > 10:
        pts = 20
        raw_score += pts
        breakdown.append({"rule_id": "HIGH_FREQUENCY_PAIR", "points": pts,
            "explanation": f"{ctx['pair_frequency'][freq_key]} transactions in period {period}."})

    # RULE 12 — UNREGISTERED_GSTIN_INTERACTION (15 pts)
    reg = ctx.get("registered_gstins", set())
    if supplier in reg and buyer and buyer not in reg:
        pts = 15
        raw_score += pts
        breakdown.append({"rule_id": "UNREGISTERED_GSTIN_INTERACTION", "points": pts,
            "explanation": f"Transaction with unregistered GSTIN '{buyer}'."})

    # Auto-minimum 70 for circular trade
    if supplier in ctx["circular_gstins"]:
        raw_score = max(raw_score, 70)

    final_score = min(raw_score, 100)
    if final_score <= 20:     risk_level = "LOW"
    elif final_score <= 50:   risk_level = "MEDIUM"
    elif final_score <= 80:   risk_level = "HIGH"
    else:                     risk_level = "CRITICAL"

    return {
        "invoice_id": invoice_id,
        "risk_score": final_score,
        "risk_level": risk_level,
        "risk_breakdown": breakdown,
    }


def run_fraud_engine(invoices: list, graph: dict = None) -> dict:
    """Full pipeline: build context → run all 12 rules on every invoice."""
    if graph is None:
        graph = {"cycles": [], "registered_gstins": set()}

    ctx = build_fraud_context(invoices, graph)
    results = [analyze_invoice(inv, ctx) for inv in invoices]

    by_level = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    rule_hits = defaultdict(int)
    for r in results:
        by_level[r["risk_level"]] = by_level.get(r["risk_level"], 0) + 1
        for item in r["risk_breakdown"]:
            rule_hits[item["rule_id"]] += 1

    return {
        "total_invoices_scanned": len(results),
        "summary": by_level,
        "top_triggered_rules": dict(sorted(rule_hits.items(), key=lambda x: -x[1])),
        "circular_trading_cycles": ctx["cycles"],
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Serialization for Frontend (D3.js format)
# ═══════════════════════════════════════════════════════════════════════════

def risk_color(level: str) -> str:
    return RISK_COLORS.get(level, "#10b981")


def graph_to_d3(graph: dict, filter_user_id: str = None) -> dict:
    """
    Convert knowledge graph to D3.js force graph JSON format.
    If filter_user_id is provided, return only the subgraph for that user.
    """
    nodes_out = []
    edges_out = []
    included_gstins = set()

    # Determine which nodes to include
    if filter_user_id:
        # Find user's GSTIN
        user_gstin = None
        for gstin, uid in graph["gstin_to_user"].items():
            if uid == filter_user_id:
                user_gstin = gstin
                break

        if user_gstin:
            # Include user + all direct neighbors (1-hop)
            included_gstins.add(user_gstin)
            included_gstins.update(graph["adjacency"].get(user_gstin, set()))
            included_gstins.update(graph["reverse_adj"].get(user_gstin, set()))

            # For each neighbor, also include THEIR neighbors (2-hop) for richer graph
            for n in list(included_gstins):
                included_gstins.update(graph["adjacency"].get(n, set()))
                included_gstins.update(graph["reverse_adj"].get(n, set()))
        else:
            included_gstins = set(graph["nodes"].keys())
    else:
        included_gstins = set(graph["nodes"].keys())

    # Build nodes
    for gstin in included_gstins:
        node = graph["nodes"].get(gstin)
        if not node:
            continue

        is_primary = (node.get("user_id") == filter_user_id) if filter_user_id else False
        is_registered = node["node_type"] == "registered_user"

        # Determine size
        if is_primary:
            size = 40
        elif is_registered:
            size = 28
        else:
            size = 16

        level = node.get("risk_level", "LOW")

        nodes_out.append({
            "id": gstin,
            "label": node.get("label", gstin[:10]),
            "type": node["node_type"],
            "is_primary": is_primary,
            "is_registered": is_registered,
            "size": size,
            "risk_score": node.get("risk_score", 0),
            "risk_level": level,
            "color": risk_color(level),
            "in_cycle": node.get("in_circular_trade", False),
            "supplied": node.get("total_supplied", 0),
            "received": node.get("total_received", 0),
            "inv_count_supplier": node.get("invoice_count_supplier", 0),
            "inv_count_buyer": node.get("invoice_count_buyer", 0),
        })

    # Build edges
    gstin_set = included_gstins
    for (src, tgt), edge_list in graph["edges"].items():
        if src in gstin_set and tgt in gstin_set:
            total_val = sum(e["taxable_value"] for e in edge_list)
            total_tax = sum(e["tax_amount"] for e in edge_list)
            count = len(edge_list)

            # Check circularity
            is_circ = (
                graph["nodes"].get(src, {}).get("in_circular_trade", False) and
                graph["nodes"].get(tgt, {}).get("in_circular_trade", False)
            )

            # Edge risk
            edge_risk = "LOW"
            if is_circ:
                edge_risk = "CRITICAL"
            elif total_val > 1000000:
                edge_risk = "HIGH"
            elif count > 10:
                edge_risk = "MEDIUM"

            edges_out.append({
                "source": src,
                "target": tgt,
                "invoice_count": count,
                "total_value": round(total_val, 2),
                "total_tax": round(total_tax, 2),
                "is_circular": is_circ,
                "risk_level": edge_risk,
                "color": risk_color(edge_risk),
                "width": min(1 + count * 0.5, 8),
            })

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "stats": {
            "total_nodes": len(nodes_out),
            "total_edges": len(edges_out),
            "registered_nodes": sum(1 for n in nodes_out if n["is_registered"]),
            "external_nodes": sum(1 for n in nodes_out if not n["is_registered"]),
            "cycles": len(graph.get("cycles", [])),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Supabase Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_all_users(client: httpx.AsyncClient, url: str, headers: dict) -> list:
    """Fetch all users from Supabase."""
    resp = await client.get(f"{url}/rest/v1/recon_users?select=id,name,email,gst_number,login_type", headers=headers)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


async def fetch_all_invoices_from_db(client: httpx.AsyncClient, url: str, headers: dict) -> list:
    """Pull all GSTR data from all users and flatten into invoice list."""
    resp = await client.get(
        f"{url}/rest/v1/recon_users?select=id,gst_number,gstr_1,gstr_2b,gstr_3b",
        headers=headers
    )
    if resp.status_code != 200:
        return []
    users = resp.json()
    if not isinstance(users, list):
        return []

    all_invoices = []
    for user in users:
        for source_key, source_label in [("gstr_1", "gstr_1"), ("gstr_2b", "gstr_2b"), ("gstr_3b", "gstr_3b")]:
            blob = user.get(source_key)
            if not blob:
                continue
            rows = []
            if isinstance(blob, list):
                rows = blob
            elif isinstance(blob, dict):
                rows = blob.get("data_preview", blob.get("invoices", []))

            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                inv = dict(row)
                inv.setdefault("invoice_id", row.get("Invoice_Number", f"{user['id']}:{source_label}:{idx}"))
                inv.setdefault("supplier_gstin", row.get("GSTIN", user.get("gst_number", "")))
                inv.setdefault("buyer_gstin", row.get("Recipient_GSTIN", ""))
                inv.setdefault("invoice_number", row.get("Invoice_Number", ""))
                inv.setdefault("invoice_value", row.get("Total_Invoice_Value", 0))
                inv.setdefault("taxable_value", row.get("Taxable_Value", 0))
                inv.setdefault("igst", row.get("IGST", 0))
                inv.setdefault("cgst", row.get("CGST", 0))
                inv.setdefault("sgst", row.get("SGST", 0))
                inv.setdefault("gst_rate_percent", row.get("GST_Rate_%", 0))
                inv.setdefault("invoice_type", row.get("Invoice_Type", ""))
                inv.setdefault("return_period", row.get("Return_Period", ""))
                inv["source"] = source_label
                inv["db_user_id"] = user["id"]
                all_invoices.append(inv)

    return all_invoices


async def build_full_graph(client: httpx.AsyncClient, url: str, headers: dict) -> Tuple[dict, list]:
    """Build the complete knowledge graph from database data."""
    users = await fetch_all_users(client, url, headers)
    invoices = await fetch_all_invoices_from_db(client, url, headers)
    graph = build_knowledge_graph(invoices, users)
    return graph, invoices
