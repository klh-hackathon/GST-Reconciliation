import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="GST Recon Engine (Lite)",
    description="A demonstration version of the GST Reconciliation Engine API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/auth/login")
async def login(data: dict):
    return {"access_token": "ok"}

@app.get("/api/dashboard/stats")
async def get_stats():
    return {
        "totalInvoices": 1250, 
        "totalTax": 4500000, 
        "mismatchCount": 42, 
        "itcAtRisk": 125000,
        "complianceScore": 94,
        "activeVendors": 128
    }

@app.get("/api/dashboard/recent-activity")
async def get_activity():
    return [
        {"time": "10 min ago", "event": "GSTR-2B Auto-sync complete", "status": "success"},
        {"time": "1 hour ago", "event": "Mismatch detected: Vendor #27AAAA...", "status": "warning"},
        {"time": "3 hours ago", "event": "Bulk report generated for FY23-24", "status": "info"}
    ]

@app.get("/api/dashboard/vendor-risks")
async def get_vendor_risks():
    return [
        {"gstin": "27AAAAA0000A1Z5", "totalInvoices": 150, "mismatchedInvoices": 12, "riskScore": 75, "lastFiled": "2024-02-15"},
        {"gstin": "27DDDDD3333D1Z4", "totalInvoices": 45, "mismatchedInvoices": 15, "riskScore": 90, "lastFiled": "2024-01-10"},
        {"gstin": "27BBBBB1111B1Z2", "totalInvoices": 85, "mismatchedInvoices": 8, "riskScore": 45, "lastFiled": "2024-02-20"},
        {"gstin": "27CCCCC2222C1Z9", "totalInvoices": 200, "mismatchedInvoices": 5, "riskScore": 15, "lastFiled": "2024-02-18"}
    ]

@app.get("/api/analytics/trends")
async def get_trends():
    return {
        "labels": ["Oct", "Nov", "Dec", "Jan", "Feb"], 
        "values": [15, 22, 18, 25, 30]
    }

@app.get("/api/settings")
async def get_settings():
    return {
        "tenant": "Demo Corp", 
        "region": "Maharashtra", 
        "autoScan": True
    }

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
