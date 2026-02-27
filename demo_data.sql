-- Demo data for recon_users table
-- This adds a few sample users with different profiles

-- Note: Passwords here are hashed using the local backend logic (SHA-256)
-- 'demo123' -> 3624e75d496a70e70a3c983a54a0a7d512a259c687e6005d59045a953e5e4922:c4e0f1d2 (Example hash:salt)

INSERT INTO public.recon_users (
    id,
    name, 
    email, 
    password, 
    login_type, 
    gst_number, 
    gstr_1, 
    gstr_2b, 
    gstr_3b, 
    json_data
) VALUES 
(
    '2024-02-27-09-00-00-01',
    'Admin User', 
    'admin@demo.com', 
    '3624e75d496a70e70a3c983a54a0a7d512a259c687e6005d59045a953e5e4922:c4e0f1d2', -- demo123
    'admin', 
    NULL, 
    NULL, 
    NULL, 
    NULL, 
    '{"access": "full", "department": "HQ"}'
),
(
    '2024-02-27-09-30-00-42',
    'Ajay Enterprises', 
    'ajay@example.com', 
    '3624e75d496a70e70a3c983a54a0a7d512a259c687e6005d59045a953e5e4922:c4e0f1d2', -- demo123
    'user', 
    '27AAAAA0000A1Z5', 
    '{"total_sales": 500000, "invoices": 12}', 
    '{"itc_available": 45000}', 
    '{"tax_paid": 5000}', 
    '{"segment": "Retail", "region": "Maharashtra"}'
),
(
    '2024-02-27-10-15-00-88',
    'Global Traders', 
    'global@test.com', 
    '3624e75d496a70e70a3c983a54a0a7d512a259c687e6005d59045a953e5e4922:c4e0f1d2', -- demo123
    'user', 
    '27BBBBB1111B1Z2', 
    '{"total_sales": 1200000, "invoices": 45}', 
    '{"itc_available": 110000}', 
    '{"tax_paid": 12000}', 
    '{"segment": "Wholesale", "region": "Karnataka"}'
);
