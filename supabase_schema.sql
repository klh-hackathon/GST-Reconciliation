-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create the custom users table
CREATE TABLE public.recon_users (
    id TEXT PRIMARY KEY, -- Using custom format: year-month-date-time-random
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL, -- Note: Supabase handles passwords via auth.users securely. Only use this if building a completely custom login flow.
    login_type TEXT NOT NULL CHECK (login_type IN ('user', 'admin')),
    gst_number TEXT, -- Only required if login_type is 'user'
    
    -- GST Return data (Storing as JSONB is recommended for flexibility)
    gstr_1 JSONB,
    gstr_2b JSONB,
    gstr_3b JSONB,
    e_invoice JSONB,
    
    -- blob storage for images (bytea in PostgreSQL)
    -- Note: Supabase Storage Buckets are much better for images, but here is the BLOB column as requested.
    e_way_bill BYTEA, 
    profile_pic BYTEA,
    
    -- Misc JSON data
    json_data JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
);

-- Optional: Add a trigger to enforce gst_number if user is 'user'
-- CREATE OR REPLACE FUNCTION check_user_gst()
-- RETURNS TRIGGER AS $$
-- BEGIN
--     IF NEW.login_type = 'user' AND NEW.gst_number IS NULL THEN
--         RAISE EXCEPTION 'gst_number is required for standard users';
--     END IF;
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- CREATE TRIGGER enforce_gst_number
-- BEFORE INSERT OR UPDATE ON public.recon_users
-- FOR EACH ROW EXECUTE FUNCTION check_user_gst();
