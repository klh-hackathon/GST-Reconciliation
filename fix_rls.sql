-- Run this in your Supabase SQL Editor (Dashboard > SQL Editor > New Query)
-- This allows the backend to read/write the recon_users table via the anon key.

-- Option 1: Disable RLS entirely (simplest for a demo project)
ALTER TABLE public.recon_users DISABLE ROW LEVEL SECURITY;

-- OR Option 2: Keep RLS enabled but add open policies (more secure)
-- CREATE POLICY "Allow inserts" ON public.recon_users FOR INSERT WITH CHECK (true);
-- CREATE POLICY "Allow selects" ON public.recon_users FOR SELECT USING (true);
