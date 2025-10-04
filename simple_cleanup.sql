-- Simple SQL cleanup script for dummy data
-- Run this in Supabase SQL Editor for immediate cleanup

-- Delete in proper order to respect foreign key constraints

-- 1. Delete dependent records first
DELETE FROM audit_logs WHERE description ILIKE '%Cebu%';
DELETE FROM medicine_prescriptions WHERE instructions ILIKE '%Cebu%';
DELETE FROM medicine_stock_movements WHERE notes ILIKE '%Cebu%';
DELETE FROM appointments WHERE notes ILIKE '%Cebu%';
DELETE FROM medical_records WHERE management ILIKE '%Cebu%';
DELETE FROM user_certifications WHERE issued_by ILIKE '%Cebu%';
DELETE FROM user_biography WHERE address ILIKE '%Cebu%';

-- 2. Delete main records
DELETE FROM medicine_inventory WHERE batch_number ILIKE 'BATCHCB%';
DELETE FROM patients WHERE address ILIKE '%Cebu%';
DELETE FROM doctor_availability WHERE doctor_id IN (
    SELECT id FROM users WHERE email ILIKE '%@cebu-medical.ph'
);

-- 3. Delete core records
DELETE FROM medicines WHERE manufacturer = 'Unilab Philippines Inc.';
DELETE FROM medicine_suppliers WHERE name ILIKE 'Cebu%';

-- 4. Finally delete users
DELETE FROM users WHERE email ILIKE '%@cebu-medical.ph';

-- Verify cleanup
SELECT 'audit_logs' as table_name, COUNT(*) as remaining FROM audit_logs WHERE description ILIKE '%Cebu%'
UNION ALL
SELECT 'medicine_prescriptions', COUNT(*) FROM medicine_prescriptions WHERE instructions ILIKE '%Cebu%'
UNION ALL
SELECT 'medicine_stock_movements', COUNT(*) FROM medicine_stock_movements WHERE notes ILIKE '%Cebu%'
UNION ALL
SELECT 'appointments', COUNT(*) FROM appointments WHERE notes ILIKE '%Cebu%'
UNION ALL
SELECT 'medical_records', COUNT(*) FROM medical_records WHERE management ILIKE '%Cebu%'
UNION ALL
SELECT 'user_certifications', COUNT(*) FROM user_certifications WHERE issued_by ILIKE '%Cebu%'
UNION ALL
SELECT 'user_biography', COUNT(*) FROM user_biography WHERE address ILIKE '%Cebu%'
UNION ALL
SELECT 'medicine_inventory', COUNT(*) FROM medicine_inventory WHERE batch_number ILIKE 'BATCHCB%'
UNION ALL
SELECT 'patients', COUNT(*) FROM patients WHERE address ILIKE '%Cebu%'
UNION ALL
SELECT 'medicines', COUNT(*) FROM medicines WHERE manufacturer = 'Unilab Philippines Inc.'
UNION ALL
SELECT 'medicine_suppliers', COUNT(*) FROM medicine_suppliers WHERE name ILIKE 'Cebu%'
UNION ALL
SELECT 'users', COUNT(*) FROM users WHERE email ILIKE '%@cebu-medical.ph';
