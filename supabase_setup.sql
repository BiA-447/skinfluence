-- =====================================================================
-- STEP A: create the products table  (Supabase -> SQL Editor -> New query)
-- =====================================================================
create table if not exists public.products (
  id                    bigint generated always as identity primary key,
  name                  text not null,
  brand                 text,
  category              text,
  product_type          text,
  origin                text,                 -- 'Pakistan' or 'Imported'
  price                 numeric,
  currency              text default 'PKR',
  description           text,
  skin_types            text[] default '{}',
  concerns              text[] default '{}',
  goals                 text[] default '{}',
  key_ingredients       text[] default '{}',
  sensitivity_level     text,
  prescription_required boolean default false,
  is_active             boolean default true,
  last_verified         date,
  seller                text,                 -- NEW: shop name
  purchase_url          text,                 -- NEW: buy link
  updated_at            timestamptz default now()
);

-- Public READ-ONLY access for the app (the anon key can read, never write).
alter table public.products enable row level security;
drop policy if exists "public read products" on public.products;
create policy "public read products" on public.products for select using (true);

-- =====================================================================
-- STEP B: AFTER importing products.csv into a table called products_staging
-- (Table Editor -> New table -> Import data from CSV), run this to copy it in:
-- =====================================================================
insert into public.products
  (name, brand, category, product_type, origin, price, currency, description,
   skin_types, concerns, goals, key_ingredients, sensitivity_level,
   prescription_required, is_active, last_verified)
select
  name, brand, category, product_type, origin, price::numeric, currency, description,
  skin_types::text[], concerns::text[], goals::text[], key_ingredients::text[],
  sensitivity_level, prescription_required::boolean, is_active::boolean, last_verified::date
from public.products_staging;

-- Optional: delete the staging table once you checked the data
-- drop table public.products_staging;

-- =====================================================================
-- STEP C: fix the prescription flag (medicine-type products)
-- =====================================================================
update public.products
set prescription_required = true
where array_to_string(key_ingredients, ' ') ~* 'tretinoin|clindamycin|isotretinoin|hydroquinone|clobetasol|betamethasone|mometasone';

-- =====================================================================
-- Everyday editing examples (or just use the Table Editor):
--   update public.products set purchase_url = 'https://...', seller = 'Daraz' where name = 'CeraVe Foaming Cleanser';
--   update public.products set price = 3200, last_verified = current_date where id = 12;
--   update public.products set is_active = false where id = 7;   -- hides it from the app
-- =====================================================================
