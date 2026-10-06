-- Run this ONCE in Supabase:  SQL Editor -> New query -> paste -> Run
-- It lets the app READ these 4 tables with the public (publishable / anon) key.
-- It does NOT allow anyone to change data with that key.

alter table public.products        enable row level security;
alter table public.product_sources enable row level security;
alter table public.retailers       enable row level security;
alter table public.reddit_insights enable row level security;

drop policy if exists "app read products"        on public.products;
drop policy if exists "app read product_sources" on public.product_sources;
drop policy if exists "app read retailers"       on public.retailers;
drop policy if exists "app read reddit_insights" on public.reddit_insights;

create policy "app read products"        on public.products        for select using (true);
create policy "app read product_sources" on public.product_sources for select using (true);
create policy "app read retailers"       on public.retailers       for select using (true);
create policy "app read reddit_insights" on public.reddit_insights for select using (true);

-- ---------- EXAMPLE: add a Buy link for one product ----------
-- 1) add the shop once:
-- insert into public.retailers (name, website, retailer_type, is_verified, country)
-- values ('Example Pharmacy', 'https://example.pk', 'pharmacy', true, 'Pakistan');
-- 2) link the product to the shop (find the ids in the Table Editor):
-- insert into public.product_sources (product_id, retailer_id, url, price, is_verified, is_available, last_checked)
-- values ('<product id>', '<retailer id>', 'https://example.pk/product-page', 1850, true, true, current_date);
