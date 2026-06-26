export const dynamic = 'force-dynamic';

/**
 * GET /api/journal
 *
 * Placeholder — journal data is currently stored entirely client-side
 * (localStorage / IndexedDB). This endpoint exists so the API surface is
 * consistent and ready for a backend migration.
 *
 * ── Future Supabase Integration ──────────────────────────────────────
 * When migrating to server-side persistence:
 *
 * 1. Install Supabase client:
 *    npm install @supabase/supabase-js
 *
 * 2. Create a Supabase client in `@/lib/supabase.js`:
 *    import { createClient } from '@supabase/supabase-js'
 *    export const supabase = createClient(
 *      process.env.NEXT_PUBLIC_SUPABASE_URL,
 *      process.env.SUPABASE_SERVICE_ROLE_KEY  // server-side only
 *    )
 *
 * 3. Define a `journal_entries` table in Supabase:
 *    - id          uuid        primary key default gen_random_uuid()
 *    - user_id     text        not null
 *    - content     text        not null
 *    - mood        text
 *    - tags        text[]
 *    - created_at  timestamptz default now()
 *    - updated_at  timestamptz default now()
 *
 * 4. Update GET to query:
 *    const { data, error } = await supabase
 *      .from('journal_entries')
 *      .select('*')
 *      .eq('user_id', userId)
 *      .order('created_at', { ascending: false })
 *
 * 5. Update POST to insert:
 *    const { data, error } = await supabase
 *      .from('journal_entries')
 *      .insert({ user_id: userId, content, mood, tags })
 *      .select()
 *      .single()
 *
 * 6. Add authentication middleware to extract userId from session/JWT.
 * ─────────────────────────────────────────────────────────────────────
 */
export async function GET() {
  return Response.json({
    message:
      'Journal data is stored client-side. ' +
      'This endpoint is a placeholder for future server-side persistence.',
    entries: [],
  });
}

/**
 * POST /api/journal
 *
 * Placeholder — accepts a journal entry payload but currently does not
 * persist it server-side. See the Supabase integration notes above.
 *
 * Expected future body shape:
 * {
 *   content: string,      // the journal text
 *   mood?: string,         // e.g. 'happy', 'sad', 'anxious'
 *   tags?: string[],       // e.g. ['school', 'friends']
 * }
 */
export async function POST() {
  return Response.json({
    message:
      'Journal entry saved client-side. ' +
      'Server-side persistence coming with Supabase integration.',
    success: true,
  });
}
