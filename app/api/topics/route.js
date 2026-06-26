import {
  topics,
  getTopicBySlug,
  getTopicsByCategory,
  getTopicsForProfile,
  categories,
} from '@/lib/topics-data';

export const dynamic = 'force-dynamic';

/**
 * GET /api/topics
 *
 * Query params:
 *   ?slug=X         → return full detail for a single topic
 *   ?category=X     → filter topics by category
 *   ?ageGroup=X     → filter by age group (used with getTopicsForProfile)
 *   ?gender=X       → filter by gender (used with getTopicsForProfile)
 *   (none)          → return metadata for all topics
 */
export async function GET(request) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const slug = searchParams.get('slug');
    const category = searchParams.get('category');
    const ageGroup = searchParams.get('ageGroup');
    const gender = searchParams.get('gender');

    // ── Single topic by slug ──────────────────────────────────────────
    if (slug) {
      const topic = getTopicBySlug(slug);
      if (!topic) {
        return Response.json(
          { error: `Topic not found: "${slug}"` },
          { status: 404 }
        );
      }
      return Response.json({ topic });
    }

    // ── Profile-based filtering ───────────────────────────────────────
    if (ageGroup || gender) {
      const profile = {};
      if (ageGroup) profile.ageGroup = ageGroup;
      if (gender) profile.gender = gender;

      const filtered = getTopicsForProfile(profile);

      // Return metadata only
      const metadata = filtered.map(({ slug, title, icon, category, ageGroups, genderRelevance }) => ({
        slug,
        title,
        icon,
        category,
        ageGroups,
        genderRelevance,
      }));

      return Response.json({ topics: metadata, total: metadata.length });
    }

    // ── Filter by category ────────────────────────────────────────────
    if (category) {
      const filtered = getTopicsByCategory(category);
      if (!filtered || filtered.length === 0) {
        return Response.json(
          {
            error: `No topics found for category: "${category}"`,
            availableCategories: categories,
          },
          { status: 404 }
        );
      }

      const metadata = filtered.map(({ slug, title, icon, category, ageGroups, genderRelevance }) => ({
        slug,
        title,
        icon,
        category,
        ageGroups,
        genderRelevance,
      }));

      return Response.json({ topics: metadata, total: metadata.length });
    }

    // ── All topics (metadata only) ────────────────────────────────────
    const allMetadata = topics.map(({ slug, title, icon, category, ageGroups, genderRelevance }) => ({
      slug,
      title,
      icon,
      category,
      ageGroups,
      genderRelevance,
    }));

    return Response.json({
      topics: allMetadata,
      total: allMetadata.length,
      categories,
    });
  } catch (error) {
    console.error('[Karen AI] Topics route error:', error);
    return Response.json(
      {
        error: 'Failed to fetch topics.',
        details:
          process.env.NODE_ENV === 'development' ? error.message : undefined,
      },
      { status: 500 }
    );
  }
}
