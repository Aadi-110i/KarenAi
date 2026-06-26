// Karen AI — Safety Filtering System
// Crisis detection, content filtering, and age-appropriateness checks

/**
 * Crisis detection patterns organized by type.
 * Each entry has a type label, regex patterns, and tailored resources.
 */
const CRISIS_PATTERNS = [
  {
    type: 'suicide',
    patterns: [
      /\bsuicid(e|al)\b/i,
      /\bkill\s+my\s*self\b/i,
      /\bwant\s+to\s+die\b/i,
      /\bend\s+my\s+life\b/i,
      /\bdon'?t\s+want\s+to\s+live\b/i,
      /\bno\s+reason\s+to\s+live\b/i,
      /\btake\s+my\s+(own\s+)?life\b/i,
      /\bwish\s+i\s+(was|were)\s+dead\b/i,
      /\bbetter\s+off\s+dead\b/i,
    ],
  },
  {
    type: 'self-harm',
    patterns: [
      /\bcutting\b/i,
      /\bself[- ]?harm\b/i,
      /\bhurting\s+my\s*self\b/i,
      /\bharming\s+my\s*self\b/i,
      /\bburn(ing)?\s+my\s*self\b/i,
      /\bhit(ting)?\s+my\s*self\b/i,
      /\bscratch(ing)?\s+my\s*self\b/i,
    ],
  },
  {
    type: 'abuse',
    patterns: [
      /\bsomeone\s+is\s+hurting\s+me\b/i,
      /\bbeing\s+abused\b/i,
      /\btouched\s+inappropriately\b/i,
      /\bmolest(ed|ation)?\b/i,
      /\bbeing\s+hit\b/i,
      /\bsexually\s+abused\b/i,
      /\bphysically\s+abused\b/i,
      /\bemotionally\s+abused\b/i,
      /\bdomestic\s+violence\b/i,
      /\bmy\s+(parent|dad|mom|step[- ]?(dad|mom|father|mother)|guardian)\s+(hits?|beats?|hurts?)\s+me\b/i,
    ],
  },
  {
    type: 'eating-disorder',
    patterns: [
      /\bstarving\s+my\s*self\b/i,
      /\bmaking\s+my\s*self\s+throw\s+up\b/i,
      /\bpurging\b/i,
      /\banorexi(a|c)\b/i,
      /\bbulimi(a|c)\b/i,
      /\bnot\s+eating\s+on\s+purpose\b/i,
      /\bbinge\s+(eat|and\s+purge)\b/i,
    ],
  },
  {
    type: 'unsafe-home',
    patterns: [
      /\bfeel\s+unsafe\s+at\s+home\b/i,
      /\bscared\s+of\s+my\s+(parent|dad|mom|step[- ]?(dad|mom|father|mother)|guardian)\b/i,
      /\bnobody\s+to\s+talk\s+to\b/i,
      /\bfeel\s+alone\s+and\s+scared\b/i,
      /\bafraid\s+to\s+go\s+home\b/i,
      /\bnowhere\s+(safe|to\s+go)\b/i,
      /\brunning\s+away\s+from\s+home\b/i,
    ],
  },
];

/**
 * Patterns that flag unsafe content in AI responses.
 */
const UNSAFE_RESPONSE_PATTERNS = [
  {
    type: 'explicit-sexual',
    patterns: [
      /\bsexual\s+intercourse\s+in\s+detail\b/i,
      /\bexplicit\s+sexual\b/i,
      /\bpornograph(y|ic)\b/i,
      /\borgas(m|mic)\s+(technique|how\s+to)\b/i,
      /\bsexual\s+position/i,
      /\bmasturbat(e|ion)\s+(technique|how\s+to|step)/i,
    ],
    replacement:
      "I want to make sure I'm giving you helpful, age-appropriate information. If you have questions about your body or relationships, I'm happy to discuss them in a way that's informative and respectful. 💙",
  },
  {
    type: 'graphic-violence',
    patterns: [
      /\bgraphic(ally)?\s+(descri(be|ption)|detail)\s+(of\s+)?(violence|injury|blood|gore)\b/i,
      /\btorture\s+(method|technique|how\s+to)\b/i,
      /\bhow\s+to\s+(hurt|injure|harm)\s+(someone|a\s+person|people)\b/i,
    ],
    replacement:
      "I'm not able to go into graphic details about violence. If you're feeling unsafe or experiencing violence, please reach out to a trusted adult or call the Childhelp Hotline at 1-800-422-4453. You deserve to be safe. 💙",
  },
  {
    type: 'encouraging-self-harm',
    patterns: [
      /\byou\s+should\s+(cut|harm|hurt|starve)\s+(yourself|your\s+body)\b/i,
      /\btry\s+(cutting|harming|hurting)\b/i,
      /\bit'?s?\s+okay\s+to\s+(cut|harm|hurt)\s+(yourself|your\s+body)\b/i,
      /\bhere'?s?\s+how\s+to\s+(cut|harm|hurt|starve)\b/i,
    ],
    replacement:
      "I care about your safety. If you're going through a tough time, please reach out to the Crisis Text Line — text HOME to 741741. You're not alone, and there are people who want to help. 💙",
  },
  {
    type: 'medical-diagnosis',
    patterns: [
      /\byou\s+(have|probably\s+have|likely\s+have|might\s+have|definitely\s+have)\s+(ADHD|depression|anxiety\s+disorder|bipolar|OCD|autism|anorexia|bulimia|PTSD)\b/i,
      /\bI('m| am)\s+diagnosing\s+you\b/i,
      /\byour\s+diagnosis\s+is\b/i,
      /\byou\s+are\s+(clinically\s+)?(depressed|anxious|bipolar|autistic|anorexic|bulimic)\b/i,
    ],
    replacement:
      "I'm not a doctor, so I can't make any diagnoses. But what you're describing sounds worth talking about with a trusted adult or a healthcare professional. They can give you the right support and answers. You're smart for paying attention to how you feel! 💙",
  },
  {
    type: 'undermining-parents',
    patterns: [
      /\byour\s+parents?\s+(are|is)\s+(wrong|stupid|dumb|idiots?|clueless)\b/i,
      /\bdon'?t\s+(listen\s+to|tell)\s+your\s+parents?\b/i,
      /\bhide\s+(this|it)\s+from\s+your\s+parents?\b/i,
      /\byour\s+parents?\s+(don'?t|doesn'?t)\s+(need\s+to|have\s+to)\s+know\b/i,
    ],
    replacement:
      "I always want to encourage open communication with your parents or guardians. Even when things feel tough, talking to them (or another trusted adult) can really help. They care about you, even when it doesn't feel like it. 💙",
  },
];

/**
 * Age-inappropriate content patterns.
 */
const AGE_INAPPROPRIATE_PATTERNS = {
  '9-12': [
    {
      pattern: /\b(sex(ual)?\s+(intercourse|positions?|acts?)|orgasm|masturbat(e|ion)|erection|ejaculat(e|ion))\b/i,
      reason: 'Contains sexual content not appropriate for ages 9-12',
    },
    {
      pattern: /\b(drug\s+use|getting\s+(high|drunk|wasted)|alcohol\s+consumption|smoking\s+weed|vaping)\b/i,
      reason: 'Contains substance use content not appropriate for ages 9-12',
    },
    {
      pattern: /\b(dating\s+(tips|advice|strategies)|how\s+to\s+(kiss|make\s+out)|romantic\s+relationship\s+advice)\b/i,
      reason: 'Contains romantic/dating content not age-appropriate for ages 9-12',
    },
  ],
  '13-16': [
    {
      pattern: /\b(explicit\s+sexual\s+(content|description|detail)|pornograph(y|ic)|sex\s+positions?|how\s+to\s+have\s+sex)\b/i,
      reason: 'Contains explicit sexual content not appropriate for minors',
    },
    {
      pattern: /\b(how\s+to\s+(buy|get|use)\s+drugs?|recreational\s+drug\s+use|how\s+to\s+get\s+(high|drunk|wasted))\b/i,
      reason: 'Contains substance promotion not appropriate for minors',
    },
  ],
};

/**
 * Scans an AI response for unsafe content and replaces flagged sections.
 * @param {string} text - The AI-generated response text
 * @returns {{ filtered: string, wasFiltered: boolean }}
 */
export function filterResponse(text) {
  let filtered = text;
  let wasFiltered = false;

  for (const rule of UNSAFE_RESPONSE_PATTERNS) {
    for (const pattern of rule.patterns) {
      if (pattern.test(filtered)) {
        wasFiltered = true;
        // Replace the entire response if it contains unsafe content,
        // as partial replacement could leave context that's still harmful.
        filtered = rule.replacement;
        return { filtered, wasFiltered };
      }
    }
  }

  return { filtered, wasFiltered };
}

/**
 * Detects crisis signals in user input text.
 * @param {string} text - The user's message text
 * @returns {{ isCrisis: boolean, type: string | null, resources: Array<{ name: string, contact: string }> }}
 */
export function detectCrisis(text) {
  if (!text || typeof text !== 'string') {
    return { isCrisis: false, type: null, resources: [] };
  }

  for (const crisisType of CRISIS_PATTERNS) {
    for (const pattern of crisisType.patterns) {
      if (pattern.test(text)) {
        return {
          isCrisis: true,
          type: crisisType.type,
          resources: getResourcesForCrisisType(crisisType.type),
        };
      }
    }
  }

  return { isCrisis: false, type: null, resources: [] };
}

/**
 * Checks if content is appropriate for the given age group.
 * @param {string} text - The content to check
 * @param {'9-12' | '13-16'} ageGroup - The age group to check against
 * @returns {{ isAppropriate: boolean, reason: string | null }}
 */
export function isAgeAppropriate(text, ageGroup) {
  if (!text || typeof text !== 'string') {
    return { isAppropriate: true, reason: null };
  }

  const patterns = AGE_INAPPROPRIATE_PATTERNS[ageGroup];
  if (!patterns) {
    return { isAppropriate: true, reason: null };
  }

  for (const { pattern, reason } of patterns) {
    if (pattern.test(text)) {
      return { isAppropriate: false, reason };
    }
  }

  return { isAppropriate: true, reason: null };
}

/**
 * Returns the full list of crisis helpline resources.
 * @returns {Array<{ name: string, contact: string, description: string }>}
 */
export function getHelplineResources() {
  return [
    {
      name: 'Crisis Text Line',
      contact: 'Text HOME to 741741',
      description:
        'Free, 24/7 crisis counseling via text message. Trained crisis counselors are available to help with any type of crisis.',
    },
    {
      name: '988 Suicide & Crisis Lifeline',
      contact: 'Call or text 988',
      description:
        'Free, confidential 24/7 support for people in suicidal crisis or emotional distress. Available in English and Spanish.',
    },
    {
      name: 'Childhelp National Child Abuse Hotline',
      contact: '1-800-422-4453',
      description:
        'Professional crisis counselors available 24/7 to help with child abuse situations. Available in over 170 languages.',
    },
    {
      name: 'SAMHSA National Helpline',
      contact: '1-800-662-4357',
      description:
        'Free, confidential, 24/7 treatment referral and information service for substance abuse and mental health disorders.',
    },
    {
      name: 'Trevor Project',
      contact: '1-866-488-7386',
      description:
        'Crisis intervention and suicide prevention for LGBTQ+ young people. Also available via text (text START to 678-678) and online chat.',
    },
  ];
}

/**
 * Sanitizes user input to prevent injection and remove potentially harmful content.
 * @param {string} text - The raw user input
 * @returns {string} - The sanitized text
 */
export function sanitizeInput(text) {
  if (!text || typeof text !== 'string') {
    return '';
  }

  let sanitized = text;

  // Remove HTML tags to prevent injection
  sanitized = sanitized.replace(/<[^>]*>/g, '');

  // Remove potential script injections
  sanitized = sanitized.replace(/javascript:/gi, '');
  sanitized = sanitized.replace(/on\w+\s*=/gi, '');

  // Remove excessive whitespace but preserve single newlines for readability
  sanitized = sanitized.replace(/[ \t]+/g, ' ');
  sanitized = sanitized.replace(/\n{3,}/g, '\n\n');

  // Trim leading and trailing whitespace
  sanitized = sanitized.trim();

  // Enforce a reasonable maximum length (2000 chars)
  if (sanitized.length > 2000) {
    sanitized = sanitized.slice(0, 2000);
  }

  return sanitized;
}

// ── Internal Helpers ──────────────────────────────────────────────

/**
 * Returns relevant helpline resources based on crisis type.
 * @param {string} crisisType
 * @returns {Array<{ name: string, contact: string }>}
 */
function getResourcesForCrisisType(crisisType) {
  const allResources = getHelplineResources();

  switch (crisisType) {
    case 'suicide':
      return allResources.filter((r) =>
        ['988 Suicide & Crisis Lifeline', 'Crisis Text Line', 'Trevor Project'].includes(r.name)
      );
    case 'self-harm':
      return allResources.filter((r) =>
        ['Crisis Text Line', '988 Suicide & Crisis Lifeline'].includes(r.name)
      );
    case 'abuse':
      return allResources.filter((r) =>
        ['Childhelp National Child Abuse Hotline', 'Crisis Text Line', '988 Suicide & Crisis Lifeline'].includes(r.name)
      );
    case 'eating-disorder':
      return allResources.filter((r) =>
        ['Crisis Text Line', '988 Suicide & Crisis Lifeline', 'SAMHSA National Helpline'].includes(r.name)
      );
    case 'unsafe-home':
      return allResources.filter((r) =>
        ['Childhelp National Child Abuse Hotline', 'Crisis Text Line', '988 Suicide & Crisis Lifeline'].includes(r.name)
      );
    default:
      return allResources;
  }
}
