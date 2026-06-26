import { getSystemPrompt, formatMessages } from '@/lib/karen-ai';
import {
  sanitizeInput,
  detectCrisis,
  filterResponse,
  getHelplineResources,
} from '@/lib/safety';
import { searchTopics, topics } from '@/lib/topics-data';

export const dynamic = 'force-dynamic';

/**
 * Helper: convert an async iterator into a ReadableStream
 * Pattern recommended by Next.js 16 streaming docs.
 */
function iteratorToStream(iterator) {
  return new ReadableStream({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) {
        controller.close();
      } else {
        controller.enqueue(value);
      }
    },
  });
}

/**
 * Helper: tiny sleep for word-by-word streaming in fallback mode
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Fallback: scripted topic-based response ────────────────────────────────

/**
 * Extract simple keywords from a user message for topic matching.
 */
function extractKeywords(message) {
  const stopWords = new Set([
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its',
    'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'can', 'may', 'might',
    'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'about', 'like', 'through', 'after',
    'over', 'between', 'out', 'up', 'down', 'and', 'but', 'or',
    'nor', 'not', 'so', 'if', 'then', 'than', 'too', 'very',
    'just', 'that', 'this', 'what', 'how', 'when', 'where', 'why',
    'who', 'which', 'there', 'here', 'all', 'each', 'some', 'any',
    'no', 'know', 'don', 'want', 'feel', 'think', 'really',
    'get', 'got', 'going', 'tell', 'need', 'help',
  ]);

  return message
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !stopWords.has(w));
}

/**
 * Build a scripted response from matched topic data.
 */
function buildFallbackResponse(userMessage) {
  const keywords = extractKeywords(userMessage);

  // Try searchTopics first (the lib's own search)
  const results = searchTopics(userMessage);

  if (results && results.length > 0) {
    const topic = results[0];
    const title = topic.title || topic.slug || 'this topic';
    const intro = topic.introduction || topic.description || '';
    const keyPoints =
      topic.sections
        ?.slice(0, 2)
        .map((s) => s.title || s.heading)
        .filter(Boolean) || [];

    let response = `That's a really important question about **${title}**. `;
    if (intro) {
      response += `${intro} `;
    }
    if (keyPoints.length > 0) {
      response += `\n\nHere are some things worth knowing:\n`;
      keyPoints.forEach((point) => {
        response += `\n• **${point}**`;
      });
    }
    response +=
      `\n\nWould you like me to go deeper into any part of this? ` +
      `I'm here to help you understand at your own pace. 💙`;
    return response;
  }

  // Generic supportive response when no topic matches
  const genericResponses = [
    `That's a great question, and I'm glad you felt comfortable asking! While I don't have specific information on that exact topic right now, I want you to know that curiosity is completely normal and healthy. Would you like to explore some of the topics I do know about? I can help with things like puberty, emotions, relationships, and staying safe online. 💙`,
    `I appreciate you reaching out about that! It sounds like something that's on your mind, and that's totally okay. I might not have a detailed answer for that specific question, but I'd love to help if you'd like to ask about growing up, feelings, friendships, or body changes. What sounds interesting? 💙`,
    `Thanks for asking — it shows real maturity! I may not have the perfect answer for that one, but I'm here to chat about all sorts of things related to growing up. Want to try asking about something like puberty, emotions, or staying safe? I'm all ears! 💙`,
  ];

  return genericResponses[Math.floor(Math.random() * genericResponses.length)];
}

// ─── Provider: Ollama streaming ─────────────────────────────────────────────

async function* streamOllama(systemPrompt, formattedMessages) {
  const baseUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
  const model = process.env.OLLAMA_MODEL || 'llama3.1';

  const ollamaMessages = [
    { role: 'system', content: systemPrompt },
    ...formattedMessages,
  ];

  const res = await fetch(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages: ollamaMessages, stream: true }),
  });

  if (!res.ok) {
    throw new Error(`Ollama API error: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Ollama sends NDJSON — one JSON object per line
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const json = JSON.parse(trimmed);
        if (json.message?.content) {
          yield json.message.content;
        }
      } catch {
        // skip malformed lines
      }
    }
  }

  // flush remaining buffer
  if (buffer.trim()) {
    try {
      const json = JSON.parse(buffer.trim());
      if (json.message?.content) {
        yield json.message.content;
      }
    } catch {
      // skip
    }
  }
}

// ─── Provider: OpenAI-compatible streaming (SSE) ────────────────────────────

async function* streamOpenAI(systemPrompt, formattedMessages) {
  const baseUrl =
    process.env.OPENAI_API_URL || 'https://api.openai.com/v1';
  const apiKey = process.env.OPENAI_API_KEY;
  const model = process.env.OPENAI_MODEL || 'gpt-4o-mini';

  if (!apiKey || apiKey === 'your-key-here') {
    throw new Error(
      'OPENAI_API_KEY is not configured. Please set it in .env.local'
    );
  }

  const messages = [
    { role: 'system', content: systemPrompt },
    ...formattedMessages,
  ];

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model, messages, stream: true }),
  });

  if (!res.ok) {
    const errorBody = await res.text().catch(() => '');
    throw new Error(
      `OpenAI API error: ${res.status} ${res.statusText} — ${errorBody}`
    );
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE format: "data: {...}\n\n"
    const parts = buffer.split('\n');
    buffer = parts.pop();

    for (const part of parts) {
      const trimmed = part.trim();
      if (!trimmed || trimmed === 'data: [DONE]') continue;
      if (!trimmed.startsWith('data: ')) continue;

      try {
        const json = JSON.parse(trimmed.slice(6));
        const content = json.choices?.[0]?.delta?.content;
        if (content) {
          yield content;
        }
      } catch {
        // skip malformed SSE
      }
    }
  }
}

// ─── Provider: Fallback word-by-word streaming ──────────────────────────────

async function* streamFallback(userMessage) {
  const response = buildFallbackResponse(userMessage);
  const words = response.split(/(\s+)/); // preserve whitespace

  for (const word of words) {
    yield word;
    await sleep(30); // simulate natural typing
  }
}

// ─── POST handler ───────────────────────────────────────────────────────────

export async function POST(request) {
  try {
    const body = await request.json();
    const { messages, profile } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return Response.json(
        { error: 'Messages array is required and must not be empty.' },
        { status: 400 }
      );
    }

    // Get the latest user message
    const lastMessage = messages[messages.length - 1];
    const userContent =
      typeof lastMessage === 'string'
        ? lastMessage
        : lastMessage?.content || '';

    // 1. Sanitize input
    const sanitizedContent = sanitizeInput(userContent);

    // 2. Crisis detection — respond immediately without calling AI
    const crisisResult = detectCrisis(sanitizedContent);
    if (crisisResult?.isCrisis) {
      const resources = getHelplineResources();
      const crisisMessage =
        `I can see you might be going through something really difficult right now, ` +
        `and I want you to know that you're not alone. **Please reach out to someone who can help:**\n\n` +
        resources
          .map((r) => `• **${r.name}**: ${r.contact}${r.description ? ` — ${r.description}` : ''}`)
          .join('\n') +
        `\n\n💙 You matter, and there are people who care about you and want to help.`;

      const encoder = new TextEncoder();
      async function* crisisIterator() {
        const words = crisisMessage.split(/(\s+)/);
        for (const word of words) {
          yield encoder.encode(word);
          await sleep(20);
        }
      }

      const stream = iteratorToStream(crisisIterator());
      return new Response(stream, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      });
    }

    // 3. Build system prompt & format messages
    const systemPrompt = getSystemPrompt(profile);
    const formattedMessages = formatMessages(messages);

    // 4. Determine provider and stream response
    const provider = process.env.KAREN_AI_PROVIDER || 'fallback';
    const encoder = new TextEncoder();
    let fullResponse = '';

    async function* responseIterator() {
      try {
        let source;

        switch (provider) {
          case 'ollama':
            source = streamOllama(systemPrompt, formattedMessages);
            break;
          case 'openai':
            source = streamOpenAI(systemPrompt, formattedMessages);
            break;
          case 'fallback':
          default:
            source = streamFallback(sanitizedContent);
            break;
        }

        for await (const chunk of source) {
          fullResponse += chunk;
          yield encoder.encode(chunk);
        }

        // 5. Filter the complete response through safety checks
        const filtered = filterResponse(fullResponse);
        if (filtered !== fullResponse) {
          // If filtering changed the response, send a replacement marker
          // followed by the filtered version. The client should handle this.
          const replacement =
            '\n\n---\n\n' +
            '⚠️ *Some content was adjusted for appropriateness.*\n\n' +
            filtered;
          yield encoder.encode(replacement);
        }
      } catch (err) {
        console.error(`[Karen AI] Streaming error (${provider}):`, err);
        const errorMsg =
          "I'm sorry, I had a little trouble with that. Could you try asking again? 💙";
        yield encoder.encode(errorMsg);
      }
    }

    const stream = iteratorToStream(responseIterator());
    return new Response(stream, {
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
  } catch (error) {
    console.error('[Karen AI] Chat route error:', error);
    return Response.json(
      {
        error: 'An internal error occurred. Please try again.',
        details:
          process.env.NODE_ENV === 'development' ? error.message : undefined,
      },
      { status: 500 }
    );
  }
}
