/** @type {import('next').NextConfig} */
const nextConfig = {
  // Expose selected env vars to server-side route handlers.
  // These are available via process.env in API routes by default,
  // but listing them here documents which ones the app relies on.
  env: {
    KAREN_AI_PROVIDER: process.env.KAREN_AI_PROVIDER,
    OLLAMA_URL: process.env.OLLAMA_URL,
    OLLAMA_MODEL: process.env.OLLAMA_MODEL,
    OPENAI_API_URL: process.env.OPENAI_API_URL,
    OPENAI_MODEL: process.env.OPENAI_MODEL,
    // Note: OPENAI_API_KEY is deliberately NOT listed here.
    // It should only be accessed server-side via process.env directly
    // and never exposed to the client bundle.
  },
};

export default nextConfig;
