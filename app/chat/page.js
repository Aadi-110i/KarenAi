'use client';
import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi there! I'm Karen. 😊 I'm here to listen, answer questions, or just chat. Whatever's on your mind, I'm ready. What's up?" }
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');

    // Simulate Karen typing (placeholder for actual API call later)
    setTimeout(() => {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "I hear you. That's totally normal to feel that way! (Note: I'm currently running in offline placeholder mode. My AI brain is still being built! 🧠)" 
      }]);
    }, 1500);
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--color-background)' }}>
      {/* Header */}
      <header style={{ padding: '1rem 2rem', background: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(10px)', borderBottom: '1px solid rgba(0,0,0,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10 }}>
        <Link href="/" style={{ fontWeight: '800', fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)' }}>
          <span>🌸</span> Karen AI
        </Link>
        <div style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)', fontWeight: '600' }}>
          Safe Space 🛡️
        </div>
      </header>

      {/* Chat Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '800px', margin: '0 auto', width: '100%' }}>
        
        {/* Date separator */}
        <div style={{ textAlign: 'center', margin: '1rem 0' }}>
          <span style={{ background: 'rgba(0,0,0,0.05)', padding: '0.2rem 1rem', borderRadius: '1rem', fontSize: '0.8rem', color: 'var(--color-text-muted)', fontWeight: '600' }}>
            Today
          </span>
        </div>

        {messages.map((msg, i) => (
          <div key={i} className="animate-fade-in" style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
            gap: '0.5rem'
          }}>
            {msg.role === 'assistant' && (
              <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', fontWeight: '700', marginLeft: '1rem' }}>Karen ✨</span>
            )}
            
            <div style={{
              maxWidth: '80%',
              padding: '1rem 1.5rem',
              borderRadius: msg.role === 'user' ? '20px 20px 0 20px' : '20px 20px 20px 0',
              background: msg.role === 'user' 
                ? 'linear-gradient(135deg, var(--color-secondary), #65b8b8)' 
                : 'white',
              color: msg.role === 'user' ? 'white' : 'var(--color-text)',
              boxShadow: 'var(--shadow-sm)',
              border: msg.role === 'assistant' ? '1px solid rgba(0,0,0,0.05)' : 'none',
              fontSize: '1.05rem',
              lineHeight: '1.5'
            }}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ padding: '2rem', background: 'white', borderTop: '1px solid rgba(0,0,0,0.05)', boxShadow: '0 -4px 20px rgba(0,0,0,0.02)' }}>
        <form onSubmit={handleSend} style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', gap: '1rem' }}>
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type anything... I'm listening." 
            style={{ 
              flex: 1, 
              padding: '1rem 1.5rem', 
              borderRadius: 'var(--radius-full)', 
              border: '2px solid rgba(0,0,0,0.05)', 
              fontSize: '1.1rem',
              fontFamily: 'inherit',
              outline: 'none',
              transition: 'border-color var(--transition-fast)'
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--color-primary)'}
            onBlur={(e) => e.target.style.borderColor = 'rgba(0,0,0,0.05)'}
          />
          <button 
            type="submit" 
            disabled={!input.trim()}
            style={{ 
              background: input.trim() ? 'linear-gradient(135deg, var(--color-primary), var(--color-accent))' : '#e2e8f0',
              color: 'white',
              width: '3.5rem',
              height: '3.5rem',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              transition: 'transform var(--transition-fast), box-shadow var(--transition-fast)',
              boxShadow: input.trim() ? 'var(--shadow-md)' : 'none'
            }}
            onMouseEnter={(e) => { if(input.trim()) e.currentTarget.style.transform = 'scale(1.05)' }}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}
