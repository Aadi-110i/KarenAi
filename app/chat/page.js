'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, ShieldAlert, MoreVertical } from 'lucide-react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';

export default function ChatPage() {
  const searchParams = useSearchParams();
  const topicId = searchParams.get('topic');
  
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi there! I'm Karen. ✨ I'm here to listen, answer questions, or just chat. Whatever's on your mind, I'm ready. What's up?" }
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (topicId) {
      setMessages([
        { role: 'assistant', content: `Hi! I saw you wanted to talk about a specific topic. I'm here to listen and help without any judgment. Whenever you're ready, tell me what's on your mind.` }
      ]);
    }
  }, [topicId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');

    setTimeout(() => {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "I hear you. It's completely normal to feel that way. I'm still running in placeholder mode while my brain is being trained, but I'm always here for you!" 
      }]);
    }, 1200);
  };

  return (
    <div style={{ height: 'calc(100vh - 70px)', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      
      {/* Chat Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '900px', margin: '0 auto', width: '100%' }}>
        
        <div style={{ textAlign: 'center', margin: '1rem 0' }}>
          <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.2rem 1rem', borderRadius: '1rem', fontSize: '0.8rem', color: 'var(--color-text-muted)', fontWeight: '600', border: '1px solid rgba(255,255,255,0.1)' }}>
            Today
          </span>
        </div>

        {messages.map((msg, i) => (
          <motion.div 
            key={i} 
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
              gap: '0.5rem'
            }}
          >
            {msg.role === 'assistant' && (
              <span style={{ fontSize: '0.8rem', color: 'var(--color-primary)', fontWeight: '700', marginLeft: '1rem' }}>Karen ✨</span>
            )}
            
            <div style={{
              maxWidth: '85%',
              padding: '1rem 1.5rem',
              borderRadius: msg.role === 'user' ? '20px 20px 0 20px' : '20px 20px 20px 0',
              background: msg.role === 'user' 
                ? 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))' 
                : 'rgba(30, 41, 59, 0.7)',
              color: 'var(--color-text)',
              boxShadow: msg.role === 'user' ? 'var(--shadow-glow)' : 'var(--shadow-md)',
              border: msg.role === 'assistant' ? '1px solid rgba(255,255,255,0.1)' : 'none',
              backdropFilter: 'blur(10px)',
              fontSize: '1.05rem',
              lineHeight: '1.5'
            }}>
              {msg.content}
            </div>
          </motion.div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ padding: '2rem', background: 'rgba(15, 23, 42, 0.9)', backdropFilter: 'blur(12px)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <form onSubmit={handleSend} style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          
          <Link href="/resources" style={{ color: 'var(--color-text-muted)', background: 'rgba(255,255,255,0.05)', padding: '0.75rem', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}>
            <ShieldAlert size={20} />
          </Link>

          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type anything... I'm listening." 
            style={{ 
              flex: 1, 
              padding: '1rem 1.5rem', 
              borderRadius: 'var(--radius-full)', 
              border: '1px solid rgba(255,255,255,0.1)', 
              background: 'rgba(255,255,255,0.05)',
              color: 'var(--color-text)',
              fontSize: '1.1rem',
              outline: 'none',
              transition: 'border-color var(--transition-fast)'
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--color-primary)'}
            onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
          />
          <button 
            type="submit" 
            disabled={!input.trim()}
            className={input.trim() ? "btn-primary" : ""}
            style={{ 
              background: input.trim() ? 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))' : 'rgba(255,255,255,0.1)',
              color: input.trim() ? 'white' : 'var(--color-text-muted)',
              width: '3.5rem',
              height: '3.5rem',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              border: 'none',
              padding: 0
            }}
          >
            <Send size={20} style={{ transform: input.trim() ? 'translateX(2px) translateY(-2px)' : 'none', transition: 'transform 0.2s' }} />
          </button>
        </form>
      </div>
    </div>
  );
}
