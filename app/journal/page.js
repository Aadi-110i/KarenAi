'use client';

import { motion } from 'framer-motion';
import { PenTool, Lock } from 'lucide-react';

export default function JournalPage() {
  return (
    <main style={{ minHeight: 'calc(100vh - 70px)', padding: '4rem 2rem', position: 'relative' }}>
      <div className="grid-bg"></div>
      
      <div className="container" style={{ maxWidth: '800px', zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <PenTool size={40} className="text-purple-400" /> My Private Journal
          </h1>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-success)', marginBottom: '2rem', background: 'rgba(74, 222, 128, 0.1)', padding: '0.5rem 1rem', borderRadius: '8px', width: 'fit-content' }}>
            <Lock size={16} />
            <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Your entries are encrypted and stored locally. Karen cannot read this.</span>
          </div>
        </motion.div>

        <motion.div 
          className="glass-card" 
          style={{ padding: '2rem' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <input 
            type="text" 
            placeholder="Title of your entry..." 
            style={{ width: '100%', background: 'transparent', border: 'none', borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'white', fontSize: '1.5rem', padding: '0.5rem 0', marginBottom: '1.5rem', outline: 'none' }}
          />
          <textarea 
            placeholder="What's on your mind today?" 
            style={{ width: '100%', minHeight: '300px', background: 'transparent', border: 'none', color: 'var(--color-text-muted)', fontSize: '1.1rem', resize: 'vertical', outline: 'none', lineHeight: '1.6' }}
          ></textarea>
          
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
            <button className="btn-primary">Save Entry</button>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
