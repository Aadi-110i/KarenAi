'use client';

import { motion } from 'framer-motion';
import { Settings, User, Bell, Shield, Moon, Sun } from 'lucide-react';
import { useState } from 'react';

export default function SettingsPage() {
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState(true);

  return (
    <main style={{ minHeight: 'calc(100vh - 70px)', padding: '4rem 2rem', position: 'relative' }}>
      <div className="grid-bg"></div>
      
      <div className="container" style={{ maxWidth: '800px', zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
            <Settings size={40} className="text-gray-400" /> Settings
          </h1>
        </motion.div>

        <div style={{ display: 'grid', gap: '1.5rem' }}>
          
          <motion.div className="glass-card" style={{ padding: '2rem' }} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
              <User size={24} className="text-blue-400" /> Profile & Identity
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', color: 'var(--color-text-muted)', marginBottom: '0.5rem' }}>Nickname</label>
                <input type="text" placeholder="What should Karen call you?" style={{ width: '100%', padding: '0.75rem 1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', outline: 'none' }} />
              </div>
              <div>
                <label style={{ display: 'block', color: 'var(--color-text-muted)', marginBottom: '0.5rem' }}>Age Group (Helps Karen tailor responses)</label>
                <select style={{ width: '100%', padding: '0.75rem 1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', outline: 'none' }}>
                  <option value="9-12">9-12 years old</option>
                  <option value="13-16">13-16 years old</option>
                </select>
              </div>
            </div>
          </motion.div>

          <motion.div className="glass-card" style={{ padding: '2rem' }} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
              <Shield size={24} className="text-green-400" /> Privacy & Safety
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <div>
                <div style={{ fontWeight: 'bold' }}>Local Storage</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Keep chat history saved locally on this device</div>
              </div>
              <input type="checkbox" defaultChecked style={{ width: '20px', height: '20px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem 0' }}>
              <div>
                <div style={{ fontWeight: 'bold', color: 'var(--color-error)' }}>Clear Chat History</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Permanently delete all previous conversations</div>
              </div>
              <button className="btn-secondary" style={{ color: 'var(--color-error)', borderColor: 'var(--color-error)' }}>Clear Data</button>
            </div>
          </motion.div>

        </div>
      </div>
    </main>
  );
}
