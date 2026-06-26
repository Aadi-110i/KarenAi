'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, MessageCircle, BookOpen, PenTool, Info, ShieldAlert, Settings } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Home', path: '/', icon: <Home size={18} /> },
    { name: 'Chat', path: '/chat', icon: <MessageCircle size={18} /> },
    { name: 'Topics', path: '/topics', icon: <BookOpen size={18} /> },
    { name: 'Journal', path: '/journal', icon: <PenTool size={18} /> },
    { name: 'Resources', path: '/resources', icon: <ShieldAlert size={18} /> },
  ];

  return (
    <nav style={{ 
      position: 'sticky', 
      top: 0, 
      zIndex: 100, 
      background: 'rgba(15, 23, 42, 0.8)', 
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid rgba(255,255,255,0.05)',
      padding: '1rem 0'
    }}>
      <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 'bold', fontSize: '1.25rem', color: 'var(--color-primary)' }}>
          <span style={{ fontSize: '1.5rem' }}>✨</span> Karen AI
        </Link>
        
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {navItems.map((item) => {
            const isActive = pathname === item.path;
            return (
              <Link key={item.name} href={item.path} style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', borderRadius: '999px', color: isActive ? 'var(--color-text)' : 'var(--color-text-muted)', fontWeight: isActive ? '700' : '500', transition: 'color 0.2s' }}>
                {item.icon}
                <span className="hidden sm:inline">{item.name}</span>
                {isActive && (
                  <motion.div
                    layoutId="navbar-indicator"
                    style={{ position: 'absolute', inset: 0, background: 'rgba(255,255,255,0.1)', borderRadius: '999px', zIndex: -1 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
              </Link>
            );
          })}
          <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)', margin: '0 0.5rem' }}></div>
          <Link href="/about" style={{ color: pathname === '/about' ? 'var(--color-primary)' : 'var(--color-text-muted)', padding: '0.5rem' }}>
            <Info size={20} />
          </Link>
          <Link href="/settings" style={{ color: pathname === '/settings' ? 'var(--color-primary)' : 'var(--color-text-muted)', padding: '0.5rem' }}>
            <Settings size={20} />
          </Link>
        </div>
      </div>
    </nav>
  );
}
