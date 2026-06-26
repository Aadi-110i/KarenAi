'use client';

import { motion } from 'framer-motion';
import { BookOpen, Heart, Shield, Star, Smile } from 'lucide-react';
import Link from 'next/link';

export default function TopicsPage() {
  const topics = [
    { id: 1, title: 'Physical Changes', icon: <Heart size={32} className="text-pink-400" />, desc: 'Understanding what is happening to your body.' },
    { id: 2, title: 'Big Emotions', icon: <Smile size={32} className="text-yellow-400" />, desc: 'Navigating mood swings and new feelings.' },
    { id: 3, title: 'Friendships', icon: <Star size={32} className="text-purple-400" />, desc: 'Dealing with drama and making real connections.' },
    { id: 4, title: 'Setting Boundaries', icon: <Shield size={32} className="text-green-400" />, desc: 'Learning to say no and stay safe.' }
  ];

  return (
    <main style={{ minHeight: 'calc(100vh - 70px)', padding: '4rem 2rem', position: 'relative', overflow: 'hidden' }}>
      <div className="grid-bg"></div>
      
      <div className="container" style={{ maxWidth: '1000px', zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <BookOpen size={40} className="text-blue-400" /> Browse Topics
          </h1>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '1.25rem', marginBottom: '3rem' }}>
            Choose a topic you want to learn more about or discuss with Karen.
          </p>
        </motion.div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '2rem' }}>
          {topics.map((topic, i) => (
            <motion.div 
              key={topic.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
            >
              <Link href={`/chat?topic=${topic.id}`} className="glass-card" style={{ display: 'block', padding: '2rem', height: '100%', textDecoration: 'none' }}>
                <div style={{ marginBottom: '1rem' }}>{topic.icon}</div>
                <h3 style={{ marginBottom: '0.5rem' }}>{topic.title}</h3>
                <p style={{ color: 'var(--color-text-muted)', margin: 0 }}>{topic.desc}</p>
                <div style={{ marginTop: '1.5rem', color: 'var(--color-primary)', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  Discuss this <span style={{ fontSize: '1.2rem' }}>→</span>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </main>
  );
}
