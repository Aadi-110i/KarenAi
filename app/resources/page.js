'use client';

import { motion } from 'framer-motion';
import { Phone, Heart, Users, ExternalLink } from 'lucide-react';
import Link from 'next/link';

export default function ResourcesPage() {
  const resources = [
    {
      category: "Immediate Help (Hotlines)",
      icon: <Phone className="text-red-400" size={24} />,
      links: [
        { name: "Crisis Text Line", desc: "Text HOME to 741741 to connect with a volunteer Crisis Counselor 24/7." },
        { name: "The Trevor Project", desc: "Call 1-866-488-7386 or text START to 678-678 (LGBTQ youth)." },
        { name: "National Suicide Prevention Lifeline", desc: "Call or text 988 (Available 24/7 in English and Spanish)." }
      ]
    },
    {
      category: "Mental Health & Support",
      icon: <Heart className="text-pink-400" size={24} />,
      links: [
        { name: "Kids Help Phone", desc: "Professional counseling and information for youth." },
        { name: "Teen Line", desc: "Teens helping teens. Text TEEN to 839863." }
      ]
    },
    {
      category: "Trusted Adults Guide",
      icon: <Users className="text-blue-400" size={24} />,
      links: [
        { name: "How to talk to parents", desc: "Tips for starting difficult conversations." },
        { name: "Finding a school counselor", desc: "What they do and how to reach out." }
      ]
    }
  ];

  return (
    <main style={{ minHeight: 'calc(100vh - 70px)', padding: '4rem 2rem', position: 'relative' }}>
      <div className="grid-bg"></div>
      
      <div className="container" style={{ maxWidth: '800px', zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <Heart className="text-red-400" size={40} /> Help & Resources
          </h1>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '1.25rem', marginBottom: '3rem' }}>
            Remember, Karen AI is not a human and cannot provide emergency help. If you are in crisis, please reach out to a professional immediately.
          </p>
        </motion.div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {resources.map((section, idx) => (
            <motion.div 
              key={idx}
              className="glass-card" 
              style={{ padding: '2rem' }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', color: 'var(--color-text)' }}>
                {section.icon} {section.category}
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {section.links.map((link, i) => (
                  <div key={i} style={{ padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontWeight: 'bold', fontSize: '1.1rem', color: 'var(--color-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {link.name} <ExternalLink size={14} />
                    </div>
                    <div style={{ color: 'var(--color-text-muted)', marginTop: '0.5rem', fontSize: '0.95rem' }}>
                      {link.desc}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </main>
  );
}
