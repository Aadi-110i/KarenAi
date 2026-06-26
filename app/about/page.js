'use client';

import { motion } from 'framer-motion';
import { Info, ShieldCheck, HeartHandshake } from 'lucide-react';

export default function AboutPage() {
  return (
    <main style={{ minHeight: 'calc(100vh - 70px)', padding: '4rem 2rem', position: 'relative' }}>
      <div className="grid-bg"></div>
      
      <div className="container" style={{ maxWidth: '800px', zIndex: 10 }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
            <Info size={40} className="text-accent" /> About Karen AI
          </h1>
        </motion.div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <motion.div className="glass-card" style={{ padding: '2rem' }} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-primary)' }}>
              <HeartHandshake size={24} /> Our Mission
            </h3>
            <p style={{ color: 'var(--color-text-muted)', margin: 0 }}>
              Growing up is hard, confusing, and sometimes a little scary. Karen AI was built to be a supportive companion that can answer awkward questions, provide emotional support, and help teenagers navigate the complexities of adolescence without judgment.
            </p>
          </motion.div>

          <motion.div className="glass-card" style={{ padding: '2rem' }} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-success)' }}>
              <ShieldCheck size={24} /> Built for Safety
            </h3>
            <p style={{ color: 'var(--color-text-muted)', margin: 0 }}>
              Karen AI is fine-tuned using a rigorous training pipeline (including QLoRA and DPO safety alignment) to ensure all conversations remain age-appropriate. Karen is trained to recognize when professional help or a trusted adult is needed, and will never compromise user boundaries.
            </p>
          </motion.div>
        </div>
      </div>
    </main>
  );
}
