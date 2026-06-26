'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Sparkles, MessageCircle, Heart, Shield, BookOpen, Star } from 'lucide-react';

export default function Home() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { staggerChildren: 0.15, delayChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  const features = [
    { icon: <Heart className="text-pink-400" size={24} />, title: "Body Changes", desc: "No judgment, just facts." },
    { icon: <Sparkles className="text-purple-400" size={24} />, title: "Big Feelings", desc: "Navigate your emotions." },
    { icon: <MessageCircle className="text-blue-400" size={24} />, title: "Friends & School", desc: "Handle social drama." },
    { icon: <Shield className="text-green-400" size={24} />, title: "Awkward Stuff", desc: "Safe space to ask anything." }
  ];

  return (
    <main style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 2rem', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
      
      {/* Background Elements */}
      <div className="grid-bg"></div>
      
      <div style={{ position: 'absolute', top: '-20%', left: '-10%', width: '60vw', height: '60vw', background: 'radial-gradient(circle, var(--color-primary) 0%, transparent 60%)', opacity: 0.15, filter: 'blur(80px)', zIndex: -1 }} className="animate-float"></div>
      <div style={{ position: 'absolute', bottom: '-20%', right: '-10%', width: '50vw', height: '50vw', background: 'radial-gradient(circle, var(--color-secondary) 0%, transparent 60%)', opacity: 0.15, filter: 'blur(80px)', zIndex: -1 }} className="animate-float-reverse"></div>
      <div style={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%, -50%)', width: '40vw', height: '40vw', background: 'radial-gradient(circle, var(--color-accent) 0%, transparent 50%)', opacity: 0.1, filter: 'blur(100px)', zIndex: -1 }}></div>

      <motion.div 
        className="container" 
        style={{ maxWidth: '900px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2.5rem', zIndex: 10 }}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        
        {/* Badge */}
        <motion.div variants={itemVariants} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '999px', backdropFilter: 'blur(10px)', color: 'var(--color-text-muted)', fontSize: '0.875rem', fontWeight: 600 }}>
          <Star size={16} className="text-yellow-400" />
          <span>Your safe space for growing up</span>
        </motion.div>

        {/* Hero Section */}
        <motion.div variants={itemVariants} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <h1 style={{ margin: 0, lineHeight: 1.1 }}>
            Hi, I'm <span className="text-gradient">Karen.</span>
          </h1>
          <p style={{ fontSize: '1.25rem', color: 'var(--color-text-muted)', maxWidth: '600px', fontWeight: '400', lineHeight: 1.6 }}>
            Like having the coolest, most understanding parent… who always knows what to say. Let's talk about the hard stuff.
          </p>
        </motion.div>

        {/* Action Buttons */}
        <motion.div variants={itemVariants} style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '1rem', marginTop: '1rem' }}>
          <Link href="/chat" className="btn-primary">
            <MessageCircle size={20} />
            Start Chatting
          </Link>
          <Link href="/topics" className="btn-secondary">
            <BookOpen size={20} />
            Browse Topics
          </Link>
        </motion.div>

        {/* Features Grid */}
        <motion.div variants={itemVariants} style={{ width: '100%', marginTop: '3rem' }}>
          <h3 style={{ fontSize: '1.5rem', marginBottom: '2rem', color: 'var(--color-text)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
            <Sparkles size={24} color="var(--color-secondary)" />
            What we can talk about
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.5rem', width: '100%' }}>
            {features.map((feature, idx) => (
              <motion.div 
                key={idx}
                className="glass-card" 
                style={{ padding: '1.5rem', textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '1rem' }}
                whileHover={{ scale: 1.03 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div style={{ background: 'rgba(255,255,255,0.05)', width: '48px', height: '48px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(255,255,255,0.1)' }}>
                  {feature.icon}
                </div>
                <div>
                  <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1.125rem', color: 'var(--color-text)' }}>{feature.title}</h4>
                  <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--color-text-muted)' }}>{feature.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

      </motion.div>
    </main>
  );
}
