import './globals.css';
import { Nunito } from 'next/font/google';

const nunito = Nunito({ 
  subsets: ['latin'],
  weight: ['400', '600', '700', '800'],
  display: 'swap',
  variable: '--font-main',
});

export const metadata = {
  title: 'Karen AI — Your Growing Up Guide',
  description: 'A warm, supportive guide to help you navigate growing up.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={nunito.variable}>
      <body>
        {/* We'll add Navbar and SafetyBanner here later */}
        {children}
      </body>
    </html>
  );
}
