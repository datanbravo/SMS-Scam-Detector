import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const splashDots = Array.from({ length: 36 }, (_, dot) => ({
  id: dot,
  left: `${(dot * 17) % 100}%`,
  top: `${(dot * 23) % 100}%`,
  size: 10 + ((dot * 9) % 45),
}));

export default function SplashScreen({ onDone }) {
  const [logoWorks, setLogoWorks] = useState(true);

  useEffect(() => {
    //This lets the splash screen show before the dashboard pops in.
    const splashTimer = window.setTimeout(() => { 
      onDone();
    }, 1800);
    return () => window.clearTimeout(splashTimer);
  }, [onDone]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden bg-[#b9f1ff]"
      exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 aquarium-gradient" />

      {splashDots.map((dot) => (
        <motion.span
          key={dot.id}
          className="plain-bubble absolute"
          style={{
            left: dot.left,
            top: dot.top,
            width: dot.size,
            height: dot.size,
          }}
          animate={{ y: [0, -30, 0], opacity: [0.15, 0.45, 0.15] }}
          transition={{ duration: 2.5 + (dot.id % 4), repeat: Infinity }}
        />
      ))}

      <motion.div
        className="relative z-10 text-center"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        {logoWorks ? (
          <div className="mx-auto mb-5 rounded-[2rem] bg-white/45 p-4 shadow-pearl">
            <img
              src="/assets/logo.png"
              alt="SharkByte AI"
              className="max-h-48 max-w-[min(76vw,32rem)] object-contain"
              onError={() => setLogoWorks(false)}
            />
          </div>
        ) : (
          <>
            <div className="mx-auto mb-4 grid h-28 w-28 place-items-center rounded-full bg-white/50 text-6xl shadow-pearl">
              🦈
            </div>
            <h1 className="text-6xl font-black text-[#146b91]">SharkByte AI</h1>
          </>
        )}

        <p className="mt-4 text-xl font-semibold text-[#2f86a8]">
          Loading the bubbly stuff...
        </p>
      </motion.div>
    </motion.div>
  );
}
