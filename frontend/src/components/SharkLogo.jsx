import { motion } from "framer-motion";

export default function SharkLogo({ active = false }) {
  return (
    <div className="relative inline-flex items-center justify-center">
      {active && (
        <motion.span
          className="absolute -left-10 top-0 text-3xl"
          initial={{ x: -70, opacity: 0 }}
          animate={{ x: 2, opacity: 1, rotate: [0, -4, 4, 0] }}
          transition={{ duration: 0.9 }}
        >
          🦈
        </motion.span>
      )}

      <span className="relative">
        SharkByte AI
        {active && <span className="bite-mark" />}
      </span>
    </div>
  );
}
