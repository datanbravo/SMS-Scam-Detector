import { motion } from "framer-motion";

export default function IncomingNotification({ message }) {
  if (!message) return null;

  const isBadMessage = message.classification === "scam" || message.classification === "suspicious";

  return (
    <motion.div
      className={`incoming-box ${isBadMessage ? "incoming-bad" : "incoming-good"}`}
      initial={{ opacity: 0, x: 60 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <p className="text-sm font-black uppercase tracking-wide text-[#2384a9]">
        New message popped up...
      </p>

      <div className="mt-3 flex items-start gap-3">
        <div className="small-avatar">{message.avatar}</div>

        <div className="min-w-0 flex-1">
          <p className="text-lg font-black text-[#166b90]">{message.name}</p>
          <p className="line-clamp-2 text-sm font-semibold text-[#397f9d]">
            {message.message_text}
          </p>

          <div className="mt-3 flex items-center gap-2 text-sm font-bold text-[#2384a9]">
            <span className="scanning-dot" />
            Checking it real quick...
          </div>
        </div>
      </div>
    </motion.div>
  );
}
