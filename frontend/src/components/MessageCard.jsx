import { motion } from "framer-motion";

export default function MessageCard({ message, onClick }) {
  const looksScammy = message.classification === "scam";
  const madeAt = new Date(message.created_at).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <motion.button
      type="button"
      onClick={() => onClick(message)}
      className={`message-card ${looksScammy ? "message-card-bad" : "message-card-good"}`}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      layout
    >
      <div className="flex items-start gap-3">
        <span className="avatar-circle">{message.avatar}</span>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-lg font-black text-[#166b90]">{message.name}</h3>
            <span className={`status-badge ${looksScammy ? "status-scam" : "status-safe"}`}>
              {looksScammy ? "Scam-ish" : "Looks okay"}
            </span>
          </div>

          <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-[#2e6f8c]">
            {message.message_text}
          </p>

          <p className="mt-3 text-xs font-bold uppercase tracking-wide text-[#5ca5bd]">
            {madeAt}
          </p>
        </div>
      </div>
    </motion.button>
  );
}
