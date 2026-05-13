import { AnimatePresence, motion } from "framer-motion";
import { Sparkles, X } from "lucide-react";
import { useMemo, useState } from "react";

function MessageWithHighlights({ message }) {
  const messagePieces = useMemo(() => {
    const fullText = message.message_text || "";
    const markedParts = [...(message.suspicious_phrases || [])]
      .filter((part) => Number.isFinite(part.start_index) && Number.isFinite(part.end_index))
      .sort((first, second) => first.start_index - second.start_index);

    if (!markedParts.length) {
      return [{ text: fullText, danger: false }];
    }

    const pieces = [];
    let lastSpot = 0;

    markedParts.forEach((part) => {
      const startSpot = Math.max(0, Number(part.start_index));
      const endSpot = Math.min(fullText.length, Number(part.end_index));

      if (startSpot > lastSpot) {
        pieces.push({ text: fullText.slice(lastSpot, startSpot), danger: false });
      }

      if (endSpot > startSpot) {
        pieces.push({ text: fullText.slice(startSpot, endSpot), danger: true });
      }

      lastSpot = Math.max(lastSpot, endSpot);
    });

    if (lastSpot < fullText.length) {
      pieces.push({ text: fullText.slice(lastSpot), danger: false });
    }

    return pieces;
  }, [message]);

  return (
    <p className="message-read-box">
      {messagePieces.map((piece, spot) => (
        <span key={`${piece.text}-${spot}`} className={piece.danger ? "danger-words" : ""}>
          {piece.text}
        </span>
      ))}
    </p>
  );
}

export default function MessageDetailPanel({ message, onClose }) {
  const [showTheWhy, setShowTheWhy] = useState(false);
  const isScamMessage = message?.classification === "scam";

  return (
    <AnimatePresence>
      {message && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-[#68cce7]/30 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          <motion.aside
            className="detail-panel"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
          >
            <div className="mb-6 flex items-center justify-between gap-4">
              <div>
                <p className="label-text">Message check</p>
                <h2 className="mt-2 text-3xl font-black text-[#146b91]">
                  {message.avatar} {message.name}
                </h2>
              </div>

              <button type="button" onClick={onClose} className="icon-button" aria-label="Close message check">
                <X size={22} />
              </button>
            </div>

            <MessageWithHighlights message={message} />

            <div className="info-row mt-5">
              <span className="label-text">Result</span>
              <span className={`status-badge ${isScamMessage ? "status-scam" : "status-safe"}`}>
                {isScamMessage ? "Scam-ish" : "Looks okay"}
              </span>
            </div>

            <button type="button" onClick={() => setShowTheWhy((oldValue) => !oldValue)} className="main-button mt-6 w-full">
              <Sparkles size={20} />
              {showTheWhy ? "Hide the why" : "Show the why"}
            </button>

            <AnimatePresence>
              {showTheWhy && (
                <motion.div className="mt-6 space-y-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <p className="plain-note">
                    {message.short_explanation || "No major scam patterns were found."}
                  </p>

                  <section>
                    <h3 className="label-text mb-3">Suspicious phrases</h3>

                    {message.suspicious_phrases?.length ? (
                      <div className="space-y-3">
                        {message.suspicious_phrases.map((phrase, spot) => (
                          <div key={`${phrase.phrase_text}-${spot}`} className="phrase-box">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="rounded-full bg-coral px-3 py-1 text-sm font-black text-white">
                                {phrase.phrase_text}
                              </span>
                              <span className="rounded-full bg-white/60 px-3 py-1 text-sm font-bold text-[#9f3348]">
                                {String(phrase.risk_category || "risk").replaceAll("_", " ")}
                              </span>
                            </div>

                            <p className="mt-3 text-sm font-semibold text-[#9f3348]">
                              {phrase.risk_explanation || "This part felt a little sketchy."}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="plain-note">Nothing weird got marked in this message.</p>
                    )}
                  </section>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
