import { motion } from "framer-motion";
import { Send, Shield } from "lucide-react";
import { useState } from "react";
import { API_BASE_URL } from "../App.jsx";
import BubbleBackground from "./BubbleBackground.jsx";

export default function SubmitPage() {
  const [personName, setPersonName] = useState("");
  const [textToCheck, setTextToCheck] = useState("");
  const [sendStatus, setSendStatus] = useState("idle");
  const [oopsText, setOopsText] = useState("");

  async function sendTheMessage(event) {
    event.preventDefault();

    if (!textToCheck.trim()) {
      return;
    }

    setSendStatus("loading");
    setOopsText("");

    try {
      const reply = await fetch(`${API_BASE_URL}/api/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: personName.trim() || "Anonymous diver",
          message: textToCheck.trim(),
        }),
      });

      if (!reply.ok) {
        throw new Error("Message did not send.");
      }

      setSendStatus("success");
      setTextToCheck("");
      window.setTimeout(() => setSendStatus("idle"), 2400);
    } catch {
      setSendStatus("idle");
      setOopsText("Uh oh, the message did not send. Try it again.");
    }
  }

  return (
    <main className="relative min-h-screen overflow-hidden text-[#155d7f]">
      <BubbleBackground dense />

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-md flex-col justify-center px-5 py-8">
        <motion.section className="bubble-panel p-6" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
          <div className="mb-6 text-center">
            <div className="mx-auto mb-4 grid h-20 w-20 place-items-center rounded-full border border-white/70 bg-white/50 text-5xl shadow-pearl">
              🦈
            </div>
            <h1 className="text-4xl font-black tracking-normal text-[#146b91]">SharkByte AI</h1>
            <p className="mt-2 text-base font-semibold text-[#327f9e]">
              Paste a text and see what happens.
            </p>
          </div>

          <form className="space-y-4" onSubmit={sendTheMessage}>
            <label className="block">
              <span className="label-text">Name</span>
              <input
                value={personName}
                onChange={(event) => setPersonName(event.target.value)}
                className="field"
                placeholder="Your name"
                maxLength={80}
              />
            </label>

            <label className="block">
              <span className="label-text">SMS message</span>
              <textarea
                value={textToCheck}
                onChange={(event) => setTextToCheck(event.target.value)}
                className="field min-h-36 resize-none"
                placeholder="Paste or make up a text message..."
                maxLength={800}
                required
              />
            </label>

            <button type="submit" disabled={sendStatus === "loading"} className="main-button w-full">
              {sendStatus === "loading" ? <Shield className="animate-pulse" size={20} /> : <Send size={20} />}
              Send it
            </button>
          </form>

          {sendStatus === "success" && (
            <motion.div className="success-note" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              Nice, it got sent to the dashboard!
            </motion.div>
          )}

          {oopsText && <p className="mt-4 text-center font-bold text-[#f66f86]">{oopsText}</p>}
        </motion.section>
      </div>
    </main>
  );
}
