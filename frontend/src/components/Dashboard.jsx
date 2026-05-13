import { AnimatePresence, motion } from "framer-motion";
import { Filter, ShieldCheck, Siren, Waves } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL, getDashboardSocketUrl } from "../App.jsx";
import BubbleBackground from "./BubbleBackground.jsx";
import IncomingNotification from "./IncomingNotification.jsx";
import MessageCard from "./MessageCard.jsx";
import MessageDetailPanel from "./MessageDetailPanel.jsx";
import QRPanel from "./QRPanel.jsx";
import SharkLogo from "./SharkLogo.jsx";
import SplashScreen from "./SplashScreen.jsx";

const fakeMessages = [
  {
    id: "sample-1",
    name: "Maya",
    avatar: "🐢",
    message_text: "Hey, our study group is meeting at 4 after class.",
    classification: "safe",
    suspicious_phrases: [],
    annotation_count: 0,
    risk_categories_present: "",
    short_explanation: "No major scam patterns detected.",
    created_at: new Date().toISOString(),
  },
  {
    id: "sample-2",
    name: "Jordan",
    avatar: "🦀",
    message_text: "URGENT: your account will be suspended. Click this link to verify payment now.",
    classification: "scam",
    suspicious_phrases: [
      {
        phrase_text: "URGENT",
        start_index: 0,
        end_index: 6,
        risk_category: "urgency",
        risk_explanation: "Urgent words can make people rush and not think it through.",
      },
      {
        phrase_text: "Click this link",
        start_index: 40,
        end_index: 55,
        risk_category: "link_request",
        risk_explanation: "Random links can lead to fake login or payment pages.",
      },
    ],
    annotation_count: 2,
    risk_categories_present: "urgency, link_request",
    short_explanation: "This one has pressure and a link, which is pretty sus.",
    created_at: new Date(Date.now() - 90000).toISOString(),
  },
];

const filterChoices = [
  { key: "all", label: "All", icon: Filter },
  { key: "safe", label: "Okay ones", icon: ShieldCheck },
  { key: "scam", label: "Sus ones", icon: Siren },
];

function StatCard({ label, value, coral = false }) {
  return (
    <div className={`stat-pill ${coral ? "stat-pill-coral" : ""}`}>
      <p className="text-xs font-black uppercase tracking-wide text-[#2384a9]">{label}</p>
      <p className="mt-1 text-3xl font-black text-[#146b91]">{value}</p>
    </div>
  );
}

export default function Dashboard() {
  const [stillSplashing, setStillSplashing] = useState(true);
  const [allMessages, setAllMessages] = useState([]);
  const [messageComingIn, setMessageComingIn] = useState(null);
  const [pickedFilter, setPickedFilter] = useState("all");
  const [openedMessage, setOpenedMessage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const socketSpot = useRef(null);
  const timerSpot = useRef(null);
  const reconnectTry = useRef(0);

  const addMessageToTop = useCallback((newMessage) => {
    setMessageComingIn(newMessage);

    window.setTimeout(() => {
      setAllMessages((currentMessages) => {
        const alreadyHere = currentMessages.some((oldMessage) => oldMessage.id === newMessage.id);

        if (alreadyHere) {
          return currentMessages;
        }

        return [newMessage, ...currentMessages];
      });

      setMessageComingIn((current) => (current?.id === newMessage.id ? null : current));
    }, 1500);
  }, []);

  useEffect(() => {
    const stopFetch = new AbortController();
    let keepTrying = true;

    //Get the messages that are already saved.
    fetch(`${API_BASE_URL}/api/messages`, { signal: stopFetch.signal })
      .then((reply) => (reply.ok ? reply.json() : []))
      .then((data) => setAllMessages(Array.isArray(data) ? data : []))
      .catch((error) => {
        if (error.name !== "AbortError") {
          setAllMessages([]);
        }
      });

    const pollingTimer = window.setInterval(() => {
      fetch(`${API_BASE_URL}/api/messages`)
        .then((reply) => (reply.ok ? reply.json() : []))
        .then((data) => setAllMessages(Array.isArray(data) ? data : []))
        .catch(() => {});
    }, 3000);

    function clearReconnectTimer() {
      if (timerSpot.current) {
        window.clearTimeout(timerSpot.current);
        timerSpot.current = null;
      }
    }

    function tryAgainSoon() {
      if (!keepTrying || timerSpot.current) {
        return;
      }

      const waitTime = Math.min(1000 * 2 ** reconnectTry.current, 10000);
      reconnectTry.current = Math.min(reconnectTry.current + 1, 4);

      timerSpot.current = window.setTimeout(() => {
        timerSpot.current = null;
        startSocket();
      }, waitTime);
    }

    function startSocket() {
      if (!keepTrying) {
        return;
      }

      const dashboardSocket = new WebSocket(getDashboardSocketUrl());
      socketSpot.current = dashboardSocket;

      dashboardSocket.onopen = () => {
        if (socketSpot.current !== dashboardSocket) return;
        reconnectTry.current = 0;
        setIsConnected(true);
      };

      dashboardSocket.onmessage = (event) => {
        if (socketSpot.current !== dashboardSocket) return;

        try {
          const socketData = JSON.parse(event.data);

          if (socketData.type === "snapshot") {
            setAllMessages(Array.isArray(socketData.payload) ? socketData.payload : []);
          }

          if (socketData.type === "message") {
            addMessageToTop(socketData.payload);
          }
        } catch {
          //Sometimes socket data is weird, so this keeps the page from crashing.
        }
      };

      dashboardSocket.onerror = () => {
        if (socketSpot.current !== dashboardSocket) return;
        setIsConnected(false);

        if (dashboardSocket.readyState !== WebSocket.CLOSED && dashboardSocket.readyState !== WebSocket.CLOSING) {
          dashboardSocket.close();
        }
      };

      dashboardSocket.onclose = () => {
        if (socketSpot.current !== dashboardSocket) return;
        setIsConnected(false);
        socketSpot.current = null;
        tryAgainSoon();
      };
    }

    startSocket();

    return () => {
      keepTrying = false;
      stopFetch.abort();
      clearReconnectTimer();
      window.clearInterval(pollingTimer);
      setIsConnected(false);

      const oldSocket = socketSpot.current;
      socketSpot.current = null;

      if (oldSocket) {
        oldSocket.onopen = null;
        oldSocket.onmessage = null;
        oldSocket.onerror = null;
        oldSocket.onclose = null;

        if (oldSocket.readyState === WebSocket.OPEN || oldSocket.readyState === WebSocket.CONNECTING) {
          oldSocket.close();
        }
      }
    };
  }, [addMessageToTop]);

  const countStuff = useMemo(() => {
    const total = allMessages.length;
    const scam = allMessages.filter((oneMessage) => oneMessage.classification === "scam").length;

    return {
      total,
      scam,
      safe: total - scam,
    };
  }, [allMessages]);

  const messagesToShow = useMemo(() => {
    const realOrFakeMessages = allMessages.length ? allMessages : fakeMessages;

    if (pickedFilter === "all") {
      return realOrFakeMessages;
    }

    return realOrFakeMessages.filter((oneMessage) => oneMessage.classification === pickedFilter);
  }, [allMessages, pickedFilter]);

  return (
    <main className="relative min-h-screen overflow-hidden text-[#155d7f]">
      <BubbleBackground dense />
      <AnimatePresence>{stillSplashing && <SplashScreen onDone={() => setStillSplashing(false)} />}</AnimatePresence>

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-[1500px] flex-col px-5 py-6 lg:px-10">
        <header className="grid grid-cols-1 items-start gap-5 lg:grid-cols-[320px_1fr_320px]">
          <QRPanel />

          <section className="text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/60 bg-white/40 px-5 py-2 text-sm font-bold text-[#1f789d] shadow-pearl backdrop-blur-xl">
              <Waves size={17} />
              {isConnected ? "Live link is working" : "Waiting for live link"}
            </div>

            <h1 className="mt-4 text-5xl font-black tracking-normal text-[#146b91] drop-shadow-[0_8px_30px_rgba(255,255,255,.65)] md:text-7xl">
              <SharkLogo active={pickedFilter === "scam"} />
            </h1>

            <p className="mt-3 text-xl font-semibold text-[#327f9e]">
              Catching sketchy texts before they get you.
            </p>
          </section>

          <section className="grid grid-cols-3 gap-3">
            <StatCard label="Total" value={countStuff.total} />
            <StatCard label="Scam" value={countStuff.scam} coral />
            <StatCard label="Safe" value={countStuff.safe} />
          </section>
        </header>

        <section className="mt-7 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap gap-3">
            {filterChoices.map((filter) => {
              const Icon = filter.icon;
              const isPicked = pickedFilter === filter.key;

              return (
                <button
                  key={filter.key}
                  type="button"
                  onClick={() => setPickedFilter(filter.key)}
                  className={`filter-button ${isPicked ? "filter-button-active" : ""}`}
                >
                  <Icon size={18} />
                  {filter.label}
                </button>
              );
            })}
          </div>

          <div className="rounded-full border border-white/60 bg-white/40 px-4 py-2 text-sm font-bold text-[#1f789d] shadow-pearl">
            {allMessages.length ? "Real messages are showing." : "Sample messages for now."}
          </div>
        </section>

        <section className="relative mt-6 flex-1">
          <div className="pointer-events-none absolute right-0 top-0 z-20 w-full max-w-md">
            <AnimatePresence>
              {messageComingIn && <IncomingNotification message={messageComingIn} />}
            </AnimatePresence>
          </div>

          <motion.div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3" layout>
            <AnimatePresence>
              {messagesToShow.map((message) => (
                <MessageCard key={message.id} message={message} onClick={setOpenedMessage} />
              ))}
            </AnimatePresence>
          </motion.div>

          {!messagesToShow.length && (
            <div className="bubble-panel p-8 text-center text-lg font-bold text-[#327f9e]">
              Nothing here yet. Kinda peaceful honestly.
            </div>
          )}
        </section>
      </div>

      <MessageDetailPanel message={openedMessage} onClose={() => setOpenedMessage(null)} />
    </main>
  );
}
