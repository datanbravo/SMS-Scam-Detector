import { motion } from "framer-motion";

//These bubbles are fake, but they make the page feel alive.
const littleBubbles = Array.from({ length: 32 }, (_, spot) => ({
  id: spot,
  leftSide: `${(spot * 31) % 100}%`,
  bubbleSize: 12 + ((spot * 11) % 34),
  wait: (spot % 8) * 0.45,
  speed: 8 + (spot % 6),
}));

export default function BubbleBackground({ dense = false }) {
  const bubblesToShow = dense ? littleBubbles : littleBubbles.slice(0, 20);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 aquarium-gradient" />
      <div className="absolute inset-0 ocean-rays" />

      {bubblesToShow.map((bubble) => (
        <motion.span
          key={bubble.id}
          className="plain-bubble absolute bottom-[-60px] rounded-full"
          style={{
            left: bubble.leftSide,
            width: bubble.bubbleSize,
            height: bubble.bubbleSize,
          }}
          animate={{
            y: ["0vh", "-110vh"],
            x: [0, bubble.id % 2 === 0 ? 18 : -18, 5],
            opacity: [0, 0.55, 0],
          }}
          transition={{
            duration: bubble.speed,
            delay: bubble.wait,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
