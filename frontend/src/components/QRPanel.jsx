import { QRCodeSVG } from "qrcode.react";
import { Smartphone } from "lucide-react";

export default function QRPanel() {
  const baseSubmitLink = window.location.origin.replace(/\/$/, "");
  const submitLink = import.meta.env.VITE_PUBLIC_SUBMIT_URL || `${baseSubmitLink}/submit`;

  return (
    <section className="bubble-panel flex items-center gap-4 p-4">
      <div className="qr-box">
        <QRCodeSVG value={submitLink} size={112} bgColor="#ffffff" fgColor="#1f7fa4" />
      </div>

      <div>
        <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-white/60 bg-white/50 px-3 py-1 text-sm font-bold text-[#1e7fa5]">
          <Smartphone size={16} />
          Scan this thing
        </div>
        <p className="max-w-48 text-sm font-semibold text-[#397f9d]">
          Send a message from your phone.
        </p>
      </div>
    </section>
  );
}
