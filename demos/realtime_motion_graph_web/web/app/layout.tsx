import type { Metadata, Viewport } from "next";

import "./globals.css";

import { RTMGBoot } from "./RTMGBoot";

export const metadata: Metadata = {
  title: "DEMON — Realtime Motion Graph",
  description: "Reference UI for the realtime motion graph engine.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0a0a10",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <RTMGBoot />
        {children}
      </body>
    </html>
  );
}
