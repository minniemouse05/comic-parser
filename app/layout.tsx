import type { Metadata } from "next";
import { SITE } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  title: `${SITE.title} ${SITE.subtitle}`,
  description:
    "A Detect-then-Match pipeline for manga speech bubble attribution: YOLOv8 detection plus an XGBoost classifier over candidate body–bubble pairs, with an interactive demo that runs the trained model in the browser.",
  openGraph: {
    title: `${SITE.title} ${SITE.subtitle}`,
    description:
      "65.1% attribution accuracy on 22 held-out Manga109 volumes. Explore the predictions interactively.",
    type: "article",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
