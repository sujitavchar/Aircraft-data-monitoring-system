import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import UploadIcon from "../assest/logo.svg";
import '../style/UploadPage.css';
import { PDFDocument, rgb, StandardFonts } from "pdf-lib";


import toast, { Toaster } from "react-hot-toast";

function BlackBoxPage() {
  const fileInputRef = useRef(null);
  const [fileData, setFileData] = useState(null);
  const navigate = useNavigate();

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
      setFileData(file);
      toast.success("📂 File selected successfully!", { duration: 4000 });
    }
  };

  const handleGoClick = async () => {
    if (!fileData) {
      toast.error("⚠️ Please select a file first.", { duration: 4000 });
      return;
    }
    navigate("/graphs", { state: { fileData } });
    const formData = new FormData();
    formData.append("file", fileData);
    try {
      // const response = await fetch("http://127.0.0.1:8000/upload"
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Upload failed");
      }
      const data = await response.json();
      console.log("Backend response:", data);
      generatePDF(data.report_text);
      toast.success(" PDF downloaded on your computer!", { duration: 4000 });

    } catch (error) {
      toast.error("Something went wrong. Please try again.", { duration: 4000 });
    }
  };



const generatePDF = async (text) => {
  const pdfDoc = await PDFDocument.create();
  const fontNormal = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
  const fontItalic = await pdfDoc.embedFont(StandardFonts.HelveticaOblique);

  const PAGE_W = 612, PAGE_H = 792;
  const MARGIN = 56;
  const CONTENT_W = PAGE_W - MARGIN * 2;
  const FOOTER_RESERVED = 50;

  const ACCENT = rgb(0.09, 0.2, 0.42);     // navy
  const TEXT_DARK = rgb(0.12, 0.12, 0.14);
  const TEXT_BODY = rgb(0.22, 0.22, 0.25);
  const TEXT_MUTED = rgb(0.5, 0.5, 0.54);
  const RULE_LIGHT = rgb(0.85, 0.85, 0.88);

  let page, y;

  const topAccentBar = (p, thick) => {
    p.drawRectangle({ x: 0, y: PAGE_H - thick, width: PAGE_W, height: thick, color: ACCENT });
  };

  const newPage = () => {
    page = pdfDoc.addPage([PAGE_W, PAGE_H]);
    topAccentBar(page, 4);
    y = PAGE_H - MARGIN - 10;
  };

  const ensureSpace = (needed) => {
    if (y - needed < FOOTER_RESERVED) newPage();
  };

  // --- tokenize a line into {text, bold} words, respecting **bold** spans ---
  const tokenize = (line) => {
    const words = [];
    const regex = /\*\*(.*?)\*\*/g;
    let lastIndex = 0, match;
    const pushPlain = (chunk, bold) => {
      chunk.split(/\s+/).filter(Boolean).forEach((w) => words.push({ text: w, bold }));
    };
    while ((match = regex.exec(line)) !== null) {
      if (match.index > lastIndex) pushPlain(line.slice(lastIndex, match.index), false);
      pushPlain(match[1], true);
      lastIndex = regex.lastIndex;
    }
    if (lastIndex < line.length) pushPlain(line.slice(lastIndex), false);
    return words;
  };

  // --- wrap tokens into lines that fit maxWidth, drawing as we go ---
  const drawRichText = (words, startX, maxWidth, fontSize, color, lineHeight) => {
    let lineWords = [];
    let lineWidth = 0;

    const flush = () => {
      if (lineWords.length === 0) return;
      ensureSpace(lineHeight);
      let x = startX;
      lineWords.forEach((w) => {
        const f = w.bold ? fontBold : fontNormal;
        page.drawText(w.text, { x, y, size: fontSize, font: f, color });
        x += f.widthOfTextAtSize(w.text + " ", fontSize);
      });
      y -= lineHeight;
      lineWords = [];
      lineWidth = 0;
    };

    words.forEach((w) => {
      const f = w.bold ? fontBold : fontNormal;
      const wWidth = f.widthOfTextAtSize(w.text, fontSize);
      const spaceWidth = f.widthOfTextAtSize(" ", fontSize);
      const projected = lineWidth + (lineWords.length ? spaceWidth : 0) + wWidth;
      if (projected > maxWidth && lineWords.length > 0) flush();
      lineWords.push(w);
      lineWidth = lineWords.length === 1 ? wWidth : lineWidth + spaceWidth + wWidth;
    });
    flush();
  };

  const drawHeading = (text, size) => {
    ensureSpace(size + 26);
    y -= 14; // space before
    drawRichText(tokenize(text), MARGIN, CONTENT_W, size, ACCENT, size + 4);
    y -= 2;
    page.drawLine({ start: { x: MARGIN, y }, end: { x: PAGE_W - MARGIN, y }, thickness: 0.75, color: RULE_LIGHT });
    y -= 14; // space after
  };

  const drawBullet = (content, indentLabel = "•") => {
    ensureSpace(16);
    page.drawText(indentLabel, { x: MARGIN, y, size: 11, font: fontBold, color: ACCENT });
    const indent = MARGIN + 16;
    drawRichText(tokenize(content), indent, CONTENT_W - 16, 11, TEXT_BODY, 16);
    y -= 4;
  };

  const drawParagraph = (line) => {
    drawRichText(tokenize(line), MARGIN, CONTENT_W, 11, TEXT_BODY, 16);
    y -= 4;
  };

  // ---------- Cover / title block ----------
  newPage();
  const lines = text.split("\n").map((l) => l.trim());
  let firstLineIsTitle = lines.length && lines[0].startsWith("## ");
  const titleText = firstLineIsTitle ? lines[0].replace(/^##\s*/, "") : "Investigation Report";
  const bodyLines = firstLineIsTitle ? lines.slice(1) : lines;

  y = PAGE_H - MARGIN - 30;
  page.drawText(titleText, { x: MARGIN, y, size: 22, font: fontBold, color: TEXT_DARK });
  y -= 22;
  page.drawText(`Generated ${new Date().toLocaleString()}`, { x: MARGIN, y, size: 9, font: fontItalic, color: TEXT_MUTED });
  y -= 16;
  page.drawLine({ start: { x: MARGIN, y }, end: { x: PAGE_W - MARGIN, y }, thickness: 1, color: ACCENT });
  y -= 26;

  // ---------- Body ----------
  for (let raw of bodyLines) {
    if (!raw) { y -= 6; continue; }

    if (raw.startsWith("## ")) {
      drawHeading(raw.replace(/^##\s*/, ""), 18);
    } else if (raw.startsWith("### ")) {
      drawHeading(raw.replace(/^###\s*/, ""), 14);
    } else if (/^[-*•]\s+/.test(raw)) {
      drawBullet(raw.replace(/^[-*•]\s+/, ""));
    } else if (/^\d+\.\s+/.test(raw)) {
      const m = raw.match(/^(\d+)\.\s+(.*)/);
      drawBullet(m[2], `${m[1]}.`);
    } else {
      drawParagraph(raw);
    }
  }

  // ---------- Footer pass (now that total page count is known) ----------
  const pages = pdfDoc.getPages();
  pages.forEach((p, i) => {
    p.drawLine({
      start: { x: MARGIN, y: 36 },
      end: { x: PAGE_W - MARGIN, y: 36 },
      thickness: 0.5,
      color: RULE_LIGHT,
    });
    const label = `Page ${i + 1} of ${pages.length}`;
    const labelWidth = fontNormal.widthOfTextAtSize(label, 9);
    p.drawText(label, { x: (PAGE_W - labelWidth) / 2, y: 22, size: 9, font: fontNormal, color: TEXT_MUTED });
  });

  // ---------- Save and download ----------
  const pdfBytes = await pdfDoc.save();
  const blob = new Blob([pdfBytes], { type: "application/pdf" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "report.pdf";
  link.click();
};





  return (
    <div>
      <Toaster position="top-right" reverseOrder={false} />

      <div className="background"></div>

      <div className="container">
        <div className="left">
          <h1>BlackBox</h1>
          <p>
            Handles post-flight or incident-based data investigation.
            Please upload FDR data.
          </p>
        </div>

        <div className="right">
          <button className="upload-btn" onClick={handleUploadClick}>
            <img src={UploadIcon} alt="Upload Icon" />
          </button>

          {/* Show file name when file is selected */}
          {fileData && (
            <p className="file-name">{fileData.name}</p>
          )}

          <input
            id="file-upload"
            type="file"
            accept=".csv"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />

          <button className="go-btn" onClick={handleGoClick}>
            Go
          </button>
        </div>
      </div>
    </div>
  );
}

export default BlackBoxPage;
