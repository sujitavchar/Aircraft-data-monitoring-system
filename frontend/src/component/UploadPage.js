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
      toast.success("✅ PDF downloaded on your computer!", { duration: 4000 });
      
    } catch (error) {
      toast.error("❌ Something went wrong. Please try again.", { duration: 4000 });
    }
  };

  

  const generatePDF = async (text) => {
  const pdfDoc = await PDFDocument.create();
  let page = pdfDoc.addPage();
  const { width, height } = page.getSize();
  const margin = 40;
  let y = height - margin;

  const fontNormal = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  const lines = text.split("\n");

  for (let line of lines) {
    line = line.trim();
    let font = fontNormal;
    let fontSize = 12;

    if (line.startsWith("## ")) {
      font = fontBold;
      fontSize = 18;
      line = line.replace("## ", "");
    } else if (line.startsWith("### ")) {
      font = fontBold;
      fontSize = 16;
      line = line.replace("### ", "");
    } else if (line.startsWith("•")) {
      font = fontNormal;
      fontSize = 12;
      line = line.replace(/^•\s*/, "• "); 
    } else if (line.includes("**")) {
      font = fontBold;
      fontSize = 12;
      line = line.replace(/\*\*(.*?)\*\*/g, "$1"); 
    }

    const lineHeight = fontSize + 4;

    const words = line.split(" ");
    let lineBuffer = "";
    for (let word of words) {
      const testLine = lineBuffer ? lineBuffer + " " + word : word;
      const testWidth = font.widthOfTextAtSize(testLine, fontSize);
      if (testWidth > width - margin * 2) {
        if (y - lineHeight < margin) {
          page = pdfDoc.addPage();
          y = height - margin;
        }
        page.drawText(lineBuffer, { x: margin, y, size: fontSize, font, color: rgb(0, 0, 0) });
        y -= lineHeight;
        lineBuffer = word;
      } else {
        lineBuffer = testLine;
      }
    }
    if (lineBuffer) {
      if (y - lineHeight < margin) {
        page = pdfDoc.addPage();
        y = height - margin;
      }
      page.drawText(lineBuffer, { x: margin, y, size: fontSize, font, color: rgb(0, 0, 0) });
      y -= lineHeight;
    }
  }

  // Save and download PDF
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
            A Blackbox is a system where only the inputs and outputs are visible,
            while the internal working remains hidden.
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
