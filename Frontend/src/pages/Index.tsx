// src/pages/Index.tsx

import React, { useState } from "react";
import {
  Upload,
  Brain,
  BarChart3,
  Image as ImageIcon,
  Folder,
} from "lucide-react";
import ImageUpload from "../components/ImageUpload";
import FolderUpload from "../components/FolderUpload";
import ResultsDisplay from "../components/ResultsDisplay";
import BatchResults from "../components/BatchResults";
import StatsPanel from "../components/StatsPanel";
import { Button } from "@/components/ui/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

const BACKEND_URL = "http://localhost:8000";

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [showResults, setShowResults] = useState(false);

  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [showBatchResults, setShowBatchResults] = useState(false);
  const [selectedBatchResult, setSelectedBatchResult] = useState<any>(null);
  const [batchReportSummary, setBatchReportSummary] = useState<any>(null); // New state for report
  const handleImageUpload = async (file: File, previewUrl: string) => {
    setUploadedImage(previewUrl);
    setIsProcessing(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(`${BACKEND_URL}/process/single`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) throw new Error("Upload failed");

      const result = await response.json();
      setResults(result);
      setShowResults(true);
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFolderUpload = async (files: File[]) => {
    setIsBatchProcessing(true);
    setBatchFiles(files);
    setShowResults(false);
    setShowBatchResults(false);
    setBatchReportSummary(null); // Clear previous report

    const form = new FormData();
    files.forEach((f) => form.append("files", f));

    try {
      const res = await fetch(`${BACKEND_URL}/process/batch`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);

      // Expecting a dict with 'results' and 'report_summary'
      const { results, report_summary, output_folder_url } = await res.json();

      // The backend should now return full URLs, but if not, re-add mapping
      const withFullUrls = results.map((r: any) => ({
        ...r,
        // Assuming backend now sends full URLs from /media/ paths
        // imageUrl: r.imageUrl ? `${BACKEND_URL}${r.imageUrl}` : undefined,
        // detections: r.detections?.map((d: any) => ({
        //   ...d,
        //   image_url: `${BACKEND_URL}${d.image_url}`,
        // })) || [],
        // fallbackClassification: r.fallbackClassification
        //   ? {
        //       ...r.fallbackClassification,
        //       image_url: `${BACKEND_URL}${r.fallbackClassification.image_url}`,
        //     }
        //   : null,
      }));

      setBatchResults(withFullUrls);
      setBatchReportSummary(report_summary); // Store the report
      setShowBatchResults(true);

      // You might want to display output_folder_url somewhere or allow download
      console.log("Batch Output Folder:", output_folder_url);

    } catch (err) {
      console.error("Batch upload error:", err);
      alert("Failed to process batch");
    } finally {
      setIsBatchProcessing(false);
    }
  };

  const handleBatchImageClick = (result: any) => {
    setSelectedBatchResult(result);
    setUploadedImage(result.imageUrl || null);
    setResults({
      detections: result.detections,
      fallbackClassification: result.fallbackClassification,
      processingTime: result.processingTime,
      adaptiveThreshold: result.adaptiveThreshold,
    });
    setShowResults(true);
    setShowBatchResults(false);
  };

  const handleExportResults = () => {
    const exportData = {
      totalImages: batchResults.length,
      successfullyProcessed: batchResults.filter(
        (r) => r.status === "success"
      ).length,
      totalDetections: batchResults.reduce(
        (sum, r) => sum + (r.detections?.length || 0),
        0
      ),
      results: batchResults.map((r) => ({
        fileName: r.fileName,
        detections: r.detections,
        fallbackClassification: r.fallbackClassification,
        processingTime: r.processingTime,
        status: r.status,
        error: r.error,
      })),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_detection_results.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const resetInterface = () => {
    setUploadedImage(null);
    setResults(null);
    setShowResults(false);
    setBatchFiles([]);
    setBatchResults([]);
    setShowBatchResults(false);
    setSelectedBatchResult(null);
  };

  const showMainInterface = !showResults && !showBatchResults;

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50">
      <header className="bg-white/80 backdrop-blur-sm border-b border-emerald-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Animal AI Detector
                </h1>
                <p className="text-sm text-gray-600">
                  Advanced animal detection & classification
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <ImageIcon className="w-4 h-4" />
                <span>91 Species Supported</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <BarChart3 className="w-4 h-4" />
                <span>95.2% Accuracy</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {showMainInterface ? (
          <div className="space-y-8">
            <div className="text-center space-y-4 mb-12">
              <div className="inline-flex items-center space-x-2 bg-emerald-100 text-emerald-800 px-4 py-2 rounded-full text-sm font-medium">
                <Brain className="w-4 h-4" />
                <span>Powered by YOLO + EfficientNet</span>
              </div>
              <h2 className="text-4xl font-bold text-gray-900 leading-tight">
                Detect & Classify Animals with
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-600">
                  Advanced AI Technology
                </span>
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Upload any image containing animals or process entire folders to
                detect, locate, and classify them with high precision using
                state-of-the-art deep learning models.
              </p>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <Tabs defaultValue="single" className="space-y-6">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="single" className="flex items-center space-x-2">
                      <ImageIcon className="w-4 h-4" />
                      <span>Single Image</span>
                    </TabsTrigger>
                    <TabsTrigger value="batch" className="flex items-center space-x-2">
                      <Folder className="w-4 h-4" />
                      <span>Batch Processing</span>
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="single">
                    <ImageUpload
                      onImageUpload={handleImageUpload}
                      isProcessing={isProcessing}
                      uploadedImage={uploadedImage}
                    />

                  </TabsContent>

                  <TabsContent value="batch">
                    <FolderUpload
                      onFolderUpload={handleFolderUpload}
                      isProcessing={isBatchProcessing}
                    />
                  </TabsContent>
                </Tabs>
              </div>

              <div className="space-y-6">
                <div className="bg-white rounded-2xl p-6 shadow-lg border border-emerald-100">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    How it works
                  </h3>
                  <div className="space-y-4">
                    {[
                      "Object Detection",
                      "Classification",
                      "Adaptive Thresholding",
                    ].map((title, idx) => (
                      <div key={idx} className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center text-sm font-bold">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="font-medium text-gray-900">{title}</p>
                          <p className="text-sm text-gray-600">
                            {
                              [
                                "YOLO model locates animals in the image",
                                "EfficientNet identifies the species",
                                "Smart confidence adjustment based on image quality",
                              ][idx]
                            }
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <StatsPanel />
              </div>
            </div>
          </div>
        ) : showResults ? (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Detection Results</h2>
              <div className="flex space-x-2">
                {selectedBatchResult && (
                  <Button
                    onClick={() => {
                      setShowResults(false);
                      setShowBatchResults(true);
                      setSelectedBatchResult(null);
                    }}
                    variant="outline"
                    className="border-emerald-200 text-emerald-700 hover:bg-emerald-50"
                  >
                    <Folder className="w-4 h-4 mr-2" />
                    Back to Batch
                  </Button>
                )}
                <Button
                  onClick={resetInterface}
                  variant="outline"
                  className="border-emerald-200 text-emerald-700 hover:bg-emerald-50"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload New
                </Button>
              </div>
            </div>
            {uploadedImage && results && (
              <ResultsDisplay image={uploadedImage} results={results} />
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">
                Batch Processing Results
              </h2>
              <Button
                onClick={resetInterface}
                variant="outline"
                className="border-emerald-200 text-emerald-700 hover:bg-emerald-50"
              >
                <Upload className="w-4 h-4 mr-2" />
                Process New Batch
              </Button>
            </div>
            <BatchResults
              results={batchResults}
              onImageClick={handleBatchImageClick}
              onExportResults={handleExportResults}
            />
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
