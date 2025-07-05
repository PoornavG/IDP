import React, { useCallback, useState, useRef } from 'react';
import { Folder, Upload, Image as ImageIcon, Loader2, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface FolderUploadProps {
  onFolderUpload: (files: File[]) => void;
  isProcessing: boolean;
  processingProgress?: number;
  totalFiles?: number;
  processedFiles?: number;
}

const FolderUpload: React.FC<FolderUploadProps> = ({
  onFolderUpload,
  isProcessing,
  processingProgress = 0,
  totalFiles = 0,
  processedFiles = 0
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null); // Create a ref

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));

    if (imageFiles.length > 0) {
      onFolderUpload(imageFiles);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const items = Array.from(e.dataTransfer.items);
    const files: File[] = [];

    items.forEach(item => {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file && file.type.startsWith('image/')) {
          files.push(file);
        }
      }
    });

    if (files.length > 0) {
      onFolderUpload(files);
    }
  }, [onFolderUpload]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click(); // Programmatically click the input
  };

  return (
    <Card className="p-6 bg-white shadow-lg border border-emerald-100">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${isDragOver
          ? 'border-emerald-400 bg-emerald-50'
          : 'border-gray-300 hover:border-emerald-300 hover:bg-emerald-25'
          }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {!isProcessing ? (
          <div className="space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-emerald-100 to-teal-100 rounded-2xl flex items-center justify-center mx-auto">
              <Folder className="w-8 h-8 text-emerald-600" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Select Folder</h3>
              <p className="text-gray-600 mb-4">Choose a folder containing images to process all at once</p>
              <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500 mb-6">
                <span className="px-2 py-1 bg-gray-100 rounded">JPG</span>
                <span className="px-2 py-1 bg-gray-100 rounded">PNG</span>
                <span className="px-2 py-1 bg-gray-100 rounded">JPEG</span>
                <span className="text-gray-400">• Batch processing</span>
              </div>
              <input
                type="file"
                {...({ webkitdirectory: "" } as any)}
                multiple
                accept="image/*"
                onChange={handleFolderSelect}
                className="hidden"
                ref={fileInputRef} // Assign the ref
              />
              {/* Change from label to direct button click handler */}
              <Button
                className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white cursor-pointer"
                onClick={handleButtonClick} // Use the new click handler
              >
                <Folder className="w-4 h-4 mr-2" />
                Choose Folder
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto">
              <Loader2 className="w-6 h-6 text-emerald-600 animate-spin" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Processing Images</h3>
              <p className="text-sm text-gray-600 mb-4">
                Processing {processedFiles} of {totalFiles} images...
              </p>
              <Progress value={processingProgress} className="w-full" />
              <p className="text-xs text-gray-500 mt-2">{Math.round(processingProgress)}% complete</p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default FolderUpload;