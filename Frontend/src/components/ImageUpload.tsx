import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImageUploadProps {
  onImageUpload: (file: File, previewUrl: string) => void;
  uploadedImage: string | null;
  isProcessing: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUpload, uploadedImage, isProcessing }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));

    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      onImageUpload(imageFile, url); // ✅ Pass File and URL
    }
  }, [onImageUpload]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      onImageUpload(file, url); // ✅ Pass File and URL
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-lg border border-emerald-100 overflow-hidden">
        {!uploadedImage ? (
          <div
            className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${isDragOver
              ? 'border-emerald-400 bg-emerald-50'
              : 'border-gray-300 hover:border-emerald-300 hover:bg-emerald-25'
              }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="space-y-4">
              <div className="w-16 h-16 bg-gradient-to-br from-emerald-100 to-teal-100 rounded-2xl flex items-center justify-center mx-auto">
                <Upload className="w-8 h-8 text-emerald-600" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Upload an image</h3>
                <p className="text-gray-600 mb-4">Drag and drop your image here, or click to browse</p>
                <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500 mb-6">
                  <span className="px-2 py-1 bg-gray-100 rounded">JPG</span>
                  <span className="px-2 py-1 bg-gray-100 rounded">PNG</span>
                  <span className="px-2 py-1 bg-gray-100 rounded">JPEG</span>
                  <span className="text-gray-400">• Max 10MB</span>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload">
                  <Button
                    type="button"
                    className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white cursor-pointer"
                    asChild
                  >
                    <span>
                      <ImageIcon className="w-4 h-4 mr-2" />
                      Choose Image
                    </span>
                  </Button>
                </label>
              </div>
            </div>
          </div>
        ) : (
          <div className="relative">
            <img
              src={uploadedImage}
              alt="Uploaded"
              className="w-full h-96 object-cover"
            />
            {isProcessing && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <div className="bg-white rounded-xl p-6 text-center space-y-4">
                  <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto">
                    <Loader2 className="w-6 h-6 text-emerald-600 animate-spin" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Processing Image</h3>
                    <p className="text-sm text-gray-600">AI is analyzing your image...</p>
                  </div>
                  <div className="w-48 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-emerald-500 to-teal-600 h-2 rounded-full animate-pulse"
                      style={{ width: '60%' }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;
