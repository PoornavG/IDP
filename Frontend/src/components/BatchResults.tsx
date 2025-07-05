
import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Download, Eye, AlertCircle, CheckCircle } from 'lucide-react';

interface BatchDetection {
  animal: string;
  confidence: number;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface BatchResultItem {
  fileName: string;
  imageUrl: string;
  detections: BatchDetection[];
  processingTime: number;
  status: 'success' | 'error';
  error?: string;
}

interface BatchResultsProps {
  results: BatchResultItem[];
  onImageClick: (result: BatchResultItem) => void;
  onExportResults: () => void;
}

const BatchResults: React.FC<BatchResultsProps> = ({ results, onImageClick, onExportResults }) => {
  const successCount = results.filter(r => r.status === 'success').length;
  const errorCount = results.filter(r => r.status === 'error').length;
  const totalDetections = results.reduce((sum, r) => sum + (r.detections?.length || 0), 0);

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <Card className="p-6 bg-white shadow-lg border border-emerald-100">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Batch Processing Results</h3>
          <Button onClick={onExportResults} variant="outline" className="text-emerald-700 border-emerald-200">
            <Download className="w-4 h-4 mr-2" />
            Export Results
          </Button>
        </div>
        
        <div className="grid grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">{results.length}</div>
            <div className="text-sm text-gray-600">Total Images</div>
          </div>
          <div className="text-center p-3 bg-emerald-50 rounded-lg">
            <div className="text-2xl font-bold text-emerald-600">{successCount}</div>
            <div className="text-sm text-gray-600">Processed</div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{errorCount}</div>
            <div className="text-sm text-gray-600">Errors</div>
          </div>
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{totalDetections}</div>
            <div className="text-sm text-gray-600">Total Detections</div>
          </div>
        </div>
      </Card>

      {/* Results Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {results.map((result, index) => (
          <Card key={index} className="p-4 bg-white shadow border border-gray-100 hover:shadow-lg transition-shadow">
            <div className="space-y-3">
              {/* Image Preview */}
              <div className="relative">
                <img 
                  src={result.imageUrl} 
                  alt={result.fileName}
                  className="w-full h-40 object-cover rounded-lg"
                />
                <div className="absolute top-2 right-2">
                  {result.status === 'success' ? (
                    <div className="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center">
                      <CheckCircle className="w-4 h-4 text-white" />
                    </div>
                  ) : (
                    <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
                      <AlertCircle className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              </div>

              {/* File Info */}
              <div>
                <h4 className="font-medium text-gray-900 truncate" title={result.fileName}>
                  {result.fileName}
                </h4>
                <p className="text-xs text-gray-500">
                  {result.processingTime}s processing time
                </p>
              </div>

              {/* Detections or Error */}
              {result.status === 'success' ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-1">
                    {result.detections.slice(0, 3).map((detection, detIndex) => (
                      <Badge key={detIndex} variant="secondary" className="text-xs">
                        {detection.animal} ({Math.round(detection.confidence * 100)}%)
                      </Badge>
                    ))}
                    {result.detections.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{result.detections.length - 3} more
                      </Badge>
                    )}
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full text-emerald-700 border-emerald-200"
                    onClick={() => onImageClick(result)}
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    View Details
                  </Button>
                </div>
              ) : (
                <div className="p-2 bg-red-50 rounded text-xs text-red-700">
                  Error: {result.error || 'Processing failed'}
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default BatchResults;
