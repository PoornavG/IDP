
import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Zap, Target, Clock, TrendingUp } from 'lucide-react';

interface Detection {
  animal: string;
  confidence: number;
  bbox: {
    x: number;
    y: number; 
    width: number;
    height: number;
  };
}

interface ResultsData {
  detections: Detection[];
  fallbackClassification: any;
  processingTime: number;
  adaptiveThreshold: number;
}

interface ResultsDisplayProps {
  image: string | null;
  results: ResultsData | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ image, results }) => {
  if (!image || !results) return null;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-emerald-500";
    if (confidence >= 0.7) return "bg-yellow-500";
    return "bg-orange-500";
  };

  const getConfidenceBadgeVariant = (confidence: number) => {
    if (confidence >= 0.9) return "default";
    if (confidence >= 0.7) return "secondary";
    return "destructive";
  };

  return (
    <div className="grid lg:grid-cols-3 gap-8">
      {/* Main Image with Detections */}
      <div className="lg:col-span-2">
        <Card className="p-6 bg-white shadow-lg border border-emerald-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detected Animals</h3>
          <div className="relative inline-block">
            <img 
              src={image} 
              alt="Detection results" 
              className="max-w-full h-auto rounded-lg"
            />
            {/* Bounding boxes overlay */}
            {results.detections.map((detection, index) => (
              <div
                key={index}
                className="absolute border-2 border-emerald-500 bg-emerald-500/10"
                style={{
                  left: `${(detection.bbox.x / 800) * 100}%`,
                  top: `${(detection.bbox.y / 600) * 100}%`,
                  width: `${(detection.bbox.width / 800) * 100}%`,
                  height: `${(detection.bbox.height / 600) * 100}%`,
                }}
              >
                <div className="absolute -top-8 left-0 bg-emerald-500 text-white px-2 py-1 rounded text-xs font-medium">
                  {detection.animal} ({Math.round(detection.confidence * 100)}%)
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Results Panel */}
      <div className="space-y-6">
        {/* Detection Summary */}
        <Card className="p-6 bg-white shadow-lg border border-emerald-100">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-5 h-5 text-emerald-600" />
            <h3 className="text-lg font-semibold text-gray-900">Detection Summary</h3>
          </div>
          <div className="space-y-4">
            {results.detections.map((detection, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                  <span className="font-medium text-gray-900 capitalize">{detection.animal}</span>
                </div>
                <Badge variant={getConfidenceBadgeVariant(detection.confidence)}>
                  {Math.round(detection.confidence * 100)}%
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        {/* Processing Stats */}
        <Card className="p-6 bg-white shadow-lg border border-emerald-100">
          <div className="flex items-center space-x-3 mb-4">
            <Zap className="w-5 h-5 text-emerald-600" />
            <h3 className="text-lg font-semibold text-gray-900">Processing Stats</h3>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">Processing Time</span>
              </div>
              <span className="font-medium text-gray-900">{results.processingTime}s</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">Adaptive Threshold</span>
              </div>
              <span className="font-medium text-gray-900">{results.adaptiveThreshold}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">Total Detections</span>
              </div>
              <span className="font-medium text-gray-900">{results.detections.length}</span>
            </div>
          </div>
        </Card>

        {/* Confidence Breakdown */}
        <Card className="p-6 bg-white shadow-lg border border-emerald-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Confidence Breakdown</h3>
          <div className="space-y-3">
            {results.detections.map((detection, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="capitalize font-medium text-gray-900">{detection.animal}</span>
                  <span className="text-gray-600">{Math.round(detection.confidence * 100)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${getConfidenceColor(detection.confidence)}`}
                    style={{ width: `${detection.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ResultsDisplay;
