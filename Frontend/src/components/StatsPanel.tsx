
import React from 'react';
import { Card } from '@/components/ui/card';
import { BarChart3, Users, Zap, Award } from 'lucide-react';

const StatsPanel = () => {
  const stats = [
    {
      icon: Users,
      label: "Supported Species",
      value: "91",
      color: "text-emerald-600"
    },
    {
      icon: Zap,
      label: "Avg Processing",
      value: "2.1s",
      color: "text-blue-600"
    },
    {
      icon: Award,
      label: "Model Accuracy",
      value: "95.2%",
      color: "text-purple-600"
    },
    {
      icon: BarChart3,
      label: "Images Processed",
      value: "12.5K+",
      color: "text-orange-600"
    }
  ];

  const topAnimals = [
    { name: "Dog", count: 2847, percentage: 23 },
    { name: "Cat", count: 2156, percentage: 17 },
    { name: "Bird", count: 1834, percentage: 15 },
    { name: "Horse", count: 1245, percentage: 10 },
    { name: "Elephant", count: 987, percentage: 8 }
  ];

  return (
    <div className="space-y-6">
      {/* Key Statistics */}
      <Card className="p-6 bg-white shadow-lg border border-emerald-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Statistics</h3>
        <div className="grid grid-cols-2 gap-4">
          {stats.map((stat, index) => (
            <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
              <stat.icon className={`w-6 h-6 mx-auto mb-2 ${stat.color}`} />
              <div className="text-lg font-bold text-gray-900">{stat.value}</div>
              <div className="text-xs text-gray-600">{stat.label}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Top Detected Animals */}

    </div>
  );
};

export default StatsPanel;
