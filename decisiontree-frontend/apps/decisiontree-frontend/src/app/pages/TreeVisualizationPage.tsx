import React from 'react';
import DecisionTreeVisualization from '../components/DecisionTreeVisualization';

const TreeVisualizationPage: React.FC = () => {
  return (
    <div className="page-container">
      <h1>Decision Tree Visualization</h1>
      <p>
        This page displays a visualization of the trained decision tree model.
        You can see the decision nodes, prediction paths, and class distributions.
      </p>
      <DecisionTreeVisualization />
    </div>
  );
};

export default TreeVisualizationPage;
