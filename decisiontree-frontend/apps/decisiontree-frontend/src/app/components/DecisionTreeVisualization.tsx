import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Tree from 'react-d3-tree';
import './DecisionTreeVisualization.css';

interface TreeNode {
  name: string;
  rule?: string;
  className?: string;
  prediction?: number;
  classDistribution?: Record<string, number>;
  children?: TreeNode[];
}

interface TreeData {
  status: string;
  treeStructure?: {
    visualTree: TreeNode;
  };
  rawTree?: string;
  message?: string;
}

const DecisionTreeVisualization: React.FC = () => {
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [trained, setTrained] = useState<boolean>(false);

  const fetchTreeData = async () => {
    try {
      setLoading(true);
      const response = await axios.get<TreeData>('/api/dtree/tree-structure');
      if (response.data.status === 'success' && response.data.treeStructure?.visualTree) {
        setTreeData(response.data.treeStructure.visualTree);
        setTrained(true);
      } else {
        setError(response.data.message || 'Failed to load tree data');
        setTrained(false);
      }
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        setTrained(false);
        setError('Model not trained yet. Please train the model first.');
      } else {
        setError('Error loading tree data');
      }
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    try {
      setLoading(true);
      const response = await axios.post('/api/dtree/train');
      if (response.data.status === 'success') {
        await fetchTreeData();
      } else {
        setError(response.data.message || 'Failed to train model');
      }
    } catch (err) {
      setError('Error training model');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTreeData();
  }, []);

  // Custom node renderer to display decision tree nodes with details
  const renderCustomNodeElement = ({ nodeDatum }: { nodeDatum: any }) => {
    const isLeafNode = !nodeDatum.children || nodeDatum.children.length === 0;
    
    // Determine class colors for the distribution bars
    const classColors = {
      '0': '#8dd3c7', // Setosa - Light teal
      '1': '#fb8072', // Versicolor - Light red
      '2': '#bebada'  // Virginica - Light purple
    };
    
    return (
      <g>
        <rect
          width="220"
          height={isLeafNode ? 100 : 80}
          x="-110"
          y="-40"
          rx="5"
          ry="5"
          fill={isLeafNode ? '#f8f9d2' : '#e8f4f8'}
          stroke={isLeafNode ? '#b4c76e' : '#8dacc3'}
          strokeWidth="1.5"
        />
        
        {/* Node title */}
        <text x="0" y="-20" textAnchor="middle" fontWeight="bold">
          {nodeDatum.name}
        </text>
        
        {/* Decision rule or class prediction */}
        <text x="0" y="0" textAnchor="middle">
          {nodeDatum.rule}
        </text>
        
        {/* For leaf nodes, show class distribution */}
        {isLeafNode && nodeDatum.classDistribution && (
          <g>
            <text x="0" y="20" textAnchor="middle" fontSize="12">
              Class Distribution
            </text>
            
            {/* Distribution bars */}
            <g transform="translate(-75, 30)">
              {Object.entries(nodeDatum.classDistribution).map(([classKey, value], index) => {
                const barWidth = Math.max(5, (value as number) * 150);
                return (
                  <g key={classKey} transform={`translate(0, ${index * 15})`}>
                    <rect
                      width={barWidth}
                      height="10"
                      fill={classColors[classKey as keyof typeof classColors] || '#ccc'}
                    />
                    <text x={barWidth + 5} y="9" fontSize="10">
                      {`Class ${classKey}: ${Math.round((value as number) * 100)}%`}
                    </text>
                  </g>
                );
              })}
            </g>
          </g>
        )}
      </g>
    );
  };

  return (
    <div className="decision-tree-container">
      <h2>Decision Tree Visualization</h2>
      
      {!trained && !loading && (
        <div className="train-section">
          <p>The model has not been trained yet.</p>
          <button 
            className="train-button"
            onClick={trainModel}
            disabled={loading}
          >
            {loading ? 'Training...' : 'Train Model'}
          </button>
        </div>
      )}
      
      {error && <div className="error-message">{error}</div>}
      
      {loading && <div className="loading">Loading tree data...</div>}
      
      {treeData && (
        <div className="tree-wrapper">
          <Tree
            data={treeData}
            orientation="vertical"
            pathFunc="step"
            translate={{ x: 500, y: 50 }}
            renderCustomNodeElement={renderCustomNodeElement}
            separation={{ siblings: 2, nonSiblings: 2 }}
            nodeSize={{ x: 240, y: 120 }}
          />
        </div>
      )}
      
      <div className="legend">
        <h3>Legend</h3>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#e8f4f8' }}></div>
          <span>Decision Node</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#f8f9d2' }}></div>
          <span>Leaf Node (Classification)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#8dd3c7' }}></div>
          <span>Setosa</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#fb8072' }}></div>
          <span>Versicolor</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#bebada' }}></div>
          <span>Virginica</span>
        </div>
      </div>
    </div>
  );
};

export default DecisionTreeVisualization;
