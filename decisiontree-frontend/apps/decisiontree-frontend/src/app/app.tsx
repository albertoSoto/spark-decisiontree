import NxWelcome from './nx-welcome';
import { Route, Routes, Link } from 'react-router-dom';
import TreeVisualizationPage from './pages/TreeVisualizationPage';
import './app.css';

export function App() {
  return (
    <div>
      <header className="app-header">
        <h1>Spark Decision Tree</h1>
        <nav>
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/tree-visualization">Tree Visualization</Link>
            </li>
            <li>
              <Link to="/page-2">Page 2</Link>
            </li>
          </ul>
        </nav>
      </header>

      <main className="app-content">
        <Routes>
          <Route
            path="/"
            element={
              <div>
                <NxWelcome title="Spark Decision Tree" />
                <div className="home-content">
                  <h2>Welcome to the Spark Decision Tree Application</h2>
                  <p>
                    This application demonstrates decision tree classification using Apache Spark.
                    Navigate to the Tree Visualization page to see the decision tree structure.
                  </p>
                  <div className="cta-buttons">
                    <Link to="/tree-visualization" className="cta-button">
                      View Decision Tree
                    </Link>
                  </div>
                </div>
              </div>
            }
          />
          <Route
            path="/tree-visualization"
            element={<TreeVisualizationPage />}
          />
          <Route
            path="/page-2"
            element={
              <div>
                <Link to="/">Click here to go back to root page.</Link>
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}

export default App;
