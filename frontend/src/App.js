// import React from "react";
// import "./App.css";
// import Generator from "./Generator";
// import './Generator.css';
//
// function App() {
//   return (
//     <div className="App">
//       <div className="animated-background"></div>
//       <div className="navbar">
//         <h1>Welcome to Lottery AI Generator</h1>
//         <nav>
//           <a href="#history-statistic">History Statistic</a>
//           <a href="#history-numbers">History Numbers</a>
//           <button className="login-button">Log In</button>
//         </nav>
//       </div>
//         <button> className="Mega-Million"> Mega Million</button>
//           <div className="generator-wrapper">
//             <Generator />
//           </div>
//     </div>
//   );
// }
//
// export default App;

// import React, { useState } from "react";
// import "./App.css";
// import Generator from "./Generator";
// import './Generator.css';
//
// function App() {
//   const [showGenerator, setShowGenerator] = useState(false); // 控制显示状态
//
//   const handleMegaMillionClick = () => {
//     setShowGenerator(true); // 更新状态为显示生成器
//   };
//
//   return (
//     <div className="App">
//       <div className="animated-background"></div>
//       <div className="navbar">
//         <h1>Welcome to Lottery AI Generator</h1>
//         <nav>
//           <a href="#history-statistic">History Statistic</a>
//           <a href="#history-numbers">History Numbers</a>
//           <button className="login-button">Log In</button>
//         </nav>
//       </div>
//       {!showGenerator ? ( // 判断是否显示Mega Million按钮
//         <button
//           className="Mega-Million"
//           onClick={handleMegaMillionClick}
//         >
//           Mega Million
//         </button>
//       ) : (
//         <div className="generator-wrapper">
//           <Generator />
//         </div>
//       )}
//     </div>
//   );
// }
//
// export default App;


// import React, { useState } from "react";
// import "./App.css";
// import Generator from "./Generator";
// import './Generator.css';
// import PowerBallGenerator from "./PowerBall";
// import './PowerBall.css'
//
// function App() {
//  const [currentGenerator, setCurrentGenerator] = useState(null); // 当前显示的生成器
//
//   const handleMegaMillionClick = () => {
//     setCurrentGenerator("MegaMillion"); // 设置当前生成器为 Mega Million
//   };
//
//   const handlePowerBallClick = () => {
//     setCurrentGenerator("PowerBall"); // 设置当前生成器为 Power Ball
//   };
//
//   return (
//     <div className="App">
//       <div className="animated-background"></div>
//       <div className="navbar">
//         <h1>Welcome to Lottery AI Generator</h1>
//         <nav>
//           <a href="#history-statistic">History Statistic</a>
//           <a href="#history-numbers">History Numbers</a>
//           <button className="login-button">Log In</button>
//         </nav>
//       </div>
//        {currentGenerator === null ? (
//         <div className="button-container">
//           <button
//             className="Mega-Million"
//             onClick={handleMegaMillionClick}
//           >
//             Mega Million
//           </button>
//           <button
//             className="Power-Ball"
//             onClick={handlePowerBallClick}
//           >
//             Power Ball
//           </button>
//         </div>
//       ) : currentGenerator === "MegaMillion" ? (
//         <div className="generator-wrapper">
//           <Generator />
//         </div>
//       ) : (
//         <div className="generator-wrapper">
//           <PowerBallGenerator />
//         </div>
//       )}
//     </div>
//   );
// }
//
// export default App;


// import React, { useState } from "react";
// import "./App.css";
// import Generator from "./Generator";
// import './Generator.css';
// import PowerBallGenerator from "./PowerBall";
// import './PowerBall.css';
//
// function App() {
//   const [currentGenerator, setCurrentGenerator] = useState(null); // 当前显示的生成器
//
//   const handleMegaMillionClick = () => {
//     setCurrentGenerator("MegaMillion"); // 设置当前生成器为 Mega Million
//   };
//
//   const handlePowerBallClick = () => {
//     setCurrentGenerator("PowerBall"); // 设置当前生成器为 Power Ball
//   };
//
//   const handleBackClick = () => {
//     setCurrentGenerator(null); // 返回到主页面
//   };
//
//   return (
//     <div className="App">
//       <div className="animated-background"></div>
//       <div className="navbar">
//         <h1>Welcome to Lottery AI Generator</h1>
//         <nav>
//           <a href="#history-statistic">History Statistic</a>
//           <a href="#history-numbers">History Numbers</a>
//           <button className="login-button">Log In</button>
//         </nav>
//       </div>
//       {currentGenerator === null ? (
//         <div className="button-container">
//           <button
//             className="Mega-Million"
//             onClick={handleMegaMillionClick}
//           >
//             Mega Million
//           </button>
//           <button
//             className="Power-Ball"
//             onClick={handlePowerBallClick}
//           >
//             Power Ball
//           </button>
//         </div>
//       ) : currentGenerator === "MegaMillion" ? (
//         <div className="generator-wrapper">
//            <Generator />
//           <button className="back-button" onClick={handleBackClick}>
//             Back
//           </button>
//         </div>
//       ) : (
//         <div className="generator-wrapper">
//           <PowerBallGenerator />
//           <button className="back-button" onClick={handleBackClick}>
//             Back
//           </button>
//         </div>
//       )}
//     </div>
//   );
// }
//
// export default App;


import React, { useState } from "react";
import "./App.css";
import Generator from "./Generator";
import './Generator.css';

import PowerBallGenerator from "./powerball_generator";
import './powerball_generator.css'

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import HistoryNumbers from "./components/HistoryNumbers";
import HistoryStatistic from "./components/HistoryStatistic";

function App() {
  const [currentGenerator, setCurrentGenerator] = useState(null); // 当前显示的生成器

  const handleMegaMillionClick = () => {
    setCurrentGenerator("MegaMillion"); // 设置当前生成器为 Mega Million
  };

  const handlePowerBallClick = () => {
    setCurrentGenerator("PowerBall"); // 设置当前生成器为 Power Ball
  };

  const handleBackClick = () => {
    setCurrentGenerator(null); // 返回到主页面
  };

  return (
    <Router>
      <div className="App">
          <div id="tsparticles"></div>
        <div className="animated-background"></div>
        <div className="navbar">
          <h1>Welcome to Lottery AI Generator</h1>
          <nav>
            <Link to="/history-statistic">History Statistic</Link>
            <Link to="/history-numbers">History Numbers</Link>
            <button className="login-button">Log In</button>
          </nav>
        </div>
        <Routes>
          <Route
            path="/"
            element={
              currentGenerator === null ? (
                <div className="button-container">
                  <button
                    className="Mega-Million"
                    onClick={handleMegaMillionClick}
                  >
                    Mega Million
                  </button>
                  <button
                    className="Power-Ball"
                    onClick={handlePowerBallClick}
                  >
                    Power Ball
                  </button>
                </div>
              ) : currentGenerator === "MegaMillion" ? (
                <div className="generator-wrapper">
                  <Generator />
                  <button className="back-button" onClick={handleBackClick}>
                    Back
                  </button>
                </div>
              ) : (
                <div className="generator-wrapper">
                  <PowerBallGenerator />
                  <button className="back-button" onClick={handleBackClick}>
                    Back
                  </button>
                </div>
              )
            }
          />
          <Route path="/history-statistic" element={<HistoryStatistic />} />
          <Route path="/history-numbers" element={<HistoryNumbers />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;


