// import React, { useState, useEffect } from "react";
// import "./powerball_generator.css"
// import { fetchGeneratedNumbers } from "./api";
//
// function PowerBallGenerator() {
//   const [results, setResults] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [displayedResults, setDisplayedResults] = useState([]);
//   const [displayIndex, setDisplayIndex] = useState(0);
//   const [isDisplaying, setIsDisplaying] = useState(false); // 用于跟踪数字显示状态
//
//   const generateNumbers = async () => {
//     setLoading(true);
//     setError(null);
//     setResults([]);
//     setDisplayedResults([]);
//     setDisplayIndex(0);
//     setIsDisplaying(true);
//
//     try {
//       // 调用后端 API 获取 3 组数字
//       const data = await fetchGeneratedNumbers("power_ball",3); // 每次请求 3 组数字
//       console.log("Generated Numbers:", data); // 打印调试信息
//       setResults(data);
//     } catch (err) {
//       setError("Failed to fetch numbers. Please try again.");
//       setIsDisplaying(false); // 确保错误时恢复状态
//     } finally {
//       setLoading(false);
//     }
//   };
//
//   useEffect(() => {
//     if (results.length > 0 && displayIndex < results.length) {
//       const timer = setTimeout(() => {
//         setDisplayedResults((prev) => [...prev, results[displayIndex]]);
//         setDisplayIndex((prev) => prev + 1);
//       }, 1000); // 每 1 秒显示一行
//       return () => clearTimeout(timer);
//     } else if (results.length > 0 && displayIndex >= results.length) {
//       setIsDisplaying(false); // 数字全部显示完成
//     }
//   }, [results, displayIndex]);
//
//   return (
//     <div className="powerball-generator-container">
//       <h1>Play Power Ball</h1>
//       <div className="powerball-number-display-wrapper">
//         {displayedResults.map((result, index) => (
//           <div key={index} className="number-display-row fade-in">
//             {result.numbers.map((num, idx) => (
//               <div key={idx} className="number filled">
//                 {num}
//               </div>
//             ))}
//             <div className="power-ball filled">{result.power_ball}</div>
//           </div>
//         ))}
//       </div>
//       {error && <p className="error">{error}</p>}
//       <button
//         className={`powerball-generate-button ${loading || isDisplaying ? "disabled" : ""}`}
//         onClick={generateNumbers}
//         disabled={loading || isDisplaying} // 禁用按钮
//       >
//         {loading || isDisplaying ? "Generating..." : "Generate"}
//       </button>
//     </div>
//   );
// }
//
// export default PowerBallGenerator;



import React, { useState, useEffect } from "react";
import "./powerball_generator.css"
import { fetchGeneratedNumbers } from "./api";

function PowerBallGenerator() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [displayedResults, setDisplayedResults] = useState([]);
  const [displayIndex, setDisplayIndex] = useState(0);
  const [isDisplaying, setIsDisplaying] = useState(false);

  const generateNumbers = async () => {
    setLoading(true);
    setError(null);
    setResults([]);
    setDisplayedResults([]);
    setDisplayIndex(0);
    setIsDisplaying(true);

    try {
      const data = await fetchGeneratedNumbers("power_ball", 3);
      console.log("Generated Numbers:", data);
      setResults(data);
    } catch (err) {
      setError("Failed to fetch numbers. Please try again.");
      setIsDisplaying(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (results.length > 0 && displayIndex < results.length) {
      const timer = setTimeout(() => {
        setDisplayedResults((prev) => [...prev, results[displayIndex]]);
        setDisplayIndex((prev) => prev + 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (results.length > 0 && displayIndex >= results.length) {
      setIsDisplaying(false);
    }
  }, [results, displayIndex]);

  return (
    <div className="powerball-generator-container">
      <h1>Play Power Ball</h1>
      <div className="powerball-number-display-wrapper">
        {displayedResults.map((result, index) => (
          <div key={index} className="powerball-number-display-row powerball-fade-in">
            {result.numbers.map((num, idx) => (
              <div key={idx} className="powerball-number filled">{num}</div>
            ))}
            <div className="power-ball filled">{result.power_ball}</div>
          </div>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
      <button
        className={`powerball-generate-button ${loading || isDisplaying ? "disabled" : ""}`}
        onClick={generateNumbers}
        disabled={loading || isDisplaying}
      >
        {loading || isDisplaying ? "Generating..." : "Generate"}
      </button>
    </div>
  );
}

export default PowerBallGenerator;
